import csv
import json
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from a2b_lightvpr import robotcar_seasons as seasons, trained_suite
from a2b_lightvpr.references import suite as reference_suite
from a2b_lightvpr.references.models import suite_rows
from a2b_lightvpr.cli import _parser, main
from a2b_lightvpr.config import DataConfig, ExperimentConfig
from a2b_lightvpr.evaluation import search_topk_descriptors


@pytest.fixture
def public_images(tmp_path):
    root = tmp_path / "RobotCar-Seasons"
    for condition, samples in {
        "overcast-reference": [("1", "red"), ("2", "blue")],
        "night": [("10", "red"), ("12", "blue")],
        "night-rain": [("11", "blue")],
    }.items():
        for camera in seasons.CAMERAS:
            folder = root / "images" / condition / camera
            folder.mkdir(parents=True)
            for timestamp, color in samples:
                Image.new("RGB", (14, 14), color).save(folder / f"{timestamp}.jpg")
    (root / "robotcar_v2_train.txt").write_text(
        "night/rear/12.jpg 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n"
    )
    (root / "robotcar_v2_test.txt").write_text("rear/10.png\nrear/11.png\n")
    return root


def test_v2_uses_exact_official_test_membership_and_preserves_camera_identity(public_images):
    database, queries = seasons.robotcar_seasons_rows(public_images)
    assert len(database) == 2
    assert {row.image_id for row in queries} == {"night/rear/10.jpg", "night-rain/rear/11.jpg"}
    assert all(row.camera == "rear" for row in (*database, *queries))
    assert queries[0].easting is None and not queries[0].ground_truth_available
    database, queries = seasons.robotcar_seasons_rows(public_images, protocol="original")
    assert len(database) == 6 and len(queries) == 9
    assert len({row.image_id for row in queries}) == 9  # Same timestamp in three cameras stays distinct.


@pytest.mark.parametrize("failure", ["missing", "ambiguous", "train_overlap", "duplicate", "empty_train"])
def test_v2_rejects_unreliable_correspondence_or_test_leakage(public_images, failure):
    if failure == "missing":
        (public_images / "images/night/rear/10.jpg").unlink()
    elif failure == "ambiguous":
        Image.new("RGB", (14, 14)).save(public_images / "images/night-rain/rear/10.jpg")
    elif failure == "train_overlap":
        with (public_images / "robotcar_v2_test.txt").open("a") as handle:
            handle.write("rear/12.png\n")
    elif failure == "empty_train":
        (public_images / "robotcar_v2_train.txt").write_text("")
    else:
        with (public_images / "robotcar_v2_test.txt").open("a") as handle:
            handle.write("rear/10.png\n")
    with pytest.raises(ValueError, match="exact|overlapping|empty"):
        seasons.robotcar_seasons_rows(public_images)


def test_audit_reports_image_readiness_separately_from_recall_and_missing_metadata(public_images, tmp_path):
    output = tmp_path / "audit.json"
    report = seasons.audit_robotcar_seasons(public_images, output=output)
    assert report["image_retrieval_ready"] and not report["local_recall_ready"]
    assert report["database"] == 2 and report["queries"] == 2
    assert len(report["split_metadata_sha256"]["robotcar_v2_test.txt"]) == 64
    assert json.loads(output.read_text()) == report
    (public_images / "robotcar_v2_test.txt").unlink()
    assert not seasons.audit_robotcar_seasons(public_images)["image_retrieval_ready"]
    assert seasons.audit_robotcar_seasons(public_images, protocol="original")["image_retrieval_ready"]
    with pytest.raises(ValueError, match="overwrite"):
        seasons.audit_robotcar_seasons(public_images, output=public_images / "images/report.json")


@pytest.mark.parametrize("kwargs", [{"conditions": ()}, {"cameras": ("front",)},
                                    {"conditions": ("night", "night")}, {"protocol": "invented"}])
def test_unknown_empty_or_duplicate_selection_fails(public_images, kwargs):
    with pytest.raises(ValueError):
        seasons.robotcar_seasons_rows(public_images, **kwargs)


class ColorModel(torch.nn.Module):
    descriptor_dim = 3

    def encode(self, images):
        mean = images.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
        std = images.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
        return F.normalize((images * std + mean).mean((2, 3)), dim=1)


def retrieval_config(tmp_path, monkeypatch, public_images):
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "manifest.csv").write_text("image_id\nfixture\n")
    config = ExperimentConfig(DataConfig(robotcar_seasons_root=str(public_images), generated_root=str(generated)))
    config = replace(config, vpr=replace(config.vpr, output_dir=str(tmp_path / "run"), descriptor_dim=3,
                                       image_height=14, image_width=14, workers=0, batch_size=2))
    checkpoint = tmp_path / "model_best.pt"
    checkpoint.write_bytes(b"fixture-checkpoint")
    monkeypatch.setattr(trained_suite, "load_vpr_model", lambda *args: (ColorModel(), {"epoch": 1}, checkpoint))
    monkeypatch.setattr(trained_suite, "benchmark_model", lambda *args, **kw: {
        "parameters": 0, "peak_cuda_mib": None, "latency_ms": {"p50": 1., "p95": 1.}})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return config, checkpoint


def test_public_seasons_uses_unified_suite_and_null_unmeasured_metrics(public_images, tmp_path, monkeypatch):
    config, checkpoint = retrieval_config(tmp_path, monkeypatch, public_images)
    output = trained_suite.evaluate_trained_suite(config, checkpoint, ("robotcar_seasons_night",))
    payload = json.loads(output.read_text())["datasets"]["robotcar_seasons_night"]
    assert payload["retrieval_top_k"] == 2 and payload["queries"] == 2
    assert payload["public_protocol"] == "robotcar_seasons_v2"
    assert payload["ground_truth_available"] is False
    assert payload["recalls"] is None and payload["mrr"] is None and payload["valid_queries"] is None
    by_query = {row["query_id"]: row for row in payload["retrieval_predictions"]}
    assert by_query["night/rear/10.jpg"]["matches"][0]["database_id"] == "overcast-reference/rear/1.jpg"
    assert by_query["night-rain/rear/11.jpg"]["matches"][0]["database_id"] == "overcast-reference/rear/2.jpg"
    assert len(payload["checkpoint_sha256"]) == 64
    assert "robotcar_seasons_night" in trained_suite.expand_suite_datasets(("all",))
    assert trained_suite.evaluate_trained_suite(config, checkpoint, ("robotcar_seasons_night",)) == output


def fake_config(config, public_images, tmp_path):
    database, queries = seasons.robotcar_seasons_rows(public_images)
    manifest = tmp_path / "fake_mapping.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "generated_path", "split"])
        writer.writeheader()
        for i, row in enumerate(database):
            fake = tmp_path / f"fake_{i}.jpg"
            Image.new("RGB", (14, 14), "blue" if i == 0 else "red").save(fake)
            writer.writerow({"source_path": row.source_path, "generated_path": str(fake), "split": "test"})
    config = replace(config, evaluation=replace(config.evaluation, database_domain="fake_night",
                                                fake_database_manifest=str(manifest)))
    return config, database, queries


def test_unified_fake_database_keeps_public_query_and_day_identity(public_images, tmp_path, monkeypatch):
    config, checkpoint = retrieval_config(tmp_path, monkeypatch, public_images)
    config, database, queries = fake_config(config, public_images, tmp_path)
    output = trained_suite.evaluate_trained_suite(config, checkpoint, ("robotcar_seasons_night",))
    payload = json.loads(output.read_text())["datasets"]["robotcar_seasons_night"]
    red_query = next(row for row in payload["retrieval_predictions"] if row["query_id"] == "night/rear/10.jpg")
    assert red_query["query_path"] == next(row.source_path for row in queries if row.image_id == red_query["query_id"])
    assert red_query["matches"][0]["database_id"] == "overcast-reference/rear/2.jpg"
    assert red_query["matches"][0]["database_path"] == str(tmp_path / "fake_1.jpg")
    assert payload["recalls"] is None and payload["database_domain"] == "fake_night"
    assert len(payload["database_manifest_sha256"]) == 64


@pytest.mark.parametrize("fake", [False, True])
def test_reference_suite_public_protocol_and_fake_database(public_images, tmp_path, monkeypatch, fake):
    config, checkpoint = retrieval_config(tmp_path, monkeypatch, public_images)
    if fake:
        config, _, _ = fake_config(config, public_images, tmp_path)
    registry = tmp_path / "registry.yaml"
    registry.write_text(f"reference_models:\n  fixture:\n    loader: fixture\n    descriptor_dim: 3\n    native_size: [14, 14]\n    checkpoint: {checkpoint}\n")
    monkeypatch.setattr(reference_suite, "build_reference_model", lambda *a: ColorModel())
    output = reference_suite.evaluate_reference_suite(config, registry, model_name="fixture",
        database_encoder="adapted", datasets=("robotcar_seasons_night",), output_dir=tmp_path / "reference")
    payload = json.loads(output.read_text())["datasets"]["robotcar_seasons_night"]
    assert payload["recalls"] is None and len(payload["retrieval_predictions"]) == 2
    assert payload["database_domain"] == ("fake_night" if fake else "day")
    assert reference_suite.evaluate_reference_suite(config, registry, model_name="fixture",
        database_encoder="adapted", datasets=("robotcar_seasons_night",), output_dir=tmp_path / "reference") == output


def test_changed_public_metadata_cannot_reuse_cached_rankings(public_images, tmp_path, monkeypatch):
    config, checkpoint = retrieval_config(tmp_path, monkeypatch, public_images)
    output = trained_suite.evaluate_trained_suite(config, checkpoint, ("robotcar_seasons_night",))
    first = json.loads(output.read_text())["datasets"]["robotcar_seasons_night"]
    metadata = public_images / "robotcar_v2_train.txt"
    metadata.write_text(metadata.read_text().replace("1 0 0 0", "1 0 0 .01", 1))
    trained_suite.evaluate_trained_suite(config, checkpoint, ("robotcar_seasons_night",))
    updated = json.loads(output.read_text())["datasets"]["robotcar_seasons_night"]
    assert updated["split_metadata_sha256"] != first["split_metadata_sha256"]


def test_public_test_is_rejected_for_training_validation(public_images):
    config = ExperimentConfig(DataConfig(robotcar_seasons_root=str(public_images)))
    with pytest.raises(ValueError, match="training validation"):
        suite_rows(config, "robotcar_seasons_night", "val")


def test_chunked_numpy_search_matches_exact_order_and_clips_k(monkeypatch):
    monkeypatch.setitem(sys.modules, "faiss", None)
    database = np.asarray([[1., 0.], [0., 1.], [-1., 0.]])
    queries = np.tile(np.asarray([[.8, .2], [-.8, .2]]), (65, 1))
    scores, indices = search_topk_descriptors(database, queries, 10)
    all_scores = queries.astype(np.float32) @ database.astype(np.float32).T
    expected = np.argsort(all_scores, axis=1)[:, ::-1]
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_allclose(scores, np.take_along_axis(all_scores, expected, axis=1))
    with pytest.raises(ValueError, match="finite"):
        search_topk_descriptors(database, np.asarray([[np.nan, 1.]]), 1)


def test_public_audit_cli_does_not_require_legacy_config(public_images, capsys):
    main(["audit-robotcar-seasons", "--data-root", str(public_images)])
    assert json.loads(capsys.readouterr().out)["queries"] == 2
    args = _parser().parse_args(["evaluate-trained-suite", "--config", "model.yaml", "--checkpoint", "model.pt", "--datasets", "robotcar_seasons_night"])
    assert args.datasets == ["robotcar_seasons_night"]
