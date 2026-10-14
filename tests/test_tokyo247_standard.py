import copy
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from a2b_lightvpr.config import DataConfig, EvalConfig, ExperimentConfig
from a2b_lightvpr.evaluation import resolve_database_domain, retrieval_metrics, search_topk_descriptors
from a2b_lightvpr.paths import DATASETS_ROOT
from a2b_lightvpr.references import models


def test_packaged_tokyo247_primary_query_selection():
    names, provenance = models._tokyo247_protocol_spec()
    assert len(names) == len(set(names)) == 315
    assert Counter(item["condition"] for item in provenance["query_metadata"]) == {"day": 105, "night": 105, "sunset": 105}
    assert provenance["database_images"] == 75984
    assert provenance["database_panoramas"] == 6332
    assert provenance["crops_per_panorama"] == 12
    assert provenance["positive_radius_m"] == 25


@pytest.fixture
def toy_tokyo(tmp_path, monkeypatch):
    names, original = models._tokyo247_protocol_spec()
    provenance = copy.deepcopy(original)
    provenance.update(database_images=24, database_panoramas=2)
    monkeypatch.setattr(models, "_tokyo247_protocol_spec", lambda: (names, provenance))
    queries = tmp_path / "queries"
    queries.mkdir()
    for item in provenance["query_metadata"]:
        name = item["image_name"]
        (queries / name).touch()
        values = [name, "0", "0", "F", "N", {"day": "D", "night": "N", "sunset": "S"}[item["condition"]],
                  "url", str(item["utm_e"]), str(item["utm_n"])]
        with (queries / Path(name).with_suffix(".csv")).open("w", newline="") as handle:
            csv.writer(handle).writerow(values)
    first = provenance["query_metadata"][0]
    folder = tmp_path / "database_gsv_vga" / "00001" / f"{first['utm_e']}_{first['utm_n']}"
    folder.mkdir(parents=True)
    for panorama in ("pano_a", "pano_b"):
        for angle in range(0, 360, 30):
            (folder / f"{panorama}_012_{angle:03d}.png").touch()
    inventory = sorted(path.relative_to(tmp_path / "database_gsv_vga").as_posix()
                       for path in folder.glob("*.png"))
    provenance["local_validation"]["database_relative_inventory_sha256"] = hashlib.sha256(
        ("\n".join(inventory) + "\n").encode()).hexdigest()
    return tmp_path, names, provenance, folder


def test_tokyo247_rejects_unverified_database_pose_change(toy_tokyo):
    root, _, _, folder = toy_tokyo
    x, y = map(float, folder.name.split("_"))
    folder.rename(folder.with_name(f"{x + 1}_{y}"))
    with pytest.raises(ValueError, match="inventory"):
        models._tokyo247_rows(root, "tokyo247")


def test_primary_tokyo247_selects_exact_queries_and_distinct_same_utm_panoramas(toy_tokyo):
    root, names, provenance, _ = toy_tokyo
    database, queries = models._tokyo247_rows(root, "tokyo247")
    assert [Path(row.source_path).name for row in queries] == names
    assert sorted(Counter(row.retrieval_group for row in database).values()) == [12, 12]
    assert len({row.retrieval_group for row in database}) == 2
    for condition in ("night", "day", "sunset"):
        _, selected = models._tokyo247_rows(root, f"tokyo247_{condition}")
        assert len(selected) == 105
        assert {row.condition for row in selected} == {condition}
    expected = provenance["query_metadata"][0]
    assert (queries[0].easting, queries[0].northing) == (expected["utm_e"], expected["utm_n"])
    config = ExperimentConfig(DataConfig(tokyo_root=str(root)))
    assert len(models.suite_rows(config, "tokyo247_night")[1]) == 105
    with pytest.raises(ValueError, match="test-only"):
        models.suite_rows(config, "tokyo247_night", "val")


@pytest.mark.parametrize("change,match", [
    ("duplicate", "Duplicate"), ("missing_metadata", "metadata missing"),
    ("missing_image", "image missing"), ("condition", "condition"),
    ("nan", "coordinates"), ("wrong_utm", "coordinates"),
    ("missing_crop", "75984 crops"), ("wrong_angle", "crop angle"),
])
def test_tokyo247_fails_closed_for_invalid_selection(toy_tokyo, change, match):
    root, names, _, folder = toy_tokyo
    metadata = root / "queries" / Path(names[0]).with_suffix(".csv")
    if change == "duplicate":
        (root / "queries/duplicate.csv").write_bytes(metadata.read_bytes())
    elif change == "missing_metadata":
        metadata.unlink()
    elif change == "missing_image":
        (root / "queries" / names[0]).unlink()
    elif change in {"condition", "nan", "wrong_utm"}:
        with metadata.open(newline="") as handle:
            row = next(csv.reader(handle))
        if change == "condition":
            row[5] = "N" if row[5] != "N" else "D"
        else:
            row[7] = "nan" if change == "nan" else str(float(row[7]) + 100)
        with metadata.open("w", newline="") as handle:
            csv.writer(handle).writerow(row)
    elif change == "missing_crop":
        next(folder.glob("*.png")).unlink()
    else:
        next(folder.glob("*.png")).rename(folder / "pano_a_012_001.png")
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        models._tokyo247_rows(root, "tokyo247_night")


@pytest.mark.parametrize("change,match", [("duplicate", "315 unique"), ("hash", "SHA256"), ("counts", "105 Queries")])
def test_primary_protocol_assets_are_validated(tmp_path, monkeypatch, change, match):
    names, provenance = models._tokyo247_protocol_spec()
    path = tmp_path / "tokyo247_query_names.txt"
    path.write_text("\n".join(names) + "\n")
    if change == "duplicate":
        names[1] = names[0]
        path.write_text("\n".join(names) + "\n")
    elif change == "hash":
        provenance["query_names_sha256"] = "0" * 64
    else:
        provenance["query_metadata"][0]["condition"] = "night"
    (tmp_path / "tokyo247_provenance.json").write_text(json.dumps(provenance))
    monkeypatch.setattr(models, "TOKYO247_PROTOCOL_DIR", tmp_path)
    with pytest.raises(ValueError, match=match):
        models._tokyo247_protocol_spec()


def test_panorama_nms_keeps_best_crop_and_restores_unique_topk():
    database = np.asarray([[1, 0], [.99, .1], [.98, .2], [.9, .3], [.8, .4]], dtype=np.float32)
    queries = np.asarray([[1, 0]], dtype=np.float32)
    groups = ["pano_a", "pano_a", "pano_a", "pano_b", "pano_c"]
    plain_scores, plain = search_topk_descriptors(database, queries, 3)
    scores, selected = search_topk_descriptors(database, queries, 3, database_groups=groups)
    assert plain.tolist() == [[0, 1, 2]]
    assert selected.tolist() == [[0, 3, 4]]
    assert scores.tolist()[0] == pytest.approx([1, .9, .8])
    positives = [np.asarray([3])]
    assert retrieval_metrics(database, queries, positives, (1, 2, 3))["recalls"]["2"] == 0
    metrics = retrieval_metrics(database, queries, positives, (1, 2, 3), database_groups=groups)
    assert metrics["recalls"] == {"1": 0, "2": 1, "3": 1}
    assert metrics["mrr"] == .5
    assert plain_scores.shape == scores.shape


def test_panorama_nms_matches_primary_netvlad_index_division():
    # Primary NetVLAD dbTokyo247.m / Patch-NetVLAD feature_match.py search K*12
    # candidates, then stably keep the first crop from each index//12 panorama.
    rng = np.random.default_rng(9)
    database, queries = rng.normal(size=(72, 16)).astype(np.float32), rng.normal(size=(3, 16)).astype(np.float32)
    _, candidates = search_topk_descriptors(database, queries, 3 * 12)
    expected = []
    for prediction in candidates:
        _, first = np.unique(prediction // 12, return_index=True)
        expected.append(prediction[np.sort(first)][:3])
    _, actual = search_topk_descriptors(database, queries, 3, database_groups=np.arange(72) // 12)
    np.testing.assert_array_equal(actual, expected)


def test_panorama_nms_numpy_fallback_and_group_validation(monkeypatch):
    import builtins

    original_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda name, *args, **kwargs: (
        (_ for _ in ()).throw(ImportError("test without faiss")) if name == "faiss" else original_import(name, *args, **kwargs)))
    database = np.asarray([[1, 0], [.99, 0], [.9, .1]], dtype=np.float32)
    query = np.asarray([[1, 0]], dtype=np.float32)
    _, selected = search_topk_descriptors(database, query, 10, database_groups=["a", "a", "b"])
    assert selected.tolist() == [[0, 2]]
    for groups in (["a"], ["a", None, "b"], ["a", "", "b"]):
        with pytest.raises(ValueError, match="groups|group IDs"):
            search_topk_descriptors(database, query, 2, database_groups=groups)


def test_group_ranking_changes_only_grouped_protocol_fingerprint():
    database = [models.SuiteRow("db", 0, 0)]
    queries = [models.SuiteRow("query", 0, 0)]
    digest = hashlib.sha256(b"radius=25.0\0yaw=None\n")
    for role, rows in (("database", database), ("queries", queries)):
        for row in rows:
            digest.update("\0".join((role, row.source_path, str(row.easting), str(row.northing), str(row.yaw_deg), "True")).encode())
            digest.update(b"\n")
    original = models.query_protocol_fingerprint(database, queries, 25.0, None)
    assert original == digest.hexdigest()
    first = models.query_protocol_fingerprint([models.SuiteRow("db", 0, 0, retrieval_group="a")], queries, 25.0, None)
    second = models.query_protocol_fingerprint([models.SuiteRow("db", 0, 0, retrieval_group="b")], queries, 25.0, None)
    assert len({original, first, second}) == 3


def test_fake_database_preserves_panorama_identity(tmp_path):
    original, fake = tmp_path / "source.png", tmp_path / "fake.png"
    original.touch()
    fake.touch()
    mapping = tmp_path / "mapping.csv"
    with mapping.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_path", "generated_path"])
        writer.writerow([str(original), str(fake)])
    rows, _ = resolve_database_domain(
        [models.SuiteRow(str(original), 3, 4, retrieval_group="panorama-a")],
        EvalConfig(database_domain="fake_night", fake_database_manifest=str(mapping)), "test")
    assert rows[0].source_path == str(fake)
    assert rows[0].retrieval_group == "panorama-a"
    assert (rows[0].easting, rows[0].northing) == (3, 4)


def test_dual_encoder_uses_panorama_groups_and_reports_primary_metadata(monkeypatch):
    database = [models.SuiteRow(str(i), 0 if i == 3 else 100, 0, retrieval_group=group)
                for i, group in enumerate(("a", "a", "a", "b", "c"))]
    queries = [models.SuiteRow("query", 0, 0)]
    db = np.asarray([[1, 0], [.99, .1], [.98, .2], [.9, .3], [.8, .4]], dtype=np.float32)
    monkeypatch.setattr(models, "extract_descriptors", lambda model, rows, *args: db if rows is database else np.asarray([[1, 0]], dtype=np.float32))
    result = models.evaluate_dual_encoder(None, None, database, queries, None, 25, None, (1, 2), torch.device("cpu"), 1, 0)
    assert result["recalls"] == {"1": 0, "2": 1}
    metadata = models.suite_protocol_metadata(None, "tokyo247_night")
    assert metadata["official_query_count"] == 105
    assert metadata["ranking_rule"] == "best_crop_per_panorama_v1"
    assert metadata["official_image_bytes_verified"] is False
    assert metadata["query_names_sha256"]
    assert metadata["positive_set_boundary_audit"]["primary_vs_local_database_positive_sets_equal"] is True


@pytest.mark.skipif(not (DATASETS_ROOT / "tokyo247").is_dir(), reason="Local Tokyo dataset unavailable")
def test_local_tokyo247_public_selection_golden_counts():
    database, queries = models._tokyo247_rows(DATASETS_ROOT / "tokyo247", "tokyo247")
    assert (len(database), len(queries)) == (75984, 315)
    assert len({row.retrieval_group for row in database}) == 6332
    assert Counter(row.condition for row in queries) == {"night": 105, "day": 105, "sunset": 105}
    assert set(Counter(row.retrieval_group for row in database).values()) == {12}
