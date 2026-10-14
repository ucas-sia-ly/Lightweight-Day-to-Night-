import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from a2b_lightvpr import trained_suite
from a2b_lightvpr.checkpointing import file_sha256
from a2b_lightvpr.config import DataConfig, EvalConfig, ExperimentConfig, VPRConfig
from a2b_lightvpr.references.models import SuiteRow


def _write_fake_manifest(path, sources, outputs):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "generated_path", "split"])
        writer.writeheader()
        # Deliberately use CSV order opposite to the retrieval database.
        writer.writerows({"source_path": source.name, "generated_path": output.name, "split": "test"}
                         for source, output in reversed(list(zip(sources, outputs))))


def _suite_fixture(tmp_path, monkeypatch, mode="day_aug", recalls=(1, 5, 10)):
    day_paths = [tmp_path / f"db_day{index}.png" for index in range(2)]
    query_path = tmp_path / "real_night_query.png"
    for path in day_paths + [query_path]:
        Image.new("RGB", (28, 28), (90, 110, 130)).save(path)
    database = [SuiteRow(str(path), float(index * 100), 0., 0.)
                for index, path in enumerate(day_paths)]
    queries = [SuiteRow(str(query_path), 0., 0., 0.)]
    source_manifest = tmp_path / "source_train.csv"
    source_manifest.write_text(
        "image_id,dataset,split,place_id,source_path\nsource0,msls,train,place0," + str(day_paths[0]) + "\n",
        encoding="utf-8",
    )
    backbone = tmp_path / "dinov2_small_pretrained.pt"
    backbone.write_bytes(b"pretrained-backbone-fixture")
    checkpoint = tmp_path / "model_best.pt"
    checkpoint.write_bytes(b"trained-stable-night-fixture")
    config = ExperimentConfig(
        DataConfig(vpr_source_manifest=str(source_manifest),
                   generated_root=str(tmp_path / "absent_generated")),
        vpr=VPRConfig(model_family="stable_night_salad", aggregator="salad", descriptor_dim=544,
                      image_height=322, image_width=322, train_domains=mode,
                      backbone_name="dinov2_vits14_reg", backbone_checkpoint=str(backbone),
                      output_dir=str(tmp_path / "run"), workers=0),
        evaluation=EvalConfig(recalls=recalls),
    )
    observed = {"database": database, "queries": queries, "evaluations": [], "loads": []}

    class Model(torch.nn.Module):
        descriptor_dim = 544

    def load(config, path):
        observed["loads"].append(config)
        return Model(), {"epoch": 3}, path

    def evaluate(*args, **kwargs):
        observed["evaluations"].append((list(args[2]), list(args[3]), tuple(args[7])))
        score = 0.1 * len(observed["evaluations"])
        return {"recalls": {str(k): score for k in args[7]}, "mrr": score, "valid_queries": 1}

    monkeypatch.setattr(trained_suite, "load_vpr_model", load)
    monkeypatch.setattr(trained_suite, "suite_rows", lambda *args: (observed["database"], observed["queries"]))
    monkeypatch.setattr(trained_suite, "evaluate_dual_encoder", evaluate)
    monkeypatch.setattr(trained_suite, "benchmark_model", lambda *args, **kwargs: {
        "parameters": 10, "peak_cuda_mib": None, "latency_ms": {"p50": 1.0, "p95": 2.0},
    })
    return config, checkpoint, observed


@pytest.mark.parametrize("mode", ["day_only", "day_aug"])
def test_source_only_suite_requires_no_generated_manifest_and_reports_dino_initialization(tmp_path, monkeypatch, mode):
    config, checkpoint, observed = _suite_fixture(tmp_path, monkeypatch, mode)
    summary = trained_suite.evaluate_trained_suite(config, checkpoint, ("svox_night",))
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert not (tmp_path / "absent_generated").exists()
    assert payload["manifest_sha256"] == file_sha256(config.data.vpr_source_manifest)
    assert payload["initialization"] == config.vpr.backbone_checkpoint
    assert payload["initial_checkpoint_sha256"] == file_sha256(config.vpr.backbone_checkpoint)
    assert payload["database_domain"] == "day"
    assert payload["datasets"]["svox_night"]["initialization"] == config.vpr.backbone_checkpoint
    assert trained_suite.evaluate_trained_suite(config, checkpoint, ("svox_night",)) == summary
    assert len(observed["loads"]) == len(observed["evaluations"]) == 1


def test_fake_suite_changes_only_database_paths_and_keeps_separate_hashed_results(tmp_path, monkeypatch):
    config, checkpoint, observed = _suite_fixture(tmp_path, monkeypatch)
    day_summary = trained_suite.evaluate_trained_suite(config, checkpoint, ("svox_night",))
    day_payload = day_summary.read_bytes()
    original_database, original_queries = list(observed["database"]), list(observed["queries"])
    fake_paths = [tmp_path / f"db_fake{index}.png" for index in range(2)]
    for path in fake_paths:
        Image.new("RGB", (28, 28), (20, 30, 40)).save(path)
    manifest = tmp_path / "fake_database.csv"
    _write_fake_manifest(manifest, [Path(row.source_path) for row in original_database], fake_paths)
    fake_config = replace(config, evaluation=replace(config.evaluation,
        database_domain="fake_night", fake_database_manifest=str(manifest)))
    fake_summary = trained_suite.evaluate_trained_suite(fake_config, checkpoint, ("svox_night",))
    payload = json.loads(fake_summary.read_text(encoding="utf-8"))
    mapped_database, queries, _ = observed["evaluations"][-1]
    assert [row.source_path for row in mapped_database] == list(map(str, fake_paths))
    assert [(row.easting, row.northing) for row in mapped_database] == [
        (row.easting, row.northing) for row in original_database
    ]
    assert queries == original_queries == observed["queries"]
    assert observed["database"] == original_database
    assert payload["database_manifest_sha256"] == {"fake_database_manifest": file_sha256(manifest)}
    assert payload["datasets"]["svox_night"]["database_manifest_sha256"] == file_sha256(manifest)
    assert fake_summary.parent == day_summary.parent / "fake_night"
    assert day_summary.read_bytes() == day_payload


def test_suite_reports_full_salad_initialization_when_supplied(tmp_path, monkeypatch):
    config, checkpoint, _ = _suite_fixture(tmp_path, monkeypatch)
    initial = tmp_path / "compact_salad.pt"
    initial.write_bytes(b"compact-salad-initialization")
    config = replace(config, vpr=replace(config.vpr, init_checkpoint=str(initial)))
    summary = trained_suite.evaluate_trained_suite(config, checkpoint, ("svox_night",))
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["initialization"] == str(initial)
    assert payload["initial_checkpoint_sha256"] == file_sha256(initial)


def test_stable_suite_recomputes_recalls_when_query_protocol_changes(tmp_path, monkeypatch):
    config, checkpoint, observed = _suite_fixture(tmp_path, monkeypatch)
    summary = trained_suite.evaluate_trained_suite(config, checkpoint, ("svox_night",))
    first = json.loads(summary.read_text(encoding="utf-8"))["datasets"]["svox_night"]
    observed["queries"] = [replace(observed["queries"][0], easting=50.)]
    trained_suite.evaluate_trained_suite(config, checkpoint, ("svox_night",))
    second = json.loads(summary.read_text(encoding="utf-8"))["datasets"]["svox_night"]
    assert len(observed["evaluations"]) == len(observed["loads"]) == 2
    assert first["query_protocol_fingerprint"] != second["query_protocol_fingerprint"]
    assert first["recalls"]["1"] != second["recalls"]["1"]


@pytest.mark.parametrize("change", ["dustbin_enabled", "sinkhorn_iterations", "gamma_max", "recalls"])
def test_stable_suite_invalidates_cache_when_retrieval_settings_change(tmp_path, monkeypatch, change):
    config, checkpoint, observed = _suite_fixture(tmp_path, monkeypatch, recalls=(1,))
    summary = trained_suite.evaluate_trained_suite(config, checkpoint, ("svox_night",))
    if change == "recalls":
        changed = replace(config, evaluation=replace(config.evaluation, recalls=(1, 5, 10)))
    else:
        field = {"dustbin_enabled": "stable_dustbin_enabled", "sinkhorn_iterations": "stable_sinkhorn_iterations",
                 "gamma_max": "stable_dustbin_gamma_max"}[change]
        value = {"dustbin_enabled": False, "sinkhorn_iterations": 24, "gamma_max": 5.0}[change]
        changed = replace(config, vpr=replace(config.vpr, **{field: value}))
    # The suite must reach its loader rather than return stale metrics. Runtime
    # separately verifies whether the actual checkpoint permits the settings.
    trained_suite.evaluate_trained_suite(changed, checkpoint, ("svox_night",))
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert len(observed["loads"]) == len(observed["evaluations"]) == 2
    assert payload["evaluation_recalls"] == list(changed.evaluation.recalls)
    assert set(payload["datasets"]["svox_night"]["recalls"]) == set(map(str, changed.evaluation.recalls))
    if change != "recalls":
        assert payload["stable_head_options"][change] == getattr(changed.vpr, field)


def test_fake_suite_invalidates_cache_when_mapping_manifest_changes(tmp_path, monkeypatch):
    config, checkpoint, observed = _suite_fixture(tmp_path, monkeypatch)
    fake_paths = [tmp_path / f"fake{index}.png" for index in range(2)]
    for path in fake_paths:
        Image.new("RGB", (28, 28)).save(path)
    manifest = tmp_path / "mapping.csv"
    sources = [tmp_path / f"db_day{index}.png" for index in range(2)]
    _write_fake_manifest(manifest, sources, fake_paths)
    config = replace(config, evaluation=replace(config.evaluation,
        database_domain="fake_night", fake_database_manifest=str(manifest)))
    summary = trained_suite.evaluate_trained_suite(config, checkpoint, ("svox_night",))
    first_digest = json.loads(summary.read_text(encoding="utf-8"))["database_manifest_sha256"]
    updated_fake = tmp_path / "new_fake0.png"
    Image.new("RGB", (28, 28), "gray").save(updated_fake)
    _write_fake_manifest(manifest, sources, [updated_fake, fake_paths[1]])
    trained_suite.evaluate_trained_suite(config, checkpoint, ("svox_night",))
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert len(observed["evaluations"]) == 2
    assert first_digest != payload["database_manifest_sha256"]
    assert observed["evaluations"][-1][0][0].source_path == str(updated_fake)
