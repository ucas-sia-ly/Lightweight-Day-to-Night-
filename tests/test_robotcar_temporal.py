import json
import inspect
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from a2b_lightvpr.config import DataConfig, ExperimentConfig
from a2b_lightvpr.cli import _parser
from a2b_lightvpr import robotcar_temporal as temporal


def row(x, y, yaw=0.0, traverse="route", timestamp="0"):
    return SimpleNamespace(
        easting=float(x), northing=float(y), yaw_deg=float(yaw),
        traverse=traverse, timestamp=str(timestamp), source_path=f"{traverse}-{timestamp}.jpg",
    )


def parameters(**changes):
    values = dict(window_size=3, top_k=2, transition_radius_m=15.0, motion_weight=1.0)
    values.update(changes)
    return temporal.TemporalParameters(**values)


def test_motion_deltas_are_translation_invariant_and_traverse_isolated():
    queries = [
        row(3, 4, traverse="b", timestamp="2"),
        row(0, 0, traverse="a", timestamp="1"),
        row(6, 8, traverse="b", timestamp="3"),
        row(0, 5, traverse="a", timestamp="2"),
    ]
    order, starts, deltas = temporal.ordered_motion_deltas(queries)
    shifted = [
        row(item.easting + 1000, item.northing - 700, traverse=item.traverse, timestamp=item.timestamp)
        for item in queries
    ]
    shifted_order, shifted_starts, shifted_deltas = temporal.ordered_motion_deltas(shifted)
    assert order.tolist() == [1, 3, 0, 2]
    assert starts.tolist() == [True, False, True, False]
    assert deltas.tolist() == pytest.approx([0, 5, 0, 5])
    assert np.array_equal(order, shifted_order)
    assert np.array_equal(starts, shifted_starts)
    assert np.allclose(deltas, shifted_deltas)
    assert "query" not in " ".join(inspect.signature(temporal.rerank_temporal).parameters)


def test_spatial_nms_suppresses_same_view_but_keeps_opposite_direction():
    xy = np.asarray([[0, 0], [2, 0], [2, 0], [20, 0]], dtype=np.float64)
    yaw = np.asarray([0, 10, 180, 0], dtype=np.float32)
    indices, scores = temporal.spatial_nms(
        np.asarray([0, 1, 2, 3]), np.asarray([1.0, 0.9, 0.8, 0.7]), xy, yaw, 3
    )
    assert indices.tolist() == [0, 2, 3]
    assert scores.tolist() == pytest.approx([1.0, 0.8, 0.7])


def test_topk_path_recovers_consistent_candidate_when_raw_top1_jumps():
    xy = np.asarray([
        [1000, 0], [2000, 0], [3000, 0],
        [0, 0], [5, 0], [10, 0],
    ], dtype=np.float64)
    yaw = np.zeros(6, dtype=np.float32)
    raw_indices = np.asarray([[0, 3], [1, 4], [2, 5]], dtype=np.int32)
    raw_scores = np.asarray([[1.0, 0.9], [1.0, 0.9], [1.0, 0.9]], dtype=np.float32)
    rankings, diagnostics, _ = temporal.rerank_temporal(
        raw_indices, raw_scores, xy, yaw,
        np.asarray([0, 5, 5], dtype=np.float32),
        np.asarray([True, False, False]), parameters(), use_nms=False,
    )
    assert rankings[2, 0] == 5
    assert diagnostics[2]["reason"] == "full_window"
    assert diagnostics[2]["window_used"] == 3


def test_impossible_transition_falls_back_to_raw_top1():
    xy = np.asarray([[0, 0], [1000, 0]], dtype=np.float64)
    raw_indices = np.asarray([[0], [1]], dtype=np.int32)
    rankings, diagnostics, _ = temporal.rerank_temporal(
        raw_indices, np.ones((2, 1), dtype=np.float32), xy, np.zeros(2),
        np.asarray([0, 5], dtype=np.float32), np.asarray([True, False]),
        parameters(top_k=1, transition_radius_m=1), use_nms=False,
    )
    assert rankings[:, 0].tolist() == [0, 1]
    assert diagnostics[0]["reason"] == "warmup_single"
    assert diagnostics[1]["reason"] == "no_valid_transition_raw_top1"


def test_shorter_window_is_used_when_oldest_transition_is_invalid():
    xy = np.asarray([[1000, 0], [0, 0], [5, 0]], dtype=np.float64)
    raw_indices = np.asarray([[0], [1], [2]], dtype=np.int32)
    rankings, diagnostics, _ = temporal.rerank_temporal(
        raw_indices, np.ones((3, 1), dtype=np.float32), xy, np.zeros(3),
        np.asarray([0, 5, 5], dtype=np.float32), np.asarray([True, False, False]),
        parameters(top_k=1, transition_radius_m=10), use_nms=False,
    )
    assert rankings[2, 0] == 2
    assert diagnostics[2]["reason"] == "shortened_window"
    assert diagnostics[2]["window_used"] == 2


def test_selection_tie_break_is_fully_deterministic():
    first = {"recall1": 0.9, "parameters": {
        "top_k": 10, "window_size": 4, "transition_radius_m": 50, "motion_weight": 1.0,
    }}
    second = {"recall1": 0.9, "parameters": {
        "top_k": 5, "window_size": 3, "transition_radius_m": 15, "motion_weight": 0.5,
    }}
    assert sorted([first, second], key=temporal._selection_sort_key)[0] is second


def test_selection_is_hard_limited_to_val(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    manifest = tmp_path / "manifest.csv"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("fixture", encoding="utf-8")
    config = ExperimentConfig(DataConfig(str(manifest), "protocol.json"))
    cache = tmp_path / "val.npz"
    metadata = {
        "condition": "val", "descriptor_dim": 512,
        "checkpoint_sha256": temporal.file_sha256(checkpoint),
        "manifest_sha256": temporal.file_sha256(manifest),
        "query_protocol_fingerprint": "val-fingerprint",
    }
    arrays = {
        "raw_indices": np.asarray([[0], [0]], dtype=np.int32),
        "raw_scores": np.ones((2, 1), dtype=np.float32),
        "database_xy": np.asarray([[0, 0]], dtype=np.float64),
        "database_yaw": np.asarray([0], dtype=np.float32),
        "motion_deltas": np.asarray([0, 5], dtype=np.float32),
        "sequence_starts": np.asarray([True, False]),
    }
    seen = []
    monkeypatch.setattr(temporal, "build_robotcar_temporal_cache", lambda c, p, condition, out: seen.append(condition) or cache)
    monkeypatch.setattr(temporal, "load_robotcar_temporal_cache", lambda p: (metadata, arrays))
    monkeypatch.setattr(temporal, "_condition_rows", lambda c, condition: (
        [row(0, 0)], [row(0, 0, timestamp=1), row(0, 5, timestamp=2)], "val"
    ) if condition == "val" else pytest.fail("test split was accessed during selection"))
    output = tmp_path / "selection.json"
    temporal.select_robotcar_temporal(config, checkpoint, output, cache)
    assert seen == ["val"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selection_split"] == "val"
    assert len(payload["trials"]) == 36


def test_selection_rejects_checkpoint_or_manifest_mismatch():
    selection = {
        "selection_split": "val", "checkpoint_sha256": "a", "manifest_sha256": "b",
        "selected_parameters": temporal.asdict(parameters()),
    }
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        temporal._validate_selection(selection, {"checkpoint_sha256": "x", "manifest_sha256": "b"})


def test_resource_accounting_is_amortized_one_new_encoding():
    resources = temporal.temporal_resource_summary(
        parameters(window_size=4, top_k=10), 1000, 40.0,
        {"model_parameters": 10_000_000, "peak_cuda_mib": 100},
        {"latency_ms": {"p50": 2.0}}, np.asarray([0.1, 0.2]),
    )
    assert resources["startup_encodings_per_observed_frame"] == 1
    assert resources["steady_state_new_encodings_per_localization"] == 1
    assert resources["sliding_window_state_bytes"] == 4 * 10 * 12 + 3 * 4


def test_dimension_guard_rejects_non_512_before_model_loading(tmp_path):
    config = ExperimentConfig(DataConfig("missing.csv", "missing.json"))
    config = replace(config, vpr=replace(config.vpr, descriptor_dim=128))
    with pytest.raises(ValueError, match="512-dimensional"):
        temporal.build_robotcar_temporal_cache(config, "missing.pt", "val", tmp_path / "x.npz")


def test_temporal_cli_exposes_selection_evaluation_and_summary():
    parser = _parser()
    selected = parser.parse_args([
        "select-robotcar-temporal", "--config", "c.yaml", "--checkpoint", "m.pt",
        "--cache", "v.npz", "--output", "selection.json",
    ])
    assert selected.command == "select-robotcar-temporal"
    evaluated = parser.parse_args([
        "evaluate-robotcar-temporal", "--config", "c.yaml", "--checkpoint", "m.pt",
        "--selection", "selection.json", "--condition", "night",
        "--cache", "n.npz", "--output", "night.json",
    ])
    assert evaluated.condition == "night"
    summarized = parser.parse_args([
        "summarize-robotcar-temporal", "--night", "n.json", "--day", "d.json",
        "--crica", "c.json", "--salad", "s.json", "--boq", "b.json",
        "--reference-resources", "resources.json",
        "--output", "o.json",
    ])
    assert summarized.command == "summarize-robotcar-temporal"


def test_summary_checks_protocol_and_applies_acceptance_rules(tmp_path):
    primary = "distance_10m_yaw_30deg"
    resources = {"rerank_cpu_p95_ms": 1.5}
    common = {
        "checkpoint_sha256": "checkpoint", "manifest_sha256": "manifest",
        "selection_sha256": "selection", "query_protocol_fingerprint": "protocol",
    }
    def evaluation(path, condition, single, mgtr):
        path.write_text(json.dumps({
            **common, "condition": condition, "resources": resources,
            "single_frame": {"grid": {primary: {"recalls": {"1": single}}}},
            "mgtr": {"grid": {primary: {"recalls": {"1": mgtr}}}},
        }), encoding="utf-8")
    night, day = tmp_path / "night.json", tmp_path / "day.json"
    evaluation(night, "night", 0.88, 0.93)
    evaluation(day, "day", 0.90, 0.896)
    references = {}
    for name, recall in (("crica", 0.89), ("salad", 0.92), ("boq", 0.97)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({
            "model": name,
            "datasets": {"robotcar_project": {
                "query_protocol_fingerprint": "protocol", "recalls": {"1": recall},
                "parameters": 1, "descriptor_dim": 2,
            }},
        }), encoding="utf-8")
        references[name] = path
    output = tmp_path / "comparison.json"
    resource_models = {}
    for name in references:
        resource_models[name] = {
            "parameters": 10, "descriptor_dim": 20, "checkpoint_bytes": 30,
            "batch1_p50_ms": 4.0, "batch1_p95_ms": 5.0, "peak_cuda_mib": 6.0,
            "database_descriptor_memory": {
                "1000000": {"fp16_gib": 0.04, "fp32_gib": 0.08}
            },
        }
    resource_path = tmp_path / "resources.json"
    resource_path.write_text(json.dumps({"models": resource_models}), encoding="utf-8")
    temporal.summarize_robotcar_temporal(
        night, day, references["crica"], references["salad"], references["boq"], output,
        resource_path,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["acceptance"]["adopt_as_robotcar_method"]
    assert payload["claims"]["reaches_salad_single_frame"]
    assert not payload["claims"]["exceeds_boq_single_frame"]
    assert payload["reference_resource_benchmark_sha256"] == temporal.file_sha256(resource_path)
    assert payload["dinov2_single_frame_references"]["salad"]["parameters"] == 10
    assert payload["dinov2_single_frame_references"]["salad"]["batch1_p95_ms"] == 5.0
    assert payload["dinov2_single_frame_references"]["salad"][
        "database_descriptor_memory_1m"
    ]["fp16_gib"] == 0.04
    empty_resources = tmp_path / "empty-resources.json"
    empty_resources.write_text(json.dumps({"models": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="has no model"):
        temporal.summarize_robotcar_temporal(
            night, day, references["crica"], references["salad"], references["boq"], output,
            empty_resources,
        )
    bad = json.loads(references["crica"].read_text(encoding="utf-8"))
    bad["datasets"]["robotcar_project"]["query_protocol_fingerprint"] = "wrong"
    references["crica"].write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="Reference protocol mismatch"):
        temporal.summarize_robotcar_temporal(
            night, day, references["crica"], references["salad"], references["boq"], output,
            resource_path,
        )
