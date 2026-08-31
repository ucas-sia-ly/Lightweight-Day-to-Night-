import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from a2b_lightvpr import robotcar_temporal as temporal
from a2b_lightvpr.cli import _parser
from a2b_lightvpr.config import DataConfig, ExperimentConfig


def row(x, y, yaw=0.0, traverse="route", timestamp="0"):
    return SimpleNamespace(easting=float(x), northing=float(y), yaw_deg=float(yaw),
                           traverse=traverse, timestamp=str(timestamp),
                           source_path=f"{traverse}-{timestamp}.jpg")


def test_query_order_is_translation_invariant_and_traverse_isolated():
    queries = [row(3, 4, traverse="b", timestamp="2"), row(0, 0, traverse="a", timestamp="1"),
               row(6, 8, traverse="b", timestamp="3"), row(0, 5, traverse="a", timestamp="2")]
    order, starts, deltas = temporal.ordered_motion_deltas(queries)
    shifted = [row(item.easting + 1000, item.northing - 700, traverse=item.traverse, timestamp=item.timestamp) for item in queries]
    shifted_order, shifted_starts, shifted_deltas = temporal.ordered_motion_deltas(shifted)
    assert order.tolist() == [1, 3, 0, 2]
    assert starts.tolist() == [True, False, True, False]
    assert deltas.tolist() == pytest.approx([0, 5, 0, 5])
    assert np.array_equal(order, shifted_order)
    assert np.array_equal(starts, shifted_starts)
    assert np.allclose(deltas, shifted_deltas)


def test_clustered_temporal_uses_center_window_and_top100_fallback():
    raw = np.asarray([[10, 11, 12, 13, 14], [10, 11, 12, 13, 14], [20, 21, 22, 23, 24],
                      [12, 13, 14, 15, 16], [12, 13, 14, 15, 16]], dtype=np.int32)
    scores = np.tile(np.asarray([[.99, .8, .7, .6, .5]], dtype=np.float32), (5, 1))
    ranking, decisions, _ = temporal.rerank_clustered_temporal(
        raw, scores, np.arange(30), np.zeros(30, dtype=object),
        np.asarray([True, False, False, False, False]),
        temporal.ClusteredTemporalParameters(window_size=3, top_k=2),
    )
    assert ranking[2, 0] == 20
    assert decisions[2]["window_used"] == 3


def test_clustered_windows_are_odd_and_bounded():
    assert temporal.CLUSTERED_WINDOWS == (1, 3, 5, 7, 9)
    with pytest.raises(ValueError):
        temporal.rerank_clustered_temporal(
            np.zeros((1, 2), dtype=np.int32), np.zeros((1, 2)), np.arange(2),
            np.zeros(2, dtype=object), np.asarray([True]),
            temporal.ClusteredTemporalParameters(window_size=4),
        )


def test_map_covered_protocol_uses_common_full_window(monkeypatch):
    monkeypatch.setattr(temporal, "_positive_indices", lambda *args: [
        np.asarray([0]) if value else np.asarray([], dtype=int)
        for value in (1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1)
    ])
    mask, covered = temporal.map_covered_center_mask(
        [object()], [object()] * 11, np.asarray([True] + [False] * 10), window_size=9)
    assert covered.sum() == 10
    assert np.flatnonzero(mask).tolist() == [4]


def test_clustered_center_window_never_crosses_sequence():
    raw = np.tile(np.arange(10, dtype=np.int32), (6, 1))
    scores = np.tile(np.linspace(1, 0, 10), (6, 1))
    _, decisions, _ = temporal.rerank_clustered_temporal(
        raw, scores, np.arange(10), np.zeros(10, dtype=object),
        np.asarray([True, False, False, True, False, False]),
        temporal.ClusteredTemporalParameters(window_size=5),
    )
    assert [item["window_used"] for item in decisions] == [3, 3, 3, 3, 3, 3]


def test_selection_is_hard_limited_to_val(tmp_path, monkeypatch):
    checkpoint, manifest, cache = tmp_path / "model.pt", tmp_path / "manifest.csv", tmp_path / "val.npz"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("fixture", encoding="utf-8")
    config = ExperimentConfig(DataConfig(str(manifest), "protocol.json"))
    metadata = {"condition": "val", "descriptor_dim": 512,
                "checkpoint_sha256": temporal.file_sha256(checkpoint),
                "manifest_sha256": temporal.file_sha256(manifest),
                "query_protocol_fingerprint": "val-fingerprint"}
    arrays = {"raw_indices": np.asarray([[0], [0]], dtype=np.int32),
              "raw_scores": np.ones((2, 1), dtype=np.float32),
              "database_xy": np.asarray([[0, 0]], dtype=np.float64),
              "database_yaw": np.asarray([0], dtype=np.float32),
              "motion_deltas": np.asarray([0, 5], dtype=np.float32),
              "sequence_starts": np.asarray([True, False])}
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
    assert len(payload["trials"]) == 5


def test_resource_summary_records_one_encoding_and_window_state():
    resources = temporal.temporal_resource_summary(
        temporal.ClusteredTemporalParameters(window_size=5, top_k=10), 1000, 40.0,
        {"model_parameters": 10_000_000, "peak_cuda_mib": 100},
        {"latency_ms": {"p50": 2.0}}, np.asarray([0.1, 0.2]))
    assert resources["startup_encodings_per_observed_frame"] == 1
    assert resources["steady_state_new_encodings_per_localization"] == 1
    assert resources["sliding_window_state_bytes"] == 5 * 10 * 12 + 4 * 4


@pytest.mark.parametrize("descriptor_dim", [128, 4096])
def test_dimension_guard_rejects_unsupported_dimension_before_model_loading(tmp_path, descriptor_dim):
    config = ExperimentConfig(DataConfig("missing.csv", "missing.json"))
    config = replace(config, vpr=replace(config.vpr, descriptor_dim=descriptor_dim))
    with pytest.raises(ValueError, match="512-dimensional"):
        temporal.build_robotcar_temporal_cache(config, "missing.pt", "val", tmp_path / "x.npz")


def test_temporal_resource_summary_uses_actual_descriptor_dimension():
    resources = temporal.temporal_resource_summary(
        temporal.ClusteredTemporalParameters(window_size=3), 1000, 40.0,
        {"descriptor_dim": 4096, "model_parameters": 1, "peak_cuda_mib": 2}, {}, np.asarray([0.1]))
    assert resources["descriptor_dim"] == 4096
    assert resources["database_descriptor_mib_fp16"] == pytest.approx(1000 * 4096 * 2 / 1024**2)


def test_temporal_cli_only_exposes_clustered_method():
    parser = _parser()
    selected = parser.parse_args(["select-robotcar-temporal", "--config", "c.yaml", "--checkpoint", "m.pt", "--cache", "v.npz", "--output", "selection.json"])
    assert selected.method == "clustered-mgtr"
    with pytest.raises(SystemExit):
        parser.parse_args(["select-robotcar-temporal", "--config", "c.yaml", "--checkpoint", "m.pt", "--cache", "v.npz", "--output", "o.json", "--method", "mgtr-v1"])
