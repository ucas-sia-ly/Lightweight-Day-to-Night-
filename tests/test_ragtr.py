from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from a2b_lightvpr import ragtr
from a2b_lightvpr.vpr.losses import mined_route_contrastive_loss
from a2b_lightvpr.vpr.route_mining import RouteMiningCache


@dataclass
class Row:
    traverse: str
    timestamp: int
    easting: float
    northing: float


def parameters(**values):
    defaults = dict(
        window_size=4,
        top_k=2,
        alpha=1.0,
        transition_radius_m=2.0,
        motion_weight=1.0,
        confidence_threshold=0.0,
    )
    defaults.update(values)
    return ragtr.RAGTRParameters(**defaults)


def test_route_progress_is_strict_and_isolated_by_traverse():
    rows = [
        Row("a", 20, 5, 0), Row("b", 10, 100, 0),
        Row("a", 10, 0, 0), Row("b", 20, 107, 0),
    ]
    progress, routes = ragtr.route_progress(rows)
    assert progress.tolist() == [5.0, 0.0, 0.0, 7.0]
    assert routes[0] == routes[2]
    assert routes[1] == routes[3]
    assert routes[0] != routes[1]


def test_route_progress_rejects_duplicate_position():
    with pytest.raises(ValueError, match="strictly increasing"):
        ragtr.route_progress([Row("a", 1, 0, 0), Row("a", 2, 0, 0)])


def test_geometry_score_handles_short_empty_and_magsac_failure():
    points = np.zeros((10, 2), dtype=np.float32)
    assert ragtr.geometric_verification_score(points, points, np.zeros((0, 2), int)) == (0, 0, 0)

    class FailedCV:
        USAC_MAGSAC = 1

        @staticmethod
        def findFundamentalMat(*args, **kwargs):
            raise RuntimeError("failure")

    matches = np.stack((np.arange(8), np.arange(8)), axis=1)
    assert ragtr.geometric_verification_score(
        points, points, matches, cv2_module=FailedCV
    ) == (0, 8, 0)


def test_geometry_score_uses_magsac_inlier_count():
    points = np.arange(20, dtype=np.float32).reshape(10, 2)

    class CV:
        USAC_MAGSAC = 7

        @staticmethod
        def findFundamentalMat(first, second, method, threshold, confidence, iterations):
            assert method == 7
            assert (threshold, confidence, iterations) == (1.0, 0.999, 10_000)
            return np.eye(3), np.asarray([1, 1, 1, 1, 0, 0, 0, 0])

    matches = np.stack((np.arange(8), np.arange(8)), axis=1)
    score, count, inliers = ragtr.geometric_verification_score(
        points, points, matches, cv2_module=CV
    )
    assert count == 8 and inliers == 4
    assert score == pytest.approx(np.log1p(4) * 0.5)


def test_utility_falls_back_to_appearance_when_geometry_is_empty():
    appearance = np.asarray([[0.9, 0.8], [0.9, 0.8]])
    geometry = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    utility = ragtr.utility_scores(appearance, geometry, 0.25)
    assert utility[0].tolist() == pytest.approx([1.0, 0.0])
    assert utility[1, 0] > utility[1, 1]


def test_fixed_direction_path_recovers_candidate_and_never_switches_direction():
    raw = np.asarray([[9, 0], [8, 1], [7, 2], [6, 3]], dtype=np.int32)
    candidates = np.asarray([[9, 0], [8, 1], [7, 2], [6, 3]], dtype=np.int32)
    # Appearance favors the first chain; geometry consistently favors the second.
    scores = np.asarray([[1.0, .9], [1.0, .9], [1.0, .9], [1.0, .9]])
    geometry = np.asarray([[0.0, 1.0]] * 4)
    route_progress = np.arange(10, dtype=np.float64) * 5
    routes = np.zeros(10, dtype=np.int32)
    deltas = np.asarray([0, 5, 5, 5], dtype=np.float32)
    starts = np.asarray([True, False, False, False])
    rankings, decisions, _ = ragtr.rerank_ragtr(
        raw, candidates, scores, geometry, route_progress, routes, deltas, starts,
        parameters(window_size=4, fixed_lag=0, alpha=0.25),
    )
    assert rankings[-1, 0] == 3
    assert decisions[-1]["direction"] == 1


def test_confidence_gate_preserves_raw_top1():
    raw = np.asarray([[0, 4], [1, 5], [2, 6]], dtype=np.int32)
    scores = np.asarray([[1.0, .9], [1.0, .9], [1.0, .9]])
    rankings, decisions, _ = ragtr.rerank_ragtr(
        raw, raw, scores, np.zeros_like(scores), np.arange(7) * 5.0,
        np.zeros(7, dtype=np.int32), np.asarray([0, 5, 5]),
        np.asarray([True, False, False]), parameters(confidence_threshold=100, fixed_lag=0),
    )
    assert np.array_equal(rankings[:, 0], raw[:, 0])
    assert all(item["reason"] != "path_override" for item in decisions)


def test_fixed_two_frame_lag_alignment_and_flush():
    raw = np.tile(np.arange(5, dtype=np.int32), (5, 1))
    scores = np.tile(np.linspace(1, 0, 5), (5, 1))
    _, decisions, _ = ragtr.rerank_ragtr(
        raw, raw[:, :2], scores[:, :2], np.zeros((5, 2)), np.arange(5) * 5.0,
        np.zeros(5, dtype=np.int32), np.asarray([0, 5, 5, 5, 5]),
        np.asarray([True, False, False, False, False]), parameters(fixed_lag=2),
    )
    assert [item["future_frames"] for item in decisions] == [2, 2, 2, 1, 0]


def test_sequence_ranges_never_cross_traverse():
    assert ragtr._sequence_ranges(np.asarray([True, False, True, False])) == [(0, 2), (2, 4)]


def test_global_and_geometry_cache_identity_must_match():
    global_metadata = {
        "condition": "val", "checkpoint_sha256": "checkpoint",
        "manifest_sha256": "manifest", "cache_sha256": "global",
    }
    geometry_metadata = {
        **global_metadata, "global_cache_sha256": "global",
    }
    ragtr._validate_geometry_identity(global_metadata, geometry_metadata)
    geometry_metadata["condition"] = "night"
    with pytest.raises(ValueError, match="condition"):
        ragtr._validate_geometry_identity(global_metadata, geometry_metadata)


def test_ragtr_selection_rejects_non_val_and_verifier_mismatch():
    selection = {
        "schema": "robotcar_ragtr_selection_v1", "selection_split": "night",
        "checkpoint_sha256": "c", "manifest_sha256": "m", "verifier_sha256": "v",
        "selected_parameters": parameters().__dict__,
    }
    global_metadata = {"checkpoint_sha256": "c", "manifest_sha256": "m"}
    geometry_metadata = {"verifier_sha256": "v"}
    with pytest.raises(ValueError, match="RobotCar val"):
        ragtr._validate_ragtr_selection(selection, global_metadata, geometry_metadata)
    selection["selection_split"] = "val"
    geometry_metadata["verifier_sha256"] = "changed"
    with pytest.raises(ValueError, match="verifier_sha256"):
        ragtr._validate_ragtr_selection(selection, global_metadata, geometry_metadata)


def test_selection_grid_contains_every_required_window(monkeypatch, tmp_path):
    # The constants are part of the public experiment contract even when formal caches are absent.
    assert ragtr.RAGTR_WINDOWS == (3, 4, 5, 6)
    assert ragtr.RAGTR_FIXED_LAG == 2
    assert set(ragtr.RAGTR_TOP_K) == {10, 20}
    assert ragtr.RAGTR_DISTANCES_M == (5, 10, 15, 20, 25, 50)
    assert ragtr.RAGTR_YAWS_DEG == (10, 20, 30, 45, 90, None)


def test_ragtr_metric_grid_contains_every_distance_yaw_and_recall():
    database = [
        type("R", (), {"easting": 0.0, "northing": 0.0, "yaw_deg": 0.0})()
    ]
    queries = [
        type("R", (), {"easting": 0.0, "northing": 0.0, "yaw_deg": 0.0})()
    ]
    grid = ragtr.evaluate_ragtr_grid(np.asarray([[0]], dtype=np.int32), database, queries)
    assert len(grid) == 36
    assert all(set(value["recalls"]) == {"1", "5", "10"} for value in grid.values())


def test_mined_route_loss_uses_frozen_candidates_and_ignores_ambiguous_pairs():
    fake = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    candidates = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
         [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0]]],
        requires_grad=True,
    )
    positives = torch.tensor([[True, False, False], [True, False, False]])
    negatives = torch.tensor([[False, True, False], [False, True, False]])
    loss, stats = mined_route_contrastive_loss(
        fake, candidates, positives, negatives, temperature=0.1
    )
    assert torch.isfinite(loss)
    assert stats["valid_anchors"].item() == 2
    loss.backward()
    assert fake.grad is not None and fake.grad.abs().sum() > 0
    assert candidates.grad is None


def test_mined_route_loss_returns_differentiable_zero_for_non_robotcar_batch():
    fake = torch.randn(2, 4, requires_grad=True)
    candidates = torch.randn(2, 3, 4)
    empty = torch.zeros(2, 3, dtype=torch.bool)
    loss, stats = mined_route_contrastive_loss(fake, candidates, empty, empty)
    assert loss.item() == 0
    assert stats["valid_anchors"].item() == 0
    loss.backward()
    assert fake.grad is not None


def test_route_mining_cache_requires_the_initial_checkpoint_fingerprint(tmp_path):
    checkpoint = tmp_path / "initial.pt"
    checkpoint.write_bytes(b"initial")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("image_id\n", encoding="utf-8")
    cache = tmp_path / "mining.npz"
    descriptors = tmp_path / "mining.descriptors.npy"
    np.savez_compressed(
        cache,
        manifest_indices=np.asarray([], dtype=np.int64),
        candidate_indices=np.zeros((0, 1), dtype=np.int32),
        positive_mask=np.zeros((0, 1), dtype=np.bool_),
        negative_mask=np.zeros((0, 1), dtype=np.bool_),
    )
    np.save(descriptors, np.zeros((0, 4096), dtype=np.float16))
    import hashlib, json
    metadata = {
        "schema": "robotcar_route_mining_v1",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "checkpoint_sha256": "wrong",
        "descriptor_dim": 4096,
        "positive_radius_m": 10.0,
        "positive_yaw_deg": 30.0,
        "negative_radius_m": 25.0,
        "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "descriptors_sha256": hashlib.sha256(descriptors.read_bytes()).hexdigest(),
    }
    cache.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    config = type("Config", (), {
        "vpr": type("VPR", (), {
            "route_mining_cache": str(cache), "descriptor_dim": 4096,
            "finetune_checkpoint": str(checkpoint), "route_positive_radius_m": 10.0,
            "route_positive_yaw_deg": 30.0, "route_negative_radius_m": 25.0,
        })()
    })()
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        RouteMiningCache(config, manifest)


def test_ragtr_summary_rejects_reference_protocol_mismatch(tmp_path):
    primary = "distance_10m_yaw_30deg"
    grid = {primary: {"recalls": {"1": 0.9, "5": 0.95, "10": 0.98}}}
    common = {
        "schema": "robotcar_ragtr_evaluation_v1",
        "checkpoint_sha256": "checkpoint",
        "manifest_sha256": "manifest",
        "selection_sha256": "selection",
        "verifier_sha256": "verifier",
        "query_protocol_fingerprint": "protocol-a",
        "single_frame": {"grid": grid},
        "ragtr": {"grid": grid},
        "raw_oracle_10m_30deg": {"recalls": {"100": 1.0}},
        "resources": {},
    }
    night = tmp_path / "night.json"
    day = tmp_path / "day.json"
    night.write_text(__import__("json").dumps({**common, "condition": "night"}))
    day.write_text(__import__("json").dumps({**common, "condition": "day"}))
    reference = tmp_path / "reference.json"
    reference.write_text(__import__("json").dumps({
        "datasets": {"robotcar_project": {
            "recalls": {"1": 0.92}, "protocol_fingerprint": "protocol-b"
        }}
    }))
    with pytest.raises(ValueError, match="protocol mismatch"):
        ragtr.summarize_robotcar_ragtr(
            night, day, reference, reference, tmp_path / "summary.json"
        )
