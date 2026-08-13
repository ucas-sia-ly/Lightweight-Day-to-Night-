from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .checkpointing import file_sha256, write_json
from .evaluation import extract_descriptors
from .robotcar_temporal import (
    NMS_RADIUS_M,
    NMS_YAW_DEG,
    RAW_TOP_K,
    _condition_rows,
    _top_scores,
    ordered_motion_deltas,
    ranking_metrics,
    spatial_nms,
)
from .runtime import load_vpr_model
from .transforms import VPRTransform
from .reference_models import query_protocol_fingerprint


RAGTR_DESCRIPTOR_DIM = 4096
RAGTR_WINDOWS = (3, 4, 5, 6)
RAGTR_TOP_K = (10, 20)
RAGTR_ALPHAS = (0.25, 0.5, 0.75, 1.0)
RAGTR_RADII_M = (10.0, 15.0, 20.0)
RAGTR_WEIGHTS = (0.25, 0.5, 1.0)
RAGTR_GATES = (0.0, 0.025, 0.05, 0.1)
RAGTR_FIXED_LAG = 2
RAGTR_MOTION_SIGMA_M = 5.0
RAGTR_DISTANCES_M = (5.0, 10.0, 15.0, 20.0, 25.0, 50.0)
RAGTR_YAWS_DEG = (10.0, 20.0, 30.0, 45.0, 90.0, None)
LOCAL_LONG_SIDE = 1024
LOCAL_MAX_FEATURES = 2048
MAGSAC_THRESHOLD_PX = 1.0
MAGSAC_CONFIDENCE = 0.999
MAGSAC_MAX_ITERS = 10_000


@dataclass(frozen=True)
class RAGTRParameters:
    window_size: int
    top_k: int
    alpha: float
    transition_radius_m: float
    motion_weight: float
    confidence_threshold: float
    fixed_lag: int = RAGTR_FIXED_LAG
    motion_sigma_m: float = RAGTR_MOTION_SIGMA_M
    raw_top_k: int = RAW_TOP_K
    nms_radius_m: float = NMS_RADIUS_M
    nms_yaw_deg: float = NMS_YAW_DEG


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _metadata_path(path: str | Path) -> Path:
    return Path(path).with_suffix(".json")


def _ordered_rows(config, condition: str):
    database, original_queries, split = _condition_rows(config, condition)
    order, starts, deltas = ordered_motion_deltas(original_queries)
    queries = [original_queries[int(index)] for index in order]
    return database, queries, split, starts, deltas


def build_ragtr_global_cache(config, checkpoint: str | Path, condition: str, cache: str | Path) -> Path:
    if config.vpr.descriptor_dim != RAGTR_DESCRIPTOR_DIM:
        raise ValueError("RA-GTR requires a 4096-dimensional MixVPR checkpoint")
    checkpoint = Path(checkpoint).expanduser().resolve()
    cache = Path(cache)
    database, queries, split, starts, deltas = _ordered_rows(config, condition)
    expected = {
        "schema": "robotcar_ragtr_global_cache_v1",
        "condition": condition,
        "split": split,
        "descriptor_dim": RAGTR_DESCRIPTOR_DIM,
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest_sha256": file_sha256(config.data.robotcar_manifest),
        "database_rows": len(database),
        "query_rows": len(queries),
        "raw_top_k": min(RAW_TOP_K, len(database)),
        "input_size": [config.vpr.image_height, config.vpr.image_width],
        "query_protocol_fingerprint": query_protocol_fingerprint(
            database, queries, 10.0, 30.0
        ),
    }
    metadata_path = _metadata_path(cache)
    if cache.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            if metadata.get("cache_sha256") == file_sha256(cache):
                with np.load(cache) as arrays:
                    if arrays["raw_indices"].shape == (len(queries), expected["raw_top_k"]):
                        return cache

    model, state, _ = load_vpr_model(config, checkpoint)
    if int(getattr(model, "descriptor_dim", 0)) != RAGTR_DESCRIPTOR_DIM:
        raise ValueError("RA-GTR model descriptor dimension must be 4096")
    if int(state.get("descriptor_dim", RAGTR_DESCRIPTOR_DIM)) != RAGTR_DESCRIPTOR_DIM:
        raise ValueError("RA-GTR checkpoint descriptor dimension must be 4096")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    transform = VPRTransform(config.vpr.image_height, config.vpr.image_width)
    batch_size = max(1, config.vpr.batch_size * 2)
    started = time.perf_counter()
    database_descriptors = extract_descriptors(
        model, database, transform, device, batch_size, config.vpr.workers
    )
    query_descriptors = extract_descriptors(
        model, queries, transform, device, batch_size, config.vpr.workers
    )
    descriptor_seconds = time.perf_counter() - started
    search_started = time.perf_counter()
    raw_indices, raw_scores = _top_scores(database_descriptors, query_descriptors, expected["raw_top_k"])
    search_seconds = time.perf_counter() - search_started
    _atomic_npz(
        cache,
        raw_indices=raw_indices,
        raw_scores=raw_scores,
        database_xy=np.asarray([(row.easting, row.northing) for row in database], dtype=np.float64),
        database_yaw=np.asarray([row.yaw_deg for row in database], dtype=np.float32),
        motion_deltas=deltas.astype(np.float32),
        sequence_starts=starts.astype(np.bool_),
    )
    write_json({
        **expected,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "descriptor_extraction_seconds": descriptor_seconds,
        "top100_search_ms_per_query": search_seconds * 1000.0 / max(1, len(queries)),
        "cache_sha256": file_sha256(cache),
    }, metadata_path)
    return cache


def load_npz_cache(path: str | Path, schema: str) -> tuple[dict, dict[str, np.ndarray]]:
    path = Path(path)
    metadata = json.loads(_metadata_path(path).read_text(encoding="utf-8"))
    if metadata.get("schema") != schema:
        raise ValueError(f"Unexpected cache schema: {metadata.get('schema')}")
    if metadata.get("cache_sha256") != file_sha256(path):
        raise RuntimeError(f"Cache SHA256 mismatch: {path}")
    with np.load(path) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    return metadata, arrays


def route_progress(database) -> tuple[np.ndarray, np.ndarray]:
    """Build per-traverse cumulative progress without joining separate routes."""
    progress = np.zeros(len(database), dtype=np.float64)
    route_ids = np.empty(len(database), dtype=np.int32)
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(database):
        grouped.setdefault(str(getattr(row, "traverse", "single")), []).append(index)
    for route_id, traverse in enumerate(sorted(grouped)):
        indices = sorted(
            grouped[traverse],
            key=lambda index: (int(getattr(database[index], "timestamp", index)), index),
        )
        route_ids[indices] = route_id
        for previous, current in zip(indices, indices[1:]):
            distance = math.hypot(
                float(database[current].easting) - float(database[previous].easting),
                float(database[current].northing) - float(database[previous].northing),
            )
            if not distance > 0:
                raise ValueError("Database route progress must be strictly increasing")
            progress[current] = progress[previous] + distance
    return progress, route_ids


def nms_candidates(arrays: dict[str, np.ndarray], top_k: int = max(RAGTR_TOP_K)):
    count = arrays["raw_indices"].shape[0]
    indices = np.full((count, top_k), -1, dtype=np.int32)
    scores = np.full((count, top_k), -np.inf, dtype=np.float32)
    for query_index in range(count):
        kept_indices, kept_scores = spatial_nms(
            arrays["raw_indices"][query_index], arrays["raw_scores"][query_index],
            arrays["database_xy"], arrays["database_yaw"], top_k,
            radius_m=NMS_RADIUS_M, yaw_deg=NMS_YAW_DEG,
        )
        indices[query_index, :len(kept_indices)] = kept_indices
        scores[query_index, :len(kept_scores)] = kept_scores
    return indices, scores


def geometric_verification_score(
    keypoints_a: np.ndarray,
    keypoints_b: np.ndarray,
    matches: np.ndarray,
    *,
    cv2_module=None,
) -> tuple[float, int, int]:
    match_count = int(len(matches))
    if match_count < 8:
        return 0.0, match_count, 0
    if cv2_module is None:
        import cv2 as cv2_module
    try:
        _, mask = cv2_module.findFundamentalMat(
            np.asarray(keypoints_a[matches[:, 0]], dtype=np.float32),
            np.asarray(keypoints_b[matches[:, 1]], dtype=np.float32),
            cv2_module.USAC_MAGSAC,
            MAGSAC_THRESHOLD_PX,
            MAGSAC_CONFIDENCE,
            MAGSAC_MAX_ITERS,
        )
    except Exception:
        mask = None
    inliers = int(np.asarray(mask).reshape(-1).astype(bool).sum()) if mask is not None else 0
    score = math.log1p(inliers) * inliers / max(match_count, 1)
    return float(score), match_count, inliers


def _resize_local_image(path: str | Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        scale = LOCAL_LONG_SIDE / max(image.size)
        size = tuple(max(1, int(round(value * scale))) for value in image.size)
        image = image.resize(size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class LocalGeometryVerifier:
    def __init__(self, cache_dir: str | Path):
        try:
            import kornia
            import kornia.feature as feature
        except ImportError as error:
            raise RuntimeError("RA-GTR local verification requires kornia>=0.8") from error
        self.kornia = kornia
        self.feature = feature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.disk = feature.DISK.from_pretrained("depth", device=self.device).eval()
        self.matcher = feature.LightGlueMatcher("disk").to(self.device).eval()
        self.cache_dir = Path(cache_dir)
        self.features_dir = self.cache_dir / "features"
        self.pairs_dir = self.cache_dir / "pairs"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.pairs_dir.mkdir(parents=True, exist_ok=True)
        self.identity = {
            "kornia_version": kornia.__version__,
            "disk_sha256": _state_sha256(self.disk),
            "lightglue_sha256": _state_sha256(self.matcher),
            "long_side": LOCAL_LONG_SIDE,
            "max_features": LOCAL_MAX_FEATURES,
            "magsac_threshold_px": MAGSAC_THRESHOLD_PX,
            "magsac_confidence": MAGSAC_CONFIDENCE,
            "magsac_max_iters": MAGSAC_MAX_ITERS,
        }
        self.identity_sha256 = hashlib.sha256(
            json.dumps(self.identity, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _image_identity(self, path: str | Path) -> tuple[str, str]:
        image_sha = file_sha256(path)
        key = hashlib.sha256(f"{image_sha}:{self.identity_sha256}".encode()).hexdigest()
        return image_sha, key

    def features(self, path: str | Path) -> dict[str, np.ndarray]:
        image_sha, key = self._image_identity(path)
        cache = self.features_dir / f"{key}.npz"
        metadata_path = _metadata_path(cache)
        expected = {"image_sha256": image_sha, "verifier_sha256": self.identity_sha256}
        if cache.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if all(metadata.get(k) == v for k, v in expected.items()):
                if metadata.get("cache_sha256") == file_sha256(cache):
                    with np.load(cache) as arrays:
                        return {name: arrays[name].copy() for name in arrays.files}
        image = _resize_local_image(path).to(self.device)
        with torch.inference_mode():
            value = self.disk(image, n=LOCAL_MAX_FEATURES, pad_if_not_divisible=True)[0]
        arrays = {
            "keypoints": value.keypoints.detach().cpu().numpy().astype(np.float32),
            "descriptors": value.descriptors.detach().cpu().numpy().astype(np.float32),
            "scores": value.detection_scores.detach().cpu().numpy().astype(np.float32),
            "image_hw": np.asarray(image.shape[-2:], dtype=np.int32),
        }
        _atomic_npz(cache, **arrays)
        write_json({**expected, "cache_sha256": file_sha256(cache)}, metadata_path)
        return arrays

    def pair(self, first: str | Path, second: str | Path) -> tuple[float, int, int]:
        first_sha, first_key = self._image_identity(first)
        second_sha, second_key = self._image_identity(second)
        pair_key = hashlib.sha256(f"{first_key}:{second_key}".encode()).hexdigest()
        cache = self.pairs_dir / f"{pair_key}.json"
        expected = {
            "first_image_sha256": first_sha,
            "second_image_sha256": second_sha,
            "verifier_sha256": self.identity_sha256,
        }
        if cache.is_file():
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if all(payload.get(key) == value for key, value in expected.items()):
                return float(payload["score"]), int(payload["matches"]), int(payload["inliers"])
        first_features = self.features(first)
        second_features = self.features(second)
        if min(len(first_features["keypoints"]), len(second_features["keypoints"])) < 2:
            result = (0.0, 0, 0)
        else:
            keypoints_a = torch.from_numpy(first_features["keypoints"]).to(self.device)
            keypoints_b = torch.from_numpy(second_features["keypoints"]).to(self.device)
            lafs_a = self.feature.laf_from_center_scale_ori(keypoints_a[None])
            lafs_b = self.feature.laf_from_center_scale_ori(keypoints_b[None])
            descriptors_a = torch.from_numpy(first_features["descriptors"]).to(self.device)
            descriptors_b = torch.from_numpy(second_features["descriptors"]).to(self.device)
            with torch.inference_mode():
                _, matches = self.matcher(
                    descriptors_a, descriptors_b, lafs_a, lafs_b,
                    tuple(first_features["image_hw"]), tuple(second_features["image_hw"]),
                )
            result = geometric_verification_score(
                first_features["keypoints"], second_features["keypoints"],
                matches.detach().cpu().numpy().astype(np.int64),
            )
        write_json({
            **expected, "score": result[0], "matches": result[1], "inliers": result[2]
        }, cache)
        return result


def build_ragtr_geometry_cache(
    config,
    checkpoint: str | Path,
    condition: str,
    global_cache: str | Path,
    output: str | Path,
) -> Path:
    global_metadata, arrays = load_npz_cache(global_cache, "robotcar_ragtr_global_cache_v1")
    checkpoint = Path(checkpoint).expanduser().resolve()
    expected_identity = {
        "condition": condition,
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest_sha256": file_sha256(config.data.robotcar_manifest),
    }
    for key, value in expected_identity.items():
        if global_metadata.get(key) != value:
            raise ValueError(f"RA-GTR global cache mismatch for {key}")
    database, queries, _, _, _ = _ordered_rows(config, condition)
    candidate_indices, candidate_scores = nms_candidates(arrays)
    output = Path(output)
    verifier = LocalGeometryVerifier(output.parent / "local_geometry")
    expected = {
        "schema": "robotcar_ragtr_geometry_cache_v1",
        **expected_identity,
        "global_cache_sha256": file_sha256(global_cache),
        "verifier_sha256": verifier.identity_sha256,
        "top_k": max(RAGTR_TOP_K),
        "query_rows": len(queries),
    }
    metadata_path = _metadata_path(output)
    if output.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in expected.items()):
            if metadata.get("cache_sha256") == file_sha256(output):
                return output
    scores = np.zeros(candidate_indices.shape, dtype=np.float32)
    matches = np.zeros(candidate_indices.shape, dtype=np.int32)
    inliers = np.zeros(candidate_indices.shape, dtype=np.int32)
    for query_index, query in enumerate(queries):
        for rank, database_index in enumerate(candidate_indices[query_index]):
            if database_index < 0:
                continue
            score, match_count, inlier_count = verifier.pair(
                query.source_path, database[int(database_index)].source_path
            )
            scores[query_index, rank] = score
            matches[query_index, rank] = match_count
            inliers[query_index, rank] = inlier_count
    _atomic_npz(
        output,
        candidate_indices=candidate_indices,
        candidate_scores=candidate_scores,
        geometry_scores=scores,
        match_counts=matches,
        inlier_counts=inliers,
    )
    write_json({**expected, "verifier": verifier.identity, "cache_sha256": file_sha256(output)}, metadata_path)
    return output


def _normalize_rows(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.zeros(values.shape, dtype=np.float64)
    for row in range(len(values)):
        selected = values[row, valid[row]]
        if not len(selected):
            continue
        low, high = float(selected.min()), float(selected.max())
        if high > low:
            output[row, valid[row]] = (selected - low) / (high - low)
    return output


def utility_scores(candidate_scores: np.ndarray, geometry_scores: np.ndarray, alpha: float) -> np.ndarray:
    valid = np.isfinite(candidate_scores)
    appearance = _normalize_rows(candidate_scores, valid)
    geometry = _normalize_rows(geometry_scores, valid)
    output = alpha * appearance + (1.0 - alpha) * geometry
    no_geometry = np.logical_not((geometry_scores > 0).any(axis=1))
    output[no_geometry] = appearance[no_geometry]
    output[~valid] = -np.inf
    return output


def _huber(values: np.ndarray) -> np.ndarray:
    absolute = np.abs(values)
    return np.where(absolute <= 1.0, 0.5 * absolute**2, absolute - 0.5)


def _direction_paths(
    candidate_indices: np.ndarray,
    utilities: np.ndarray,
    route_progress_values: np.ndarray,
    route_ids: np.ndarray,
    motion_deltas: np.ndarray,
    direction: int,
    parameters: RAGTRParameters,
) -> tuple[np.ndarray, np.ndarray] | None:
    count, top_k = candidate_indices.shape
    valid_first = candidate_indices[0] >= 0
    scores = np.where(valid_first, utilities[0], -np.inf)
    paths = np.arange(top_k, dtype=np.int32)[:, None]
    for step in range(1, count):
        previous = candidate_indices[step - 1]
        current = candidate_indices[step]
        valid = (previous[:, None] >= 0) & (current[None, :] >= 0)
        safe_previous = np.maximum(previous, 0)
        safe_current = np.maximum(current, 0)
        valid &= route_ids[safe_previous][:, None] == route_ids[safe_current][None, :]
        signed = (
            route_progress_values[safe_current][None, :]
            - route_progress_values[safe_previous][:, None]
        )
        residual = np.abs(signed - direction * float(motion_deltas[step]))
        valid &= residual <= parameters.transition_radius_m
        transitions = scores[:, None] - parameters.motion_weight * _huber(
            residual / parameters.motion_sigma_m
        )
        transitions[~valid] = -np.inf
        parents = np.argmax(transitions, axis=0)
        best = transitions[parents, np.arange(top_k)]
        scores = best + utilities[step]
        scores[current < 0] = -np.inf
        paths = np.concatenate([paths[parents], np.arange(top_k, dtype=np.int32)[:, None]], axis=1)
        if not np.isfinite(scores).any():
            return None
    return scores, paths


def _sequence_ranges(sequence_starts: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(sequence_starts).tolist()
    if not starts or starts[0] != 0:
        raise ValueError("The first query must start a temporal sequence")
    return list(zip(starts, starts[1:] + [len(sequence_starts)]))


def rerank_ragtr(
    raw_indices: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    geometry_scores: np.ndarray,
    route_progress_values: np.ndarray,
    route_ids: np.ndarray,
    motion_deltas: np.ndarray,
    sequence_starts: np.ndarray,
    parameters: RAGTRParameters,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    query_count = len(raw_indices)
    if any(len(value) != query_count for value in (
        candidate_indices, candidate_scores, geometry_scores, motion_deltas, sequence_starts
    )):
        raise ValueError("RA-GTR arrays have incompatible query counts")
    candidates = candidate_indices[:, :parameters.top_k]
    utilities = utility_scores(
        candidate_scores[:, :parameters.top_k], geometry_scores[:, :parameters.top_k],
        parameters.alpha,
    )
    rankings = raw_indices.copy()
    diagnostics: list[dict] = [{} for _ in range(query_count)]
    timings = np.zeros(query_count, dtype=np.float64)
    for sequence_start, sequence_end in _sequence_ranges(sequence_starts):
        for target in range(sequence_start, sequence_end):
            started = time.perf_counter()
            observed_end = min(sequence_end, target + parameters.fixed_lag + 1)
            window_start = max(sequence_start, observed_end - parameters.window_size)
            target_offset = target - window_start
            window_indices = candidates[window_start:observed_end]
            window_utilities = utilities[window_start:observed_end]
            constrained: dict[int, tuple[float, int]] = {}
            for direction in (-1, 1):
                result = _direction_paths(
                    window_indices, window_utilities, route_progress_values, route_ids,
                    motion_deltas[window_start:observed_end], direction, parameters,
                )
                if result is None:
                    continue
                ending_scores, paths = result
                for endpoint in np.flatnonzero(np.isfinite(ending_scores)):
                    target_rank = int(paths[endpoint, target_offset])
                    candidate = int(window_indices[target_offset, target_rank])
                    value = float(ending_scores[endpoint])
                    if candidate >= 0 and (candidate not in constrained or value > constrained[candidate][0]):
                        constrained[candidate] = (value, direction)
            raw_top1 = int(raw_indices[target, 0])
            reason = "no_valid_path_raw_top1"
            selected_direction = None
            confidence = None
            if constrained and raw_top1 in constrained:
                ordered = sorted(constrained, key=lambda item: constrained[item][0], reverse=True)
                winner = ordered[0]
                confidence = (
                    constrained[winner][0] - constrained[raw_top1][0]
                ) / max(1, observed_end - window_start)
                if winner == raw_top1:
                    reason = "raw_top1_best_path"
                elif confidence >= parameters.confidence_threshold:
                    selected_direction = constrained[winner][1]
                    remaining = [int(value) for value in raw_indices[target] if int(value) not in ordered]
                    rankings[target] = np.asarray(
                        [*ordered, *remaining][:raw_indices.shape[1]], dtype=np.int32
                    )
                    reason = "path_override"
                else:
                    reason = "confidence_gate_raw_top1"
            elif constrained:
                reason = "raw_top1_incomparable"
            timings[target] = (time.perf_counter() - started) * 1000.0
            diagnostics[target] = {
                "query_index": target,
                "raw_top1": raw_top1,
                "output_top1": int(rankings[target, 0]),
                "window_used": observed_end - window_start,
                "history_frames": target - window_start,
                "future_frames": observed_end - target - 1,
                "fixed_lag": parameters.fixed_lag,
                "direction": selected_direction,
                "confidence": confidence,
                "reason": reason,
                "motion_delta_m": float(motion_deltas[target]),
            }
    return rankings, diagnostics, timings


def _candidate_only_rankings(
    raw_indices: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    geometry_scores: np.ndarray,
    top_k: int,
    alpha: float,
) -> np.ndarray:
    candidates = candidate_indices[:, :top_k]
    utilities = utility_scores(candidate_scores[:, :top_k], geometry_scores[:, :top_k], alpha)
    output = raw_indices.copy()
    for index in range(len(output)):
        valid = candidates[index] >= 0
        ranked = candidates[index, valid][np.argsort(utilities[index, valid])[::-1]].tolist()
        remaining = [int(value) for value in raw_indices[index] if int(value) not in ranked]
        output[index] = np.asarray([*ranked, *remaining][:raw_indices.shape[1]], dtype=np.int32)
    return output


def _metric_key(metrics: dict) -> tuple[float, float, float]:
    recalls = metrics["recalls"]
    return float(recalls["1"]), float(recalls["5"]), float(recalls["10"])


def _validate_geometry_identity(global_metadata: dict, geometry_metadata: dict) -> None:
    for field in ("condition", "checkpoint_sha256", "manifest_sha256"):
        if geometry_metadata.get(field) != global_metadata.get(field):
            raise ValueError(f"RA-GTR global/geometry cache mismatch for {field}")
    if geometry_metadata.get("global_cache_sha256") != global_metadata.get("cache_sha256"):
        raise ValueError("RA-GTR geometry cache was built from another global cache")


def select_robotcar_ragtr(
    config, checkpoint: str | Path, output: str | Path,
    global_cache: str | Path, geometry_cache: str | Path,
) -> Path:
    checkpoint = Path(checkpoint).expanduser().resolve()
    build_ragtr_global_cache(config, checkpoint, "val", global_cache)
    build_ragtr_geometry_cache(config, checkpoint, "val", global_cache, geometry_cache)
    global_metadata, arrays = load_npz_cache(global_cache, "robotcar_ragtr_global_cache_v1")
    geometry_metadata, geometry = load_npz_cache(
        geometry_cache, "robotcar_ragtr_geometry_cache_v1"
    )
    _validate_geometry_identity(global_metadata, geometry_metadata)
    database, queries, _, _, _ = _ordered_rows(config, "val")
    progress, route_ids = route_progress(database)

    geometry_trials = []
    for top_k, alpha in product(RAGTR_TOP_K, RAGTR_ALPHAS):
        rankings = _candidate_only_rankings(
            arrays["raw_indices"], geometry["candidate_indices"],
            geometry["candidate_scores"], geometry["geometry_scores"], top_k, alpha,
        )
        metrics = ranking_metrics(rankings, database, queries, 10.0, 30.0)
        geometry_trials.append({"top_k": top_k, "alpha": alpha, "metrics": metrics})
    geometry_trials.sort(
        key=lambda item: (*(-value for value in _metric_key(item["metrics"])),
                          item["top_k"], -item["alpha"])
    )
    selected_top_k = int(geometry_trials[0]["top_k"])
    selected_alpha = float(geometry_trials[0]["alpha"])

    temporal_trials = []
    for window, radius, weight, gate in product(
        RAGTR_WINDOWS, RAGTR_RADII_M, RAGTR_WEIGHTS, RAGTR_GATES
    ):
        parameters = RAGTRParameters(
            window, selected_top_k, selected_alpha, radius, weight, gate
        )
        rankings, diagnostics, timings = rerank_ragtr(
            arrays["raw_indices"], geometry["candidate_indices"],
            geometry["candidate_scores"], geometry["geometry_scores"],
            progress, route_ids, arrays["motion_deltas"], arrays["sequence_starts"], parameters,
        )
        metrics = ranking_metrics(rankings, database, queries, 10.0, 30.0)
        temporal_trials.append({
            "parameters": asdict(parameters),
            "metrics": metrics,
            "overrides": sum(item["reason"] == "path_override" for item in diagnostics),
            "rerank_p95_ms": float(np.percentile(timings, 95)),
        })
    temporal_trials.sort(key=lambda item: (
        *(-value for value in _metric_key(item["metrics"])),
        item["parameters"]["top_k"], item["parameters"]["window_size"],
        item["parameters"]["transition_radius_m"], item["parameters"]["motion_weight"],
        -item["parameters"]["confidence_threshold"],
    ))
    raw = ranking_metrics(arrays["raw_indices"], database, queries, 10.0, 30.0)
    payload = {
        "schema": "robotcar_ragtr_selection_v1",
        "selection_split": "val",
        "selection_metric": "10m+30deg_R1_R5_R10",
        "selected_parameters": temporal_trials[0]["parameters"],
        "selected_metrics": temporal_trials[0]["metrics"],
        "single_frame_metrics": raw,
        "checkpoint_sha256": global_metadata["checkpoint_sha256"],
        "manifest_sha256": global_metadata["manifest_sha256"],
        "query_protocol_fingerprint": global_metadata["query_protocol_fingerprint"],
        "global_cache_sha256": global_metadata["cache_sha256"],
        "geometry_cache_sha256": geometry_metadata["cache_sha256"],
        "verifier_sha256": geometry_metadata["verifier_sha256"],
        "geometry_trials": geometry_trials,
        "temporal_trials": temporal_trials,
        "test_results_used_for_selection": False,
    }
    write_json(payload, output)
    return Path(output)


def _validate_ragtr_selection(selection: dict, global_metadata: dict, geometry_metadata: dict):
    if selection.get("schema") != "robotcar_ragtr_selection_v1" or selection.get("selection_split") != "val":
        raise ValueError("RA-GTR parameters must come from RobotCar val")
    for field in ("checkpoint_sha256", "manifest_sha256", "verifier_sha256"):
        actual = geometry_metadata.get(field, global_metadata.get(field))
        if selection.get(field) != actual:
            raise ValueError(f"RA-GTR selection mismatch for {field}")
    parameters = RAGTRParameters(**selection["selected_parameters"])
    if parameters.fixed_lag != RAGTR_FIXED_LAG or parameters.motion_sigma_m != RAGTR_MOTION_SIGMA_M:
        raise ValueError("RA-GTR selection changes a fixed parameter")
    return parameters


def _raw_oracle(raw_indices: np.ndarray, database, queries) -> dict:
    from .robotcar_temporal import _positive_indices
    positives = _positive_indices(database, queries, 10.0, 30.0)
    valid = np.asarray([len(items) > 0 for items in positives])
    recalls = {}
    for count in (1, 5, 10, 20, 100):
        hits = np.asarray([
            bool(valid[index] and np.isin(raw_indices[index, :count], positives[index]).any())
            for index in range(len(queries))
        ])
        recalls[str(count)] = float(hits[valid].mean())
    return {"recalls": recalls, "valid_queries": int(valid.sum())}


def evaluate_ragtr_grid(rankings: np.ndarray, database, queries) -> dict:
    output = {}
    for distance, yaw in product(RAGTR_DISTANCES_M, RAGTR_YAWS_DEG):
        yaw_text = "none" if yaw is None else str(int(yaw))
        output[f"distance_{int(distance)}m_yaw_{yaw_text}deg"] = {
            **ranking_metrics(rankings, database, queries, distance, yaw),
            "positive_radius_m": distance,
            "positive_yaw_deg": yaw,
        }
    return output


def evaluate_robotcar_ragtr(
    config, checkpoint: str | Path, selection_path: str | Path, condition: str,
    output: str | Path, global_cache: str | Path, geometry_cache: str | Path,
) -> Path:
    if condition not in {"night", "day"}:
        raise ValueError("Formal RA-GTR evaluation condition must be night or day")
    checkpoint = Path(checkpoint).expanduser().resolve()
    build_ragtr_global_cache(config, checkpoint, condition, global_cache)
    build_ragtr_geometry_cache(config, checkpoint, condition, global_cache, geometry_cache)
    global_metadata, arrays = load_npz_cache(global_cache, "robotcar_ragtr_global_cache_v1")
    geometry_metadata, geometry = load_npz_cache(
        geometry_cache, "robotcar_ragtr_geometry_cache_v1"
    )
    _validate_geometry_identity(global_metadata, geometry_metadata)
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    parameters = _validate_ragtr_selection(selection, global_metadata, geometry_metadata)
    database, queries, split, _, _ = _ordered_rows(config, condition)
    progress, route_ids = route_progress(database)
    rankings, decisions, timings = rerank_ragtr(
        arrays["raw_indices"], geometry["candidate_indices"], geometry["candidate_scores"],
        geometry["geometry_scores"], progress, route_ids, arrays["motion_deltas"],
        arrays["sequence_starts"], parameters,
    )
    payload = {
        "schema": "robotcar_ragtr_evaluation_v1",
        "method": "ragtr",
        "condition": condition,
        "split": split,
        "development_result": True,
        "parameters": asdict(parameters),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": global_metadata["checkpoint_sha256"],
        "manifest_sha256": global_metadata["manifest_sha256"],
        "query_protocol_fingerprint": global_metadata["query_protocol_fingerprint"],
        "global_cache_sha256": global_metadata["cache_sha256"],
        "geometry_cache_sha256": geometry_metadata["cache_sha256"],
        "verifier_sha256": geometry_metadata["verifier_sha256"],
        "selection_sha256": file_sha256(selection_path),
        "database": len(database),
        "queries": len(queries),
        "single_frame": {"grid": evaluate_ragtr_grid(arrays["raw_indices"], database, queries)},
        "ragtr": {"grid": evaluate_ragtr_grid(rankings, database, queries), "decisions": decisions},
        "raw_oracle_10m_30deg": _raw_oracle(arrays["raw_indices"], database, queries),
        "resources": {
            "model_parameters": global_metadata.get("model_parameters"),
            "descriptor_dim": RAGTR_DESCRIPTOR_DIM,
            "checkpoint_mib": checkpoint.stat().st_size / 1024**2,
            "database_descriptor_mib_fp16": len(database) * RAGTR_DESCRIPTOR_DIM * 2 / 1024**2,
            "database_descriptor_mib_fp32": len(database) * RAGTR_DESCRIPTOR_DIM * 4 / 1024**2,
            "rerank_cpu_p50_ms": float(np.percentile(timings, 50)),
            "rerank_cpu_p95_ms": float(np.percentile(timings, 95)),
            "fixed_output_lag_frames": RAGTR_FIXED_LAG,
            "local_verifier": "Kornia DISK-depth + LightGlue-disk + USAC_MAGSAC",
        },
    }
    write_json(payload, output)
    return Path(output)


def summarize_robotcar_ragtr(
    night_path: str | Path, day_path: str | Path, salad_path: str | Path,
    boq_path: str | Path, output: str | Path,
) -> Path:
    night = json.loads(Path(night_path).read_text(encoding="utf-8"))
    day = json.loads(Path(day_path).read_text(encoding="utf-8"))
    if night.get("condition") != "night" or day.get("condition") != "day":
        raise ValueError("RA-GTR summary requires frozen night and day evaluations")
    for field in ("checkpoint_sha256", "manifest_sha256", "selection_sha256", "verifier_sha256"):
        if night.get(field) != day.get(field):
            raise ValueError(f"RA-GTR night/day mismatch for {field}")
    expected_protocol = night.get("query_protocol_fingerprint")
    if not expected_protocol:
        raise ValueError("RA-GTR night result is missing the query protocol fingerprint")

    def reference(path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metrics = payload["datasets"]["robotcar_project"]
        fingerprint = metrics.get(
            "query_protocol_fingerprint", metrics.get("protocol_fingerprint")
        )
        if fingerprint != expected_protocol:
            raise ValueError(f"RA-GTR/reference protocol mismatch: {path}")
        return float(metrics["recalls"]["1"])
    salad = reference(salad_path)
    boq = reference(boq_path)
    primary = "distance_10m_yaw_30deg"
    single_night = float(night["single_frame"]["grid"][primary]["recalls"]["1"])
    ragtr_night = float(night["ragtr"]["grid"][primary]["recalls"]["1"])
    single_day = float(day["single_frame"]["grid"][primary]["recalls"]["1"])
    ragtr_day = float(day["ragtr"]["grid"][primary]["recalls"]["1"])
    acceptance = {
        "night_strictly_exceeds_salad": ragtr_night > salad,
        "day_drop_within_0_5pp": ragtr_day >= single_day - 0.005,
        "parameters_frozen_on_val": True,
        "salad_or_boq_used_at_inference": False,
    }
    acceptance["adopt_as_robotcar_method"] = all(acceptance.values())
    write_json({
        "schema": "robotcar_ragtr_comparison_v1",
        "development_result": True,
        "primary_protocol": "10m+30deg_R1",
        "mixvpr_4096_cdnd": {
            "single_frame_recall1": single_night,
            "ragtr_recall1": ragtr_night,
            "delta": ragtr_night - single_night,
            "day_single_frame_recall1": single_day,
            "day_ragtr_recall1": ragtr_day,
            "raw_oracle": night["raw_oracle_10m_30deg"],
            "resources": night["resources"],
        },
        "single_frame_references": {"salad": salad, "boq": boq},
        "claims": {
            "exceeds_salad_single_frame": ragtr_night > salad,
            "exceeds_boq_single_frame": ragtr_night > boq,
            "same_reranker_comparison": False,
        },
        "acceptance": acceptance,
    }, output)
    return Path(output)


def runtime_identity() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__}


def ordinal_grid(rankings: np.ndarray) -> dict:
    truth = np.arange(len(rankings))
    output = {}
    for tolerance in (0, 2, 5, 10):
        recalls = {}
        for count in (1, 5, 10):
            hits = (np.abs(rankings[:, :count] - truth[:, None]) <= tolerance).any(axis=1)
            recalls[str(count)] = float(hits.mean())
        output[f"frame_tolerance_{tolerance}"] = {
            "frame_tolerance": tolerance,
            "recalls": recalls,
            "total_queries": len(rankings),
        }
    return output


def _gardens_geometry(
    database_rows,
    query_rows,
    candidate_indices: np.ndarray,
    cache_dir: str | Path,
) -> tuple[np.ndarray, dict]:
    verifier = LocalGeometryVerifier(cache_dir)
    scores = np.zeros(candidate_indices.shape, dtype=np.float32)
    for query_index, query in enumerate(query_rows):
        for rank, database_index in enumerate(candidate_indices[query_index]):
            if database_index < 0:
                continue
            scores[query_index, rank] = verifier.pair(
                query.source_path, database_rows[int(database_index)].source_path
            )[0]
    return scores, {"verifier": verifier.identity, "verifier_sha256": verifier.identity_sha256}


def evaluate_gardens_ragtr_rankings(
    groups,
    val_indices: np.ndarray,
    val_scores: np.ndarray,
    test_indices: np.ndarray,
    test_scores: np.ndarray,
    identity: dict,
    output_dir: str | Path,
) -> dict:
    """Select ordinal RA-GTR on day-right, then apply once to night-right."""
    output_dir = Path(output_dir)
    top_k = max(RAGTR_TOP_K)
    val_candidates = val_indices[:, :top_k].astype(np.int32)
    test_candidates = test_indices[:, :top_k].astype(np.int32)
    val_candidate_scores = val_scores[:, :top_k].astype(np.float32)
    test_candidate_scores = test_scores[:, :top_k].astype(np.float32)
    val_geometry, verifier_identity = _gardens_geometry(
        groups["day_left"], groups["day_right"], val_candidates,
        output_dir / "local_geometry",
    )
    geometry_trials = []
    for candidate_count, alpha in product(RAGTR_TOP_K, RAGTR_ALPHAS):
        rankings = _candidate_only_rankings(
            val_indices, val_candidates, val_candidate_scores, val_geometry,
            candidate_count, alpha,
        )
        recalls = ordinal_grid(rankings)["frame_tolerance_0"]["recalls"]
        geometry_trials.append({"top_k": candidate_count, "alpha": alpha, "recalls": recalls})
    geometry_trials.sort(key=lambda item: (
        -item["recalls"]["1"], -item["recalls"]["5"], -item["recalls"]["10"],
        item["top_k"], -item["alpha"],
    ))
    selected_top_k = int(geometry_trials[0]["top_k"])
    selected_alpha = float(geometry_trials[0]["alpha"])
    count = len(val_indices)
    progress = np.arange(len(groups["day_left"]), dtype=np.float64)
    route_ids = np.zeros(len(progress), dtype=np.int32)
    motion = np.r_[0, np.ones(count - 1)].astype(np.float32)
    starts = np.r_[True, np.zeros(count - 1, dtype=np.bool_)]
    temporal_trials = []
    for window, weight, gate in product(RAGTR_WINDOWS, RAGTR_WEIGHTS, RAGTR_GATES):
        parameters = RAGTRParameters(
            window, selected_top_k, selected_alpha,
            transition_radius_m=2.0, motion_weight=weight,
            confidence_threshold=gate, motion_sigma_m=1.0,
            nms_radius_m=0.0, nms_yaw_deg=0.0,
        )
        rankings, _, timings = rerank_ragtr(
            val_indices, val_candidates, val_candidate_scores, val_geometry,
            progress, route_ids, motion, starts, parameters,
        )
        grid = ordinal_grid(rankings)
        recalls = grid["frame_tolerance_0"]["recalls"]
        temporal_trials.append({
            "parameters": asdict(parameters), "recalls": recalls,
            "rerank_p95_ms": float(np.percentile(timings, 95)),
        })
    temporal_trials.sort(key=lambda item: (
        -item["recalls"]["1"], -item["recalls"]["5"], -item["recalls"]["10"],
        item["parameters"]["top_k"], item["parameters"]["window_size"],
        item["parameters"]["motion_weight"],
        -item["parameters"]["confidence_threshold"],
    ))
    parameters = RAGTRParameters(**temporal_trials[0]["parameters"])
    selection_path = output_dir / "day_right_ragtr_selection.json"
    write_json({
        "schema": "gardens_ragtr_selection_v1",
        "selection_split": "day_right",
        "night_right_used_for_selection": False,
        "selected_parameters": asdict(parameters),
        "geometry_trials": geometry_trials,
        "temporal_trials": temporal_trials,
        **identity,
        **verifier_identity,
    }, selection_path)

    # Night-right is not touched until every selection parameter is serialized and frozen.
    test_geometry, test_verifier_identity = _gardens_geometry(
        groups["day_left"], groups["night_right"], test_candidates,
        output_dir / "local_geometry",
    )
    if verifier_identity["verifier_sha256"] != test_verifier_identity["verifier_sha256"]:
        raise RuntimeError("Gardens local verifier changed between val and night")

    val_rankings, val_decisions, val_timings = rerank_ragtr(
        val_indices, val_candidates, val_candidate_scores, val_geometry,
        progress, route_ids, motion, starts, parameters,
    )
    test_motion = np.r_[0, np.ones(len(test_indices) - 1)].astype(np.float32)
    test_starts = np.r_[True, np.zeros(len(test_indices) - 1, dtype=np.bool_)]
    test_rankings, test_decisions, test_timings = rerank_ragtr(
        test_indices, test_candidates, test_candidate_scores, test_geometry,
        progress, route_ids, test_motion, test_starts, parameters,
    )
    window_sweep = {}
    for window in RAGTR_WINDOWS:
        trial_parameters = RAGTRParameters(**{**asdict(parameters), "window_size": window})
        val_window = rerank_ragtr(
            val_indices, val_candidates, val_candidate_scores, val_geometry,
            progress, route_ids, motion, starts, trial_parameters,
        )[0]
        test_window = rerank_ragtr(
            test_indices, test_candidates, test_candidate_scores, test_geometry,
            progress, route_ids, test_motion, test_starts, trial_parameters,
        )[0]
        window_sweep[str(window)] = {
            "day_right": ordinal_grid(val_window),
            "night_right": ordinal_grid(test_window),
        }
    return {
        "temporal_method": "ragtr",
        "selection_sha256": file_sha256(selection_path),
        "selected_parameters": asdict(parameters),
        "primary_window_size": parameters.window_size,
        "window_sweep": window_sweep,
        "day_right": {
            "single_frame": ordinal_grid(val_indices),
            "temporal": ordinal_grid(val_rankings),
            "decisions": val_decisions,
            "rerank_p95_ms": float(np.percentile(val_timings, 95)),
        },
        "night_right": {
            "single_frame": ordinal_grid(test_indices),
            "temporal": ordinal_grid(test_rankings),
            "decisions": test_decisions,
            "rerank_p95_ms": float(np.percentile(test_timings, 95)),
        },
        **verifier_identity,
    }
