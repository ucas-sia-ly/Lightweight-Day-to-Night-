from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import torch

from .checkpointing import file_sha256, write_json
from .evaluation import extract_descriptors
from .reference_models import query_protocol_fingerprint, suite_rows
from .runtime import load_vpr_model
from .transforms import VPRTransform


RAW_TOP_K = 100
NMS_RADIUS_M = 5.0
NMS_YAW_DEG = 30.0
MOTION_SIGMA_M = 5.0
DISTANCES_M = (5.0, 10.0, 25.0, 50.0)
YAWS_DEG = (10.0, 30.0, None)
RECALLS = (1, 5, 10)


@dataclass(frozen=True)
class TemporalParameters:
    window_size: int
    top_k: int
    transition_radius_m: float
    motion_weight: float
    raw_top_k: int = RAW_TOP_K
    nms_radius_m: float = NMS_RADIUS_M
    nms_yaw_deg: float = NMS_YAW_DEG
    motion_sigma_m: float = MOTION_SIGMA_M


def _condition_rows(config, condition: str):
    if condition == "val":
        return (*suite_rows(config, "robotcar", "val"), "val")
    if condition == "night":
        return (*suite_rows(config, "robotcar", "test"), "test")
    if condition == "day":
        return (*suite_rows(config, "robotcar_day", "test"), "day")
    raise ValueError("RobotCar temporal condition must be val, night, or day")


def ordered_motion_deltas(queries) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return temporal order, sequence starts, and scalar motion increments only."""
    groups: dict[str, list[tuple[int, object]]] = {}
    for index, row in enumerate(queries):
        groups.setdefault(str(getattr(row, "traverse", "single")), []).append((index, row))
    order = []
    starts = []
    deltas = []
    for traverse in sorted(groups):
        items = sorted(
            groups[traverse],
            key=lambda item: (int(getattr(item[1], "timestamp", item[0])), item[0]),
        )
        previous = None
        for index, row in items:
            order.append(index)
            starts.append(previous is None)
            if previous is None:
                deltas.append(0.0)
            else:
                deltas.append(float(np.hypot(
                    float(row.easting) - float(previous.easting),
                    float(row.northing) - float(previous.northing),
                )))
            previous = row
    return (
        np.asarray(order, dtype=np.int64),
        np.asarray(starts, dtype=np.bool_),
        np.asarray(deltas, dtype=np.float32),
    )


def _top_scores(database: np.ndarray, queries: np.ndarray, count: int):
    count = min(int(count), len(database))
    try:
        import faiss

        index = faiss.IndexFlatIP(database.shape[1])
        index.add(np.asarray(database, dtype=np.float32))
        scores, indices = index.search(np.asarray(queries, dtype=np.float32), count)
    except ImportError:
        similarities = np.asarray(queries, dtype=np.float32) @ np.asarray(database, dtype=np.float32).T
        indices = np.argpartition(-similarities, count - 1, axis=1)[:, :count]
        scores = np.take_along_axis(similarities, indices, axis=1)
        ordering = np.argsort(scores, axis=1)[:, ::-1]
        indices = np.take_along_axis(indices, ordering, axis=1)
        scores = np.take_along_axis(scores, ordering, axis=1)
    return indices.astype(np.int32), scores.astype(np.float32)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _cache_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".json")


def _expected_cache_metadata(config, checkpoint: Path, condition: str, database, queries) -> dict:
    query_fingerprint = query_protocol_fingerprint(database, queries, 10.0, 30.0)
    return {
        "schema": "robotcar_temporal_cache_v1",
        "condition": condition,
        "descriptor_dim": 512,
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest_sha256": file_sha256(config.data.robotcar_manifest),
        "query_protocol_fingerprint": query_fingerprint,
        "database_rows": len(database),
        "query_rows": len(queries),
        "raw_top_k": min(RAW_TOP_K, len(database)),
        "input_size": [config.vpr.image_height, config.vpr.image_width],
    }


def build_robotcar_temporal_cache(config, checkpoint: str | Path, condition: str, cache: str | Path) -> Path:
    if config.vpr.descriptor_dim != 512:
        raise ValueError("MGTR requires a 512-dimensional MixVPR checkpoint")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    database, original_queries, split = _condition_rows(config, condition)
    order, starts, deltas = ordered_motion_deltas(original_queries)
    queries = [original_queries[int(index)] for index in order]
    expected = _expected_cache_metadata(config, checkpoint_path, condition, database, queries)
    expected["split"] = split
    cache_path = Path(cache)
    metadata_path = _cache_metadata_path(cache_path)
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            all(metadata.get(key) == value for key, value in expected.items())
            and metadata.get("cache_sha256") == file_sha256(cache_path)
        ):
            with np.load(cache_path) as arrays:
                if (
                    arrays["raw_indices"].shape == (len(queries), expected["raw_top_k"])
                    and arrays["raw_scores"].shape == arrays["raw_indices"].shape
                    and arrays["motion_deltas"].shape == (len(queries),)
                ):
                    return cache_path

    model, state, _ = load_vpr_model(config, checkpoint_path)
    if int(getattr(model, "descriptor_dim", 0)) != 512 or int(state.get("descriptor_dim", 512)) != 512:
        raise ValueError("MGTR checkpoint descriptor dimension must be 512")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    transform = VPRTransform(config.vpr.image_height, config.vpr.image_width)
    batch_size = max(1, config.vpr.batch_size * 2)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    database_descriptors = extract_descriptors(
        model, database, transform, device, batch_size, config.vpr.workers
    )
    query_descriptors = extract_descriptors(
        model, queries, transform, device, batch_size, config.vpr.workers
    )
    retrieval_started = time.perf_counter()
    descriptor_elapsed = retrieval_started - started
    raw_indices, raw_scores = _top_scores(
        database_descriptors, query_descriptors, expected["raw_top_k"]
    )
    retrieval_elapsed = time.perf_counter() - retrieval_started
    db_xy = np.asarray(
        [(float(row.easting), float(row.northing)) for row in database], dtype=np.float64
    )
    db_yaw = np.asarray([float(getattr(row, "yaw_deg", 0.0)) for row in database], dtype=np.float32)
    _atomic_npz(
        cache_path,
        raw_indices=raw_indices,
        raw_scores=raw_scores,
        database_xy=db_xy,
        database_yaw=db_yaw,
        motion_deltas=deltas,
        sequence_starts=starts,
    )
    metadata = {
        **expected,
        "split": split,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "descriptor_extraction_seconds": descriptor_elapsed,
        "top100_search_total_ms": retrieval_elapsed * 1000.0,
        "top100_search_ms_per_query": retrieval_elapsed * 1000.0 / max(1, len(queries)),
        "encoded_images": len(database) + len(queries),
        "peak_cuda_mib": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if device.type == "cuda" else None
        ),
        "cache_sha256": file_sha256(cache_path),
    }
    write_json(metadata, metadata_path)
    return cache_path


def load_robotcar_temporal_cache(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    cache_path = Path(path)
    metadata = json.loads(_cache_metadata_path(cache_path).read_text(encoding="utf-8"))
    if metadata.get("cache_sha256") != file_sha256(cache_path):
        raise RuntimeError("RobotCar temporal cache SHA256 mismatch")
    with np.load(cache_path) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    return metadata, arrays


def circular_yaw_difference(first: float, second: float) -> float:
    return abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)


def spatial_nms(
    raw_indices: np.ndarray,
    raw_scores: np.ndarray,
    database_xy: np.ndarray,
    database_yaw: np.ndarray,
    top_k: int,
    *,
    enabled: bool = True,
    radius_m: float = NMS_RADIUS_M,
    yaw_deg: float = NMS_YAW_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    if not enabled:
        return raw_indices[:top_k].astype(np.int64), raw_scores[:top_k].astype(np.float64)
    kept_indices = []
    kept_scores = []
    radius_squared = float(radius_m) ** 2
    for raw_index, raw_score in zip(raw_indices, raw_scores):
        index = int(raw_index)
        duplicate = False
        x, y = database_xy[index]
        for kept in kept_indices:
            kept_x, kept_y = database_xy[kept]
            if (
                (x - kept_x) ** 2 + (y - kept_y) ** 2 <= radius_squared
                and circular_yaw_difference(database_yaw[index], database_yaw[kept]) <= yaw_deg
            ):
                duplicate = True
                break
        if not duplicate:
            kept_indices.append(index)
            kept_scores.append(float(raw_score))
            if len(kept_indices) == top_k:
                break
    return np.asarray(kept_indices, dtype=np.int64), np.asarray(kept_scores, dtype=np.float64)


def _appearance_scores(scores: np.ndarray) -> np.ndarray:
    if not len(scores):
        return scores.astype(np.float64)
    denominator = float(scores[0] - scores[-1])
    return (scores - scores[-1]) / (denominator + 1e-8)


def _huber(values: np.ndarray) -> np.ndarray:
    absolute = np.abs(values)
    return np.where(absolute <= 1.0, 0.5 * absolute**2, absolute - 0.5)


def _ending_path_scores(
    candidate_indices: list[np.ndarray],
    candidate_scores: list[np.ndarray],
    database_xy: np.ndarray,
    motion_deltas: np.ndarray,
    parameters: TemporalParameters,
) -> np.ndarray | None:
    scores = _appearance_scores(candidate_scores[0])
    if not len(scores):
        return None
    for step in range(1, len(candidate_indices)):
        previous = candidate_indices[step - 1]
        current = candidate_indices[step]
        if not len(previous) or not len(current):
            return None
        distances = np.linalg.norm(
            database_xy[previous][:, None, :] - database_xy[current][None, :, :], axis=2
        )
        delta = float(motion_deltas[step])
        valid = distances <= delta + parameters.transition_radius_m
        residual = np.abs(distances - delta) / parameters.motion_sigma_m
        transitions = scores[:, None] - parameters.motion_weight * _huber(residual)
        transitions[~valid] = -np.inf
        best = transitions.max(axis=0)
        scores = _appearance_scores(candidate_scores[step]) + best
        if not np.isfinite(scores).any():
            return None
    return scores


def _append_raw_candidates(ranked: np.ndarray, raw: np.ndarray) -> np.ndarray:
    output = [int(value) for value in ranked]
    seen = set(output)
    for value in raw:
        value = int(value)
        if value not in seen:
            output.append(value)
            seen.add(value)
    return np.asarray(output[: len(raw)], dtype=np.int32)


def rerank_temporal(
    raw_indices: np.ndarray,
    raw_scores: np.ndarray,
    database_xy: np.ndarray,
    database_yaw: np.ndarray,
    motion_deltas: np.ndarray,
    sequence_starts: np.ndarray,
    parameters: TemporalParameters,
    *,
    use_nms: bool = True,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """Rerank from scalar motion only; query absolute positions are intentionally absent."""
    query_count, raw_count = raw_indices.shape
    if any(len(value) != query_count for value in (raw_scores, motion_deltas, sequence_starts)):
        raise ValueError("Temporal cache arrays have incompatible query counts")
    candidates = []
    nms_timings = np.empty(query_count, dtype=np.float64)
    for index in range(query_count):
        nms_started = time.perf_counter()
        candidates.append(spatial_nms(
            raw_indices[index], raw_scores[index], database_xy, database_yaw,
            parameters.top_k, enabled=use_nms,
            radius_m=parameters.nms_radius_m, yaw_deg=parameters.nms_yaw_deg,
        ))
        nms_timings[index] = (time.perf_counter() - nms_started) * 1000.0
    rankings = np.empty((query_count, raw_count), dtype=np.int32)
    diagnostics = []
    timings = np.empty(query_count, dtype=np.float64)
    sequence_start = 0
    for query_index in range(query_count):
        started = time.perf_counter()
        if bool(sequence_starts[query_index]):
            sequence_start = query_index
        available = query_index - sequence_start + 1
        maximum_window = min(parameters.window_size, available)
        if maximum_window == 1:
            rankings[query_index] = raw_indices[query_index]
            reason = "warmup_single"
            used_window = 1
        else:
            ending_scores = None
            used_window = 1
            for window in range(maximum_window, 1, -1):
                start = query_index - window + 1
                ending_scores = _ending_path_scores(
                    [candidates[item][0] for item in range(start, query_index + 1)],
                    [candidates[item][1] for item in range(start, query_index + 1)],
                    database_xy,
                    motion_deltas[start : query_index + 1],
                    parameters,
                )
                if ending_scores is not None:
                    used_window = window
                    break
            if ending_scores is None:
                rankings[query_index] = raw_indices[query_index]
                reason = "no_valid_transition_raw_top1"
            else:
                current_candidates = candidates[query_index][0]
                order = np.argsort(ending_scores)[::-1]
                rankings[query_index] = _append_raw_candidates(
                    current_candidates[order], raw_indices[query_index]
                )
                if used_window < maximum_window:
                    reason = "shortened_window"
                elif available < parameters.window_size:
                    reason = "warmup_path"
                else:
                    reason = "full_window"
        timings[query_index] = (
            (time.perf_counter() - started) * 1000.0 + nms_timings[query_index]
        )
        diagnostics.append({
            "query_index": query_index,
            "raw_top1": int(raw_indices[query_index, 0]),
            "output_top1": int(rankings[query_index, 0]),
            "window_used": used_window,
            "reason": reason,
            "motion_delta_m": float(motion_deltas[query_index]),
        })
    return rankings, diagnostics, timings


def _positive_indices(database, queries, distance: float, yaw: float | None):
    db_xy = np.asarray([(float(row.easting), float(row.northing)) for row in database])
    db_yaw = np.asarray([float(getattr(row, "yaw_deg", 0.0)) for row in database])
    positives = []
    for query in queries:
        distances = np.linalg.norm(
            db_xy - np.asarray([float(query.easting), float(query.northing)]), axis=1
        )
        valid = distances <= distance
        if yaw is not None:
            yaw_difference = np.abs(
                (db_yaw - float(getattr(query, "yaw_deg", 0.0)) + 180.0) % 360.0 - 180.0
            )
            valid &= yaw_difference <= yaw
        positives.append(np.flatnonzero(valid))
    return positives


def ranking_metrics(rankings: np.ndarray, database, queries, distance: float, yaw: float | None):
    positives = _positive_indices(database, queries, distance, yaw)
    valid = np.asarray([len(items) > 0 for items in positives], dtype=bool)
    recalls = {}
    success = {}
    for recall in RECALLS:
        hits = np.asarray([
            bool(valid[index] and np.isin(rankings[index, :recall], positives[index]).any())
            for index in range(len(queries))
        ])
        recalls[str(recall)] = float(hits[valid].mean())
        success[str(recall)] = hits.astype(np.uint8).tolist()
    return {
        "recalls": recalls,
        "valid_queries": int(valid.sum()),
        "total_queries": len(queries),
        "success_by_k": success,
        "valid_by_query": valid.astype(np.uint8).tolist(),
    }


def evaluate_grid(rankings: np.ndarray, database, queries) -> dict:
    output = {}
    for distance, yaw in product(DISTANCES_M, YAWS_DEG):
        distance_text = str(int(distance))
        yaw_text = "none" if yaw is None else str(int(yaw))
        output[f"distance_{distance_text}m_yaw_{yaw_text}deg"] = {
            **ranking_metrics(rankings, database, queries, distance, yaw),
            "positive_radius_m": distance,
            "positive_yaw_deg": yaw,
        }
    return output


def _validate_cache_for_run(metadata: dict, config, checkpoint: Path, condition: str) -> None:
    expected = {
        "condition": condition,
        "descriptor_dim": 512,
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest_sha256": file_sha256(config.data.robotcar_manifest),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items() if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"RobotCar temporal cache mismatch: {mismatches}")


def _selection_sort_key(item: dict) -> tuple:
    parameters = item["parameters"]
    return (
        -item["recall1"],
        parameters["top_k"],
        parameters["window_size"],
        parameters["transition_radius_m"],
        parameters["motion_weight"],
    )


def select_robotcar_temporal(
    config, checkpoint: str | Path, output: str | Path, cache: str | Path
) -> Path:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    cache_path = build_robotcar_temporal_cache(config, checkpoint_path, "val", cache)
    metadata, arrays = load_robotcar_temporal_cache(cache_path)
    _validate_cache_for_run(metadata, config, checkpoint_path, "val")
    database, original_queries, _ = _condition_rows(config, "val")
    order, _, _ = ordered_motion_deltas(original_queries)
    queries = [original_queries[int(index)] for index in order]
    trials = []
    for window_size, top_k, radius, weight in product(
        (3, 4), (5, 10, 20), (15.0, 30.0, 50.0), (0.5, 1.0)
    ):
        parameters = TemporalParameters(window_size, top_k, radius, weight)
        rankings, diagnostics, timings = rerank_temporal(
            arrays["raw_indices"], arrays["raw_scores"], arrays["database_xy"],
            arrays["database_yaw"], arrays["motion_deltas"], arrays["sequence_starts"],
            parameters,
        )
        primary = ranking_metrics(rankings, database, queries, 10.0, 30.0)
        trials.append({
            "parameters": asdict(parameters),
            "recall1": primary["recalls"]["1"],
            "fallback_queries": sum(
                item["reason"] == "no_valid_transition_raw_top1" for item in diagnostics
            ),
            "rerank_p95_ms": float(np.percentile(timings, 95)),
        })
    trials.sort(key=_selection_sort_key)
    single = ranking_metrics(arrays["raw_indices"], database, queries, 10.0, 30.0)
    payload = {
        "schema": "robotcar_temporal_selection_v1",
        "selection_split": "val",
        "selection_metric": "recall1_10m_30deg",
        "selected_parameters": trials[0]["parameters"],
        "selected_recall1": trials[0]["recall1"],
        "single_frame_recall1": single["recalls"]["1"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "manifest_sha256": metadata["manifest_sha256"],
        "val_query_protocol_fingerprint": metadata["query_protocol_fingerprint"],
        "cache": str(cache_path.resolve()),
        "trials": trials,
    }
    write_json(payload, output)
    return Path(output)


def _validate_selection(selection: dict, metadata: dict) -> TemporalParameters:
    if selection.get("selection_split") != "val":
        raise ValueError("MGTR parameters must be selected on RobotCar val")
    for field in ("checkpoint_sha256", "manifest_sha256"):
        if selection.get(field) != metadata.get(field):
            raise ValueError(f"MGTR selection/cache mismatch for {field}")
    parameters = TemporalParameters(**selection["selected_parameters"])
    fixed = (
        parameters.raw_top_k == RAW_TOP_K
        and parameters.nms_radius_m == NMS_RADIUS_M
        and parameters.nms_yaw_deg == NMS_YAW_DEG
        and parameters.motion_sigma_m == MOTION_SIGMA_M
    )
    if not fixed:
        raise ValueError("MGTR selection has incompatible fixed parameters")
    return parameters


def temporal_resource_summary(
    parameters: TemporalParameters,
    database_count: int,
    checkpoint_mib: float,
    metadata: dict,
    benchmark: dict,
    timings: np.ndarray,
) -> dict:
    state_bytes = (
        parameters.window_size * parameters.top_k * (8 + 4)
        + max(0, parameters.window_size - 1) * 4
    )
    return {
        "model_parameters": metadata.get("model_parameters"),
        "checkpoint_mib": checkpoint_mib,
        "batch1_encoding_p50_ms": benchmark.get("latency_ms", {}).get("p50"),
        "peak_cuda_mib": metadata.get("peak_cuda_mib", benchmark.get("peak_cuda_mib")),
        "top100_search_ms_per_query": metadata.get("top100_search_ms_per_query"),
        "rerank_cpu_p50_ms": float(np.percentile(timings, 50)),
        "rerank_cpu_p95_ms": float(np.percentile(timings, 95)),
        "database_descriptor_mib_fp32": database_count * 512 * 4 / 1024**2,
        "database_descriptor_mib_fp16": database_count * 512 * 2 / 1024**2,
        "sliding_window_state_bytes": state_bytes,
        "startup_encodings_per_observed_frame": 1,
        "steady_state_new_encodings_per_localization": 1,
    }


def evaluate_robotcar_temporal(
    config,
    checkpoint: str | Path,
    selection_path: str | Path,
    condition: str,
    output: str | Path,
    cache: str | Path,
    ablations_output: str | Path | None = None,
) -> Path:
    if condition not in {"night", "day"}:
        raise ValueError("Formal MGTR evaluation condition must be night or day")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    cache_path = build_robotcar_temporal_cache(config, checkpoint_path, condition, cache)
    metadata, arrays = load_robotcar_temporal_cache(cache_path)
    _validate_cache_for_run(metadata, config, checkpoint_path, condition)
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    selection_sha256 = file_sha256(selection_path)
    parameters = _validate_selection(selection, metadata)
    database, original_queries, split = _condition_rows(config, condition)
    order, _, _ = ordered_motion_deltas(original_queries)
    queries = [original_queries[int(index)] for index in order]

    full_rankings, diagnostics, timings = rerank_temporal(
        arrays["raw_indices"], arrays["raw_scores"], arrays["database_xy"],
        arrays["database_yaw"], arrays["motion_deltas"], arrays["sequence_starts"],
        parameters,
    )
    single_grid = evaluate_grid(arrays["raw_indices"], database, queries)
    full_grid = evaluate_grid(full_rankings, database, queries)
    benchmark_path = Path(config.vpr.output_dir) / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.is_file() else {}
    payload = {
        "schema": "robotcar_temporal_evaluation_v1",
        "condition": condition,
        "split": split,
        "development_result": True,
        "parameters": asdict(parameters),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "manifest_sha256": metadata["manifest_sha256"],
        "query_protocol_fingerprint": metadata["query_protocol_fingerprint"],
        "selection_sha256": selection_sha256,
        "database": len(database),
        "queries": len(queries),
        "single_frame": {"grid": single_grid},
        "mgtr": {"grid": full_grid, "decisions": diagnostics},
        "resources": temporal_resource_summary(
            parameters,
            len(database),
            checkpoint_path.stat().st_size / 1024**2,
            metadata,
            benchmark,
            timings,
        ),
    }
    write_json(payload, output)

    if ablations_output:
        fixed_deltas = arrays["motion_deltas"].copy()
        fixed_deltas[~arrays["sequence_starts"]] = 5.0
        fixed_rankings, _, _ = rerank_temporal(
            arrays["raw_indices"], arrays["raw_scores"], arrays["database_xy"],
            arrays["database_yaw"], fixed_deltas, arrays["sequence_starts"], parameters,
        )
        no_nms_rankings, _, _ = rerank_temporal(
            arrays["raw_indices"], arrays["raw_scores"], arrays["database_xy"],
            arrays["database_yaw"], arrays["motion_deltas"], arrays["sequence_starts"],
            parameters, use_nms=False,
        )
        write_json({
            "schema": "robotcar_temporal_ablations_v1",
            "condition": condition,
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "manifest_sha256": metadata["manifest_sha256"],
            "query_protocol_fingerprint": metadata["query_protocol_fingerprint"],
            "selection_sha256": selection_sha256,
            "parameters": asdict(parameters),
            "single_frame": evaluate_grid(arrays["raw_indices"], database, queries),
            "fixed_step_5m": evaluate_grid(fixed_rankings, database, queries),
            "no_nms": evaluate_grid(no_nms_rankings, database, queries),
            "full_mgtr": full_grid,
        }, ablations_output)
    return Path(output)


def _reference_robotcar(summary_path: str | Path) -> tuple[dict, dict]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    metrics = summary.get("datasets", {}).get("robotcar_project")
    if not metrics:
        raise ValueError(f"Reference summary has no robotcar_project result: {summary_path}")
    return summary, metrics


def summarize_robotcar_temporal(
    night_path: str | Path,
    day_path: str | Path,
    crica_path: str | Path,
    salad_path: str | Path,
    boq_path: str | Path,
    output: str | Path,
    reference_resources_path: str | Path | None = None,
) -> Path:
    night = json.loads(Path(night_path).read_text(encoding="utf-8"))
    day = json.loads(Path(day_path).read_text(encoding="utf-8"))
    if night.get("condition") != "night" or day.get("condition") != "day":
        raise ValueError("MGTR summary requires night and day evaluations")
    for field in ("checkpoint_sha256", "manifest_sha256", "selection_sha256"):
        if night.get(field) != day.get(field):
            raise ValueError(f"Night/day MGTR mismatch for {field}")
    primary = "distance_10m_yaw_30deg"
    single_night = night["single_frame"]["grid"][primary]["recalls"]["1"]
    mgtr_night = night["mgtr"]["grid"][primary]["recalls"]["1"]
    single_day = day["single_frame"]["grid"][primary]["recalls"]["1"]
    mgtr_day = day["mgtr"]["grid"][primary]["recalls"]["1"]
    reference_resources = {}
    if reference_resources_path is not None:
        benchmark = json.loads(Path(reference_resources_path).read_text(encoding="utf-8"))
        reference_resources = benchmark.get("models", {})
        if not isinstance(reference_resources, dict):
            raise ValueError("Reference resource benchmark must contain a models mapping")
    references = {}
    expected_protocol = night["query_protocol_fingerprint"]
    for label, path in (("cricavpr", crica_path), ("salad", salad_path), ("boq", boq_path)):
        summary, metrics = _reference_robotcar(path)
        fingerprint = metrics.get("query_protocol_fingerprint", metrics.get("protocol_fingerprint"))
        if fingerprint != expected_protocol:
            raise ValueError(f"Reference protocol mismatch for {label}")
        model_name = summary.get("model", label)
        resources = reference_resources.get(model_name, {})
        if reference_resources_path is not None and not resources:
            raise ValueError(f"Reference resource benchmark has no model: {model_name}")
        references[label] = {
            "model": model_name,
            "recall1": float(metrics["recalls"]["1"]),
            "parameters": resources.get(
                "parameters", summary.get("parameters", metrics.get("parameters"))
            ),
            "descriptor_dim": resources.get(
                "descriptor_dim", summary.get("descriptor_dim", metrics.get("descriptor_dim"))
            ),
            "checkpoint_bytes": resources.get("checkpoint_bytes"),
            "batch1_p50_ms": resources.get(
                "batch1_p50_ms",
                metrics.get("batch1_latency_p50_ms", metrics.get("batch1_p50_ms")),
            ),
            "batch1_p95_ms": resources.get("batch1_p95_ms"),
            "peak_cuda_mib": resources.get(
                "peak_cuda_mib",
                metrics.get("fp16_peak_cuda_mib", metrics.get("peak_cuda_mib")),
            ),
            "database_descriptor_memory_1m": resources.get(
                "database_descriptor_memory", {}
            ).get("1000000"),
        }
    rerank_p95 = float(night["resources"]["rerank_cpu_p95_ms"])
    acceptance = {
        "night_improves_single_frame": mgtr_night > single_night,
        "day_drop_within_0_5pp": mgtr_day >= single_day - 0.005,
        "rerank_cpu_p95_at_most_2ms": rerank_p95 <= 2.0,
    }
    acceptance["adopt_as_robotcar_method"] = all(acceptance.values())
    payload = {
        "schema": "robotcar_temporal_comparison_v1",
        "development_result": True,
        "primary_protocol": "10m+30deg",
        "reference_resource_benchmark_sha256": (
            file_sha256(reference_resources_path) if reference_resources_path is not None else None
        ),
        "mixvpr_512_cdnd": {
            "single_frame_recall1": single_night,
            "mgtr_recall1": mgtr_night,
            "delta": mgtr_night - single_night,
            "day_single_frame_recall1": single_day,
            "day_mgtr_recall1": mgtr_day,
            "resources": night["resources"],
        },
        "dinov2_single_frame_references": references,
        "claims": {
            "reaches_salad_single_frame": mgtr_night >= references["salad"]["recall1"],
            "exceeds_boq_single_frame": mgtr_night > references["boq"]["recall1"],
            "comparison_type": "lightweight_temporal_system_vs_single_frame_references",
            "not_a_same_reranker_comparison": True,
        },
        "acceptance": acceptance,
        "limitations": [
            "Query GPS is reduced to incremental motion; MGTR is motion-assisted, not pure single-frame VPR.",
            "DINOv2 references do not use MGTR, so this is a system-level comparison.",
            "RobotCar results are development results, not a sealed external test.",
        ],
    }
    write_json(payload, output)
    return Path(output)
