from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .checkpointing import file_sha256, write_json
from .evaluation import extract_descriptors
from .references.models import query_protocol_fingerprint, suite_rows
from .runtime import load_vpr_model
from .transforms import VPRTransform


RAW_TOP_K = 100
DISTANCES_M = (5.0, 10.0, 25.0, 50.0)
YAWS_DEG = (10.0, 30.0, None)
RECALLS = (1, 5, 10)
TEMPORAL_DESCRIPTOR_DIMS = (512,)
CLUSTERED_WINDOWS = (1, 3, 5, 7, 9)
CLUSTER_GAP_FRAMES = 2
CLUSTER_RADIUS_FRAMES = 5


@dataclass(frozen=True)
class ClusteredTemporalParameters:
    window_size: int
    top_k: int = 10
    cluster_gap_frames: int = CLUSTER_GAP_FRAMES
    center_radius_frames: int = CLUSTER_RADIUS_FRAMES


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
        "descriptor_dim": int(config.vpr.descriptor_dim),
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest_sha256": file_sha256(config.data.robotcar_manifest),
        "query_protocol_fingerprint": query_fingerprint,
        "database_rows": len(database),
        "query_rows": len(queries),
        "raw_top_k": min(RAW_TOP_K, len(database)),
        "input_size": [config.vpr.image_height, config.vpr.image_width],
    }


def build_robotcar_temporal_cache(config, checkpoint: str | Path, condition: str, cache: str | Path) -> Path:
    descriptor_dim = int(config.vpr.descriptor_dim)
    if descriptor_dim not in TEMPORAL_DESCRIPTOR_DIMS:
        raise ValueError("TCR requires a 512-dimensional MixVPR checkpoint")
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
    if (
        int(getattr(model, "descriptor_dim", 0)) != descriptor_dim
        or int(state.get("descriptor_dim", descriptor_dim)) != descriptor_dim
    ):
        raise ValueError("TCR checkpoint descriptor dimension does not match the config")
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


def _database_frame_positions(database):
    positions = np.full(len(database), -1, dtype=np.int64)
    routes = np.empty(len(database), dtype=object)
    grouped = {}
    for index, row in enumerate(database):
        route = str(getattr(row, "traverse", "single"))
        routes[index] = route
        grouped.setdefault(route, []).append(index)
    for route, indices in grouped.items():
        ordered = sorted(indices, key=lambda i: (int(getattr(database[i], "timestamp", i)), i))
        for position, index in enumerate(ordered):
            positions[index] = position
    return positions, routes


def _chain_clusters(values, scores, routes, gap):
    order = sorted(range(len(values)), key=lambda i: (routes[i], int(values[i]), i))
    clusters = []
    for item in order:
        if not clusters or routes[item] != routes[clusters[-1][-1]] or values[item] - values[clusters[-1][-1]] > gap:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    return clusters


def _append_raw_candidates(ranked: np.ndarray, raw: np.ndarray) -> np.ndarray:
    output = [int(value) for value in ranked]
    seen = set(output)
    for value in raw:
        value = int(value)
        if value not in seen:
            output.append(value)
            seen.add(value)
    return np.asarray(output[: len(raw)], dtype=np.int32)


def rerank_clustered_temporal(
    raw_indices, raw_scores, database_positions, database_routes,
    sequence_starts, parameters: ClusteredTemporalParameters,
):
    count, raw_count = raw_indices.shape
    if parameters.window_size not in CLUSTERED_WINDOWS or parameters.window_size % 2 == 0:
        raise ValueError("TCR window must be one of 1,3,5,7,9")
    output = raw_indices.copy().astype(np.int32)
    diagnostics, timings = [], np.zeros(count, dtype=np.float64)
    sequence_start = 0
    sequence_starts_array = np.asarray(sequence_starts, dtype=bool)
    start_indices = np.flatnonzero(sequence_starts_array).tolist()
    sequence_end_by_query = np.empty(count, dtype=np.int64)
    for seq_start, seq_end in zip(start_indices, start_indices[1:] + [count]):
        sequence_end_by_query[seq_start:seq_end] = seq_end
    half = parameters.window_size // 2
    for target in range(count):
        started = time.perf_counter()
        if bool(sequence_starts[target]):
            sequence_start = target
        start = max(sequence_start, target - half)
        end = min(int(sequence_end_by_query[target]), target + half + 1)
        if end - start == 1:
            reason = "single_frame"
        else:
            top1 = raw_indices[start:end, 0]
            values = database_positions[top1]
            routes = database_routes[top1]
            clusters = _chain_clusters(values, raw_scores[start:end, 0], routes, parameters.cluster_gap_frames)
            clusters.sort(key=lambda c: (-len(c), -float(np.mean(raw_scores[start:end, 0][c])), int(np.median(values[c]))))
            chosen = clusters[0]
            center = int(np.median(values[chosen]))
            allowed = set()
            for rank in range(min(parameters.top_k, raw_count)):
                candidate = int(raw_indices[target, rank])
                if candidate >= 0 and database_routes[candidate] == routes[chosen[0]] and abs(int(database_positions[candidate]) - center) <= parameters.center_radius_frames:
                    allowed.add(candidate)
            if not allowed:
                for rank in range(raw_count):
                    candidate = int(raw_indices[target, rank])
                    if candidate >= 0 and database_routes[candidate] == routes[chosen[0]] and abs(int(database_positions[candidate]) - center) <= parameters.center_radius_frames:
                        allowed.add(candidate)
            if allowed:
                ranked = sorted(allowed, key=lambda c: (-float(raw_scores[target, np.flatnonzero(raw_indices[target] == c)[0]]), c))
                output[target] = _append_raw_candidates(np.asarray(ranked, dtype=np.int32), raw_indices[target])
                reason = "cluster_center" if len(allowed) else "single_frame"
            else:
                reason = "cluster_no_candidate_raw_top1"
        timings[target] = (time.perf_counter() - started) * 1000.0
        diagnostics.append({"query_index": target, "output_top1": int(output[target, 0]), "window_used": end - start, "reason": reason, "center_delay_frames": half})
    return output, diagnostics, timings


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


def ranking_metrics(
    rankings: np.ndarray, database, queries, distance: float, yaw: float | None,
    query_mask: np.ndarray | None = None,
):
    positives = _positive_indices(database, queries, distance, yaw)
    valid = np.asarray([len(items) > 0 for items in positives], dtype=bool)
    if query_mask is not None:
        query_mask = np.asarray(query_mask, dtype=bool)
        if query_mask.shape != valid.shape:
            raise ValueError("Query mask shape does not match the evaluation queries")
        valid &= query_mask
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


def map_covered_center_mask(database, queries, sequence_starts, window_size: int = 9):
    """Select centers whose full window has 10m+30deg database coverage."""
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("Map-covered protocol requires a positive odd window")
    covered = np.asarray(
        [len(items) > 0 for items in _positive_indices(database, queries, 10.0, 30.0)],
        dtype=bool,
    )
    starts = np.flatnonzero(np.asarray(sequence_starts, dtype=bool)).tolist()
    if not starts or starts[0] != 0:
        raise ValueError("The first query must start a sequence")
    eligible = np.zeros(len(queries), dtype=bool)
    half = window_size // 2
    for start, end in zip(starts, starts[1:] + [len(queries)]):
        for center in range(start + half, end - half):
            eligible[center] = bool(covered[center - half:center + half + 1].all())
    return eligible, covered


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


def evaluate_grid_masked(rankings, database, queries, query_mask) -> dict:
    output = {}
    for distance, yaw in product(DISTANCES_M, YAWS_DEG):
        yaw_text = "none" if yaw is None else str(int(yaw))
        output[f"distance_{int(distance)}m_yaw_{yaw_text}deg"] = {
            **ranking_metrics(rankings, database, queries, distance, yaw, query_mask),
            "positive_radius_m": distance,
            "positive_yaw_deg": yaw,
        }
    return output


def _validate_cache_for_run(metadata: dict, config, checkpoint: Path, condition: str) -> None:
    expected = {
        "condition": condition,
        "descriptor_dim": int(config.vpr.descriptor_dim),
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest_sha256": file_sha256(config.data.robotcar_manifest),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items() if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"RobotCar temporal cache mismatch: {mismatches}")


def select_robotcar_temporal(
    config, checkpoint: str | Path, output: str | Path, cache: str | Path,
    method: str = "clustered-mgtr",
) -> Path:
    if method != "clustered-mgtr":
        raise ValueError("Only the clustered-mgtr temporal method is supported")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    cache_path = build_robotcar_temporal_cache(config, checkpoint_path, "val", cache)
    metadata, arrays = load_robotcar_temporal_cache(cache_path)
    _validate_cache_for_run(metadata, config, checkpoint_path, "val")
    database, original_queries, _ = _condition_rows(config, "val")
    order, _, _ = ordered_motion_deltas(original_queries)
    queries = [original_queries[int(index)] for index in order]
    positions, routes = _database_frame_positions(database)
    covered_mask, raw_coverage = map_covered_center_mask(
        database, queries, arrays["sequence_starts"], max(CLUSTERED_WINDOWS)
    )
    trials = []
    for window in CLUSTERED_WINDOWS:
        parameters = ClusteredTemporalParameters(window)
        rankings, diagnostics, timings = rerank_clustered_temporal(
            arrays["raw_indices"], arrays["raw_scores"], positions, routes,
            arrays["sequence_starts"], parameters,
        )
        metrics = ranking_metrics(rankings, database, queries, 10.0, 30.0, covered_mask)
        trials.append({"parameters": asdict(parameters), "metrics": metrics,
                       "recall1": metrics["recalls"]["1"],
                       "recall5": metrics["recalls"]["5"], "recall10": metrics["recalls"]["10"],
                       "fallback_queries": sum(x["reason"] == "cluster_no_candidate_raw_top1" for x in diagnostics),
                       "rerank_p95_ms": float(np.percentile(timings, 95))})
    trials.sort(key=lambda x: (-x["recall1"], -x["recall5"], -x["recall10"], x["parameters"]["window_size"]))
    single = ranking_metrics(arrays["raw_indices"], database, queries, 10.0, 30.0)
    write_json({"schema": "robotcar_clustered_mgtr_selection_v1", "selection_split": "val",
                "selection_metric": "10m+30deg_R1_R5_R10", "selected_parameters": trials[0]["parameters"],
                "selected_metrics": trials[0]["metrics"], "single_frame_metrics": single,
                "checkpoint_sha256": metadata["checkpoint_sha256"], "manifest_sha256": metadata["manifest_sha256"],
                "val_query_protocol_fingerprint": metadata["query_protocol_fingerprint"],
                "top100_used_for_fallback": True,
                "coverage_protocol": {
                    "positive_radius_m": 10.0, "positive_yaw_deg": 30.0,
                    "common_window_size": max(CLUSTERED_WINDOWS),
                    "covered_queries": int(raw_coverage.sum()),
                    "eligible_centers": int(covered_mask.sum()),
                    "total_queries": len(queries),
                },
                "trials": trials}, output)
    return Path(output)


def temporal_resource_summary(
    parameters: ClusteredTemporalParameters,
    database_count: int,
    checkpoint_mib: float,
    metadata: dict,
    benchmark: dict,
    timings: np.ndarray,
) -> dict:
    descriptor_dim = int(metadata.get("descriptor_dim", 512))
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
        "descriptor_dim": descriptor_dim,
        "database_descriptor_mib_fp32": database_count * descriptor_dim * 4 / 1024**2,
        "database_descriptor_mib_fp16": database_count * descriptor_dim * 2 / 1024**2,
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
    method: str = "clustered-mgtr",
) -> Path:
    if condition not in {"night", "day"}:
        raise ValueError("Formal TCR evaluation condition must be night or day")
    if method != "clustered-mgtr":
        raise ValueError("Only the clustered-mgtr temporal method is supported")
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    cache_path = build_robotcar_temporal_cache(config, checkpoint_path, condition, cache)
    metadata, arrays = load_robotcar_temporal_cache(cache_path)
    _validate_cache_for_run(metadata, config, checkpoint_path, condition)
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    selection_sha256 = file_sha256(selection_path)
    if selection.get("schema") != "robotcar_clustered_mgtr_selection_v1" or selection.get("selection_split") != "val":
        raise ValueError("TCR parameters must be selected on RobotCar val")
    if selection.get("checkpoint_sha256") != metadata.get("checkpoint_sha256") or selection.get("manifest_sha256") != metadata.get("manifest_sha256"):
        raise ValueError("TCR selection/cache mismatch")
    parameters = ClusteredTemporalParameters(**selection["selected_parameters"])
    database, original_queries, split = _condition_rows(config, condition)
    order, _, _ = ordered_motion_deltas(original_queries)
    queries = [original_queries[int(index)] for index in order]

    selected = ClusteredTemporalParameters(**selection["selected_parameters"])
    positions, routes = _database_frame_positions(database)
    full_rankings, diagnostics, timings = rerank_clustered_temporal(
        arrays["raw_indices"], arrays["raw_scores"], positions, routes,
        arrays["sequence_starts"], selected,
    )
    single_grid = evaluate_grid(arrays["raw_indices"], database, queries)
    full_grid = evaluate_grid(full_rankings, database, queries)
    covered_grid = None
    coverage_protocol = None
    covered_mask, raw_coverage = map_covered_center_mask(
        database, queries, arrays["sequence_starts"], max(CLUSTERED_WINDOWS)
    )
    covered_grid = {
        "single_frame": evaluate_grid_masked(
            arrays["raw_indices"], database, queries, covered_mask
        ),
        "tcr": evaluate_grid_masked(full_rankings, database, queries, covered_mask),
    }
    coverage_protocol = {
        "positive_radius_m": 10.0, "positive_yaw_deg": 30.0,
        "common_window_size": max(CLUSTERED_WINDOWS),
        "covered_queries": int(raw_coverage.sum()),
        "eligible_centers": int(covered_mask.sum()),
        "total_queries": len(queries),
    }
    benchmark_path = Path(config.vpr.output_dir) / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.is_file() else {}
    payload = {
        "schema": "robotcar_clustered_mgtr_evaluation_v1",
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
        "descriptor_dim": int(config.vpr.descriptor_dim),
        "single_frame": {"grid": single_grid},
        "tcr": {"grid": full_grid, "decisions": diagnostics, "method": method},
        "map_covered_protocol": coverage_protocol,
        "map_covered_grid": covered_grid,
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

    return Path(output)


def _reference_robotcar(summary_path: str | Path) -> tuple[dict, dict]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    metrics = summary.get("datasets", {}).get("robotcar_project")
    if not metrics:
        raise ValueError(f"Reference summary has no robotcar_project result: {summary_path}")
    return summary, metrics


def _tcr_grid(payload: dict) -> dict:
    """Read the current TCR field, with a one-way compatibility read for old files."""
    result = payload.get("tcr") or payload.get("mgtr")
    if not isinstance(result, dict) or "grid" not in result:
        raise ValueError("Temporal evaluation does not contain a TCR ranking grid")
    return result["grid"]


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
        raise ValueError("TCR summary requires night and day evaluations")
    for field in ("checkpoint_sha256", "manifest_sha256", "selection_sha256"):
        if night.get(field) != day.get(field):
            raise ValueError(f"Night/day TCR mismatch for {field}")
    primary = "distance_10m_yaw_30deg"
    single_night = night["single_frame"]["grid"][primary]["recalls"]["1"]
    tcr_night = _tcr_grid(night)[primary]["recalls"]["1"]
    single_day = day["single_frame"]["grid"][primary]["recalls"]["1"]
    tcr_day = _tcr_grid(day)[primary]["recalls"]["1"]
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
        "night_improves_single_frame": tcr_night > single_night,
        "day_drop_within_0_5pp": tcr_day >= single_day - 0.005,
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
        f"mixvpr_{int(night['resources'].get('descriptor_dim', night.get('descriptor_dim', 512)))}_cdnd": {
            "single_frame_recall1": single_night,
            "tcr_recall1": tcr_night,
            "delta": tcr_night - single_night,
            "day_single_frame_recall1": single_day,
            "day_tcr_recall1": tcr_day,
            "resources": night["resources"],
        },
        "dinov2_single_frame_references": references,
        "claims": {
            "reaches_salad_single_frame": tcr_night >= references["salad"]["recall1"],
            "exceeds_boq_single_frame": tcr_night > references["boq"]["recall1"],
            "comparison_type": "lightweight_temporal_system_vs_single_frame_references",
            "not_a_same_reranker_comparison": True,
        },
        "acceptance": acceptance,
        "limitations": [
            "TCR uses ordered queries and finite future context; it is not pure single-frame VPR.",
            "DINOv2 references do not use TCR, so this is a system-level comparison.",
            "RobotCar results are development results, not a sealed external test.",
        ],
    }
    write_json(payload, output)
    return Path(output)
