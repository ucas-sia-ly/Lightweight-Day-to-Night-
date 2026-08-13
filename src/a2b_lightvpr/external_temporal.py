from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .checkpointing import file_sha256, write_json
from .evaluation import extract_descriptors
from .reference_models import (
    FixedResizeTransform,
    _reference_checkpoint,
    _registry,
    audit_reference_models,
    build_reference_model,
)
from .robotcar_temporal import TemporalParameters, _top_scores, rerank_temporal
from .runtime import load_vpr_model
from .transforms import VPRTransform


GARDENS_WINDOWS = (3, 4, 5, 6)
GARDENS_PRIMARY_WINDOW = 4


def _runtime_identity():
    return {"python": platform.python_version(), "torch": torch.__version__}


@dataclass(frozen=True)
class SequenceRow:
    source_path: str
    sequence_index: int


def _descriptor_cache(
    path, expected, model, rows, transform, device, workers, batch_size=16
):
    path = Path(path); metadata_path = path.with_suffix(".json")
    if path.is_file() and metadata_path.is_file():
        actual = json.loads(metadata_path.read_text())
        if (actual.get("inputs") == expected
                and actual.get("cache_sha256") == file_sha256(path)):
            descriptors = np.load(path)
            if descriptors.shape == (len(rows), model.descriptor_dim):
                return descriptors
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptors = extract_descriptors(
        model, rows, transform, device, int(batch_size), workers
    )
    temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, descriptors.astype(np.float32))
    os.replace(temporary, path)
    write_json({"inputs": expected, "cache_sha256": file_sha256(path)}, metadata_path)
    return descriptors


def gardens_rows(root: str | Path):
    root = Path(root).expanduser().resolve()
    groups = {}
    for suffix in ("day_left", "day_right", "night_right"):
        paths = sorted(root.glob(f"*_{suffix}.jpg"), key=lambda p: int(p.stem.split("_")[0]))
        if len(paths) != 200:
            raise ValueError(f"Gardens Point requires 200 {suffix} images, got {len(paths)}")
        groups[suffix] = [SequenceRow(str(path), index) for index, path in enumerate(paths)]
    return groups


def _sequence_fingerprint(groups):
    digest = hashlib.sha256()
    for name in sorted(groups):
        for row in groups[name]:
            digest.update(
                f"{name}\0{row.sequence_index}\0{Path(row.source_path).name}\n".encode()
            )
    return digest.hexdigest()


def _index_grid(rankings):
    output = {}
    truth = np.arange(len(rankings))
    for tolerance in (0, 2, 5, 10):
        recalls, success = {}, {}
        for k in (1, 5, 10):
            hit = (np.abs(rankings[:, :k] - truth[:, None]) <= tolerance).any(axis=1)
            recalls[str(k)] = float(hit.mean()); success[str(k)] = hit.astype(np.uint8).tolist()
        output[f"frame_tolerance_{tolerance}"] = {
            "frame_tolerance": tolerance, "recalls": recalls, "success_by_k": success,
            "total_queries": len(rankings),
        }
    return output


def _garden_trial(indices, scores, parameters):
    count = len(indices)
    rankings, diagnostics, timings = rerank_temporal(
        indices, scores, np.stack([np.arange(200), np.zeros(200)], axis=1),
        np.zeros(200), np.r_[0, np.ones(count - 1)], np.r_[True, np.zeros(count - 1, dtype=bool)],
        parameters, use_nms=False,
    )
    return rankings, diagnostics, timings, _index_grid(rankings)


def _evaluate_gardens_rankings(
    val_indices, val_scores, test_indices, test_scores, identity, output_dir
):
    trials = []
    for window in GARDENS_WINDOWS:
        parameters = TemporalParameters(
            window_size=window, top_k=10, transition_radius_m=5,
            motion_weight=.5, motion_sigma_m=1, nms_radius_m=0,
        )
        _, _, timings, grid = _garden_trial(val_indices, val_scores, parameters)
        trials.append({"parameters": asdict(parameters),
                       "recalls": grid["frame_tolerance_0"]["recalls"],
                       "rerank_p95_ms": float(np.percentile(timings, 95))})
    selected = next(
        TemporalParameters(**trial["parameters"]) for trial in trials
        if trial["parameters"]["window_size"] == GARDENS_PRIMARY_WINDOW
    )
    selection_path = Path(output_dir) / "day_right_selection.json"
    write_json({"schema": "gardens_temporal_selection_v1", "selection_split": "day_right",
                "temporal_method": "mgtr_v1", **identity,
                "primary_window_size": GARDENS_PRIMARY_WINDOW,
                "selected_parameters": asdict(selected), "trials": trials}, selection_path)
    _, _, val_time, val_grid = _garden_trial(val_indices, val_scores, selected)
    _, decisions, test_time, test_grid = _garden_trial(test_indices, test_scores, selected)
    window_sweep = {}
    for window in GARDENS_WINDOWS:
        parameters = TemporalParameters(**{**asdict(selected), "window_size": window})
        val_window = _garden_trial(val_indices, val_scores, parameters)
        test_window = _garden_trial(test_indices, test_scores, parameters)
        window_sweep[str(window)] = {
            "window_size": window, "day_right": val_window[3],
            "night_right": test_window[3],
            "day_right_rerank_p95_ms": float(np.percentile(val_window[2], 95)),
            "night_right_rerank_p95_ms": float(np.percentile(test_window[2], 95)),
        }
    return {
        "selection_sha256": file_sha256(selection_path), "parameters": asdict(selected),
        "temporal_method": "mgtr_v1", "primary_window_size": GARDENS_PRIMARY_WINDOW,
        "window_sweep": window_sweep,
        "day_right": {"single_frame": _index_grid(val_indices), "temporal": val_grid,
                      "rerank_p95_ms": float(np.percentile(val_time, 95))},
        "night_right": {"single_frame": _index_grid(test_indices), "temporal": test_grid,
                        "decisions": decisions,
                        "rerank_p95_ms": float(np.percentile(test_time, 95))},
    }


def evaluate_gardens_temporal(config, checkpoint, root, output_dir, method="mgtr-v1"):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    groups = gardens_rows(root); fingerprint = _sequence_fingerprint(groups)
    model, _, checkpoint_path = load_vpr_model(config, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval(); transform = VPRTransform(320, 320)
    descriptor_dim = int(config.vpr.descriptor_dim)
    descriptors = {
        name: _descriptor_cache(
            output_dir / "cache" / f"{name}.npy",
            {"schema": "descriptor_cache_v1", "dataset": "gardens_point", "sequence": name,
             "manifest_sha256": fingerprint, "checkpoint_sha256": file_sha256(checkpoint_path),
             "descriptor_dim": descriptor_dim, "input_size": [320, 320], "rows": len(rows)},
            model, rows, transform, device, config.vpr.workers,
        ) for name, rows in groups.items()
    }
    val_indices, val_scores = _top_scores(descriptors["day_left"], descriptors["day_right"], 100)
    test_indices, test_scores = _top_scores(descriptors["day_left"], descriptors["night_right"], 100)
    identity = {"manifest_sha256": fingerprint,
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "model": f"mixvpr-{descriptor_dim}-cdnd", "runtime": _runtime_identity()}
    if method == "ragtr":
        if descriptor_dim != 4096:
            raise ValueError("Gardens RA-GTR requires the 4096-dimensional CDND checkpoint")
        from .ragtr import evaluate_gardens_ragtr_rankings
        result = evaluate_gardens_ragtr_rankings(
            groups, val_indices, val_scores, test_indices, test_scores, identity, output_dir
        )
        schema = "gardens_ragtr_evaluation_v1"
    elif method == "mgtr-v1":
        result = _evaluate_gardens_rankings(
            val_indices, val_scores, test_indices, test_scores, identity, output_dir
        )
        schema = "gardens_temporal_evaluation_v1"
    else:
        raise ValueError(f"Unknown Gardens temporal method: {method}")
    payload = {"schema": schema, "development_result": False,
               "metric_unit": "sequence_frame_index", "not_metric_distance": True,
               "database": "day_left", "selection_query": "day_right",
               "external_query": "night_right", "descriptor_dim": descriptor_dim,
               **identity, **result}
    output = output_dir / "evaluation.json"; write_json(payload, output); return output


def evaluate_gardens_reference(
    config, registry_path, model_name, root, output_dir, database_encoder="adapted"
):
    registry = _registry(registry_path)
    if model_name not in registry:
        raise ValueError(f"Unknown reference model: {model_name}")
    audit = audit_reference_models(registry_path)
    if not audit["models"][model_name]["ready"]:
        raise FileNotFoundError(json.dumps(audit["models"][model_name], indent=2))
    spec = dict(registry[model_name])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # Must be set before a DINOv2-backed reference imports its attention module.
        os.environ.setdefault("XFORMERS_DISABLED", "1")
    query_model = build_reference_model(spec)
    query_checkpoint = _reference_checkpoint(spec)
    if database_encoder == "original":
        if not spec.get("database_checkpoint"):
            raise ValueError(f"{model_name} has no original database checkpoint")
        database_checkpoint = Path(spec["database_checkpoint"]).expanduser().resolve()
        database_model = build_reference_model({**spec, "checkpoint": str(database_checkpoint)})
    else:
        database_checkpoint = query_checkpoint
        database_model = query_model
    groups = gardens_rows(root); fingerprint = _sequence_fingerprint(groups)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    query_model.to(device).eval(); database_model.to(device).eval()
    height, width = map(int, spec["native_size"])
    transform = FixedResizeTransform(height, width)
    batch_size = 1 if spec.get("batch_dependent") else int(spec.get("native_batch_size", 16))
    common = {"schema": "descriptor_cache_v1", "dataset": "gardens_point",
              "model": model_name, "manifest_sha256": fingerprint,
              "descriptor_dim": int(spec["descriptor_dim"]), "input_size": [height, width],
              "database_encoder": database_encoder, "runtime": _runtime_identity()}
    descriptors = {
        "day_left": _descriptor_cache(
            output_dir / "cache" / "day_left.npy",
            {**common, "sequence": "day_left", "rows": 200,
             "checkpoint_sha256": file_sha256(database_checkpoint)},
            database_model, groups["day_left"], transform, device, config.vpr.workers,
            batch_size,
        )
    }
    for name in ("day_right", "night_right"):
        descriptors[name] = _descriptor_cache(
            output_dir / "cache" / f"{name}.npy",
            {**common, "sequence": name, "rows": 200,
             "checkpoint_sha256": file_sha256(query_checkpoint)},
            query_model, groups[name], transform, device, config.vpr.workers, batch_size,
        )
    val_indices, val_scores = _top_scores(descriptors["day_left"], descriptors["day_right"], 100)
    test_indices, test_scores = _top_scores(descriptors["day_left"], descriptors["night_right"], 100)
    identity = {"model": model_name, "manifest_sha256": fingerprint,
                "query_checkpoint_sha256": file_sha256(query_checkpoint),
                "database_checkpoint_sha256": file_sha256(database_checkpoint),
                "database_encoder": database_encoder, "runtime": _runtime_identity()}
    result = _evaluate_gardens_rankings(
        val_indices, val_scores, test_indices, test_scores, identity, output_dir
    )
    payload = {"schema": "gardens_reference_evaluation_v1", "development_result": False,
               "metric_unit": "sequence_frame_index", "not_metric_distance": True,
               "database": "day_left", "selection_query": "day_right",
               "external_query": "night_right", "descriptor_dim": int(spec["descriptor_dim"]),
               "input_size": [height, width], "extraction_batch_size": batch_size,
               **identity, **result}
    output = output_dir / "evaluation.json"; write_json(payload, output); return output


def summarize_gardens_models(results_root, output, primary_result=None):
    rows = []
    paths = list(sorted(Path(results_root).glob("*/evaluation.json")))
    if primary_result:
        paths.insert(0, Path(primary_result))
    for path in paths:
        item = json.loads(path.read_text())
        if item.get("schema") not in {
            "gardens_reference_evaluation_v1", "gardens_temporal_evaluation_v1",
            "gardens_ragtr_evaluation_v1",
        }:
            continue
        method = item.get("temporal_method")
        if method not in {"mgtr_v1", "ragtr"}:
            raise ValueError(f"Unknown Gardens temporal method in {path}: {method}")
        exact_single = item["night_right"]["single_frame"]["frame_tolerance_0"]["recalls"]
        exact_temporal = item["night_right"]["temporal"]["frame_tolerance_0"]["recalls"]
        rows.append({"model": item.get("model", "mixvpr-512-cdnd"),
                     "descriptor_dim": item.get("descriptor_dim", 512),
                     "database_encoder": item.get("database_encoder", "adapted"),
                     "temporal_method": method,
                     "primary_window_size": item.get("primary_window_size", 4),
                     "single_frame": exact_single, "temporal": exact_temporal,
                     "delta_r1": exact_temporal["1"] - exact_single["1"],
                     "window_sweep": {
                         window: values["night_right"]["frame_tolerance_0"]["recalls"]
                         for window, values in item["window_sweep"].items()
                     }, "result": str(path.resolve())})
    if not rows:
        raise ValueError("No Gardens reference evaluation results found")
    methods = sorted({row["temporal_method"] for row in rows})
    payload = {"schema": "gardens_models_summary_v1",
               "temporal_method": methods[0] if len(methods) == 1 else "mixed",
               "primary_window_size": GARDENS_PRIMARY_WINDOW,
               "metric_unit": "sequence_frame_index", "tolerance": "exact",
               "models": rows}
    write_json(payload, output); return Path(output)
