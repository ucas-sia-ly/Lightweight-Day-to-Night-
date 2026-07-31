from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .checkpointing import file_sha256, write_json
from .evaluation import extract_descriptors, positive_indices, retrieval_metrics
from .reference_models import (
    FixedResizeTransform, _registry, audit_reference_models, build_reference_model,
    protocol_fingerprint, suite_rows,
)
from .runtime import load_vpr_model
from .transforms import VPRTransform


def _parse_yaws(values: tuple[str, ...]) -> tuple[float | None, ...]:
    parsed = []
    for value in values:
        parsed.append(None if str(value).lower() == "none" else float(value))
    return tuple(parsed)


def _positives(database, queries, distance: float, yaw: float | None):
    if yaw is not None:
        return positive_indices(database, queries, distance, yaw)
    db_xy = np.asarray([(row.easting, row.northing) for row in database], dtype=np.float64)
    return [
        np.flatnonzero(
            np.linalg.norm(db_xy - np.asarray([row.easting, row.northing]), axis=1)
            <= distance
        )
        for row in queries
    ]


def _cell_key(distance: float, yaw: float | None) -> str:
    distance_text = str(int(distance)) if float(distance).is_integer() else str(distance)
    yaw_text = "none" if yaw is None else (
        str(int(yaw)) if float(yaw).is_integer() else str(yaw)
    )
    return f"distance_{distance_text}m_yaw_{yaw_text}deg"


def evaluate_robotcar_grid(
    config,
    checkpoint: str | Path,
    *,
    conditions: tuple[str, ...] = ("night", "day"),
    distances: tuple[float, ...] = (5.0, 10.0, 25.0),
    yaw_degrees: tuple[str, ...] = ("10", "30", "none"),
    output: str | Path | None = None,
) -> Path:
    if not conditions or any(value not in {"night", "day"} for value in conditions):
        raise ValueError("RobotCar grid conditions must be night and/or day")
    if not distances or any(value <= 0 for value in distances):
        raise ValueError("RobotCar grid distances must be positive")
    yaws = _parse_yaws(yaw_degrees)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    model, state, checkpoint_path = load_vpr_model(config, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    transform = VPRTransform(config.vpr.image_height, config.vpr.image_width)
    output_path = Path(output) if output else (
        Path(config.vpr.output_dir) / "robotcar_grid" / "summary.json"
    )
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": int(state.get("epoch", 0)),
        "input_size": [config.vpr.image_height, config.vpr.image_width],
        "descriptor_dim": config.vpr.descriptor_dim,
        "conditions": {},
    }
    batch_size = max(1, config.vpr.batch_size * 2)
    for condition in conditions:
        dataset = "robotcar" if condition == "night" else "robotcar_day"
        database, queries = suite_rows(config, dataset, "test")
        # Descriptor extraction is deliberately outside the protocol loops.
        database_descriptors = extract_descriptors(
            model, database, transform, device, batch_size, config.vpr.workers
        )
        query_descriptors = extract_descriptors(
            model, queries, transform, device, batch_size, config.vpr.workers
        )
        condition_payload = {
            "database": len(database),
            "queries": len(queries),
            "descriptor_extractions": {"database": 1, "queries": 1},
            "cells": {},
        }
        for distance in distances:
            for yaw in yaws:
                positives = _positives(database, queries, float(distance), yaw)
                metrics = retrieval_metrics(
                    database_descriptors, query_descriptors, positives,
                    config.evaluation.recalls,
                )
                metrics.update({
                    "positive_radius_m": float(distance),
                    "positive_yaw_deg": yaw,
                    "database": len(database),
                    "queries": len(queries),
                    "excluded_queries": len(queries) - metrics["valid_queries"],
                    "positive_count_by_query": [int(len(items)) for items in positives],
                    "protocol_fingerprint": protocol_fingerprint(
                        database, queries, float(distance), yaw
                    ),
                })
                condition_payload["cells"][_cell_key(float(distance), yaw)] = metrics
        payload["conditions"][condition] = condition_payload
    write_json(payload, output_path)
    return output_path


def evaluate_reference_robotcar_grid(
    config,
    registry_path: str | Path,
    *,
    model_name: str,
    database_encoder: str = "adapted",
    transform_name: str = "unified",
    conditions: tuple[str, ...] = ("night", "day"),
    distances: tuple[float, ...] = (5.0, 10.0, 25.0),
    yaw_degrees: tuple[str, ...] = ("10", "30", "none"),
    output: str | Path | None = None,
) -> Path:
    registry = _registry(registry_path)
    if model_name not in registry:
        raise ValueError(f"Unknown reference model: {model_name}")
    audit = audit_reference_models(registry_path)
    if not audit["models"][model_name]["ready"]:
        raise FileNotFoundError(json.dumps(audit["models"][model_name], indent=2))
    spec = registry[model_name]
    query_model = build_reference_model(spec)
    if database_encoder == "original":
        database_checkpoint = spec.get("database_checkpoint")
        if not database_checkpoint:
            raise ValueError(f"{model_name} has no original database checkpoint")
        database_model = build_reference_model({**spec, "checkpoint": database_checkpoint})
    elif database_encoder == "adapted":
        database_model = query_model
    else:
        raise ValueError("database_encoder must be original or adapted")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_model.eval().to(device)
    database_model.eval().to(device)
    if transform_name == "native":
        height, width = spec["native_size"]
        transform = FixedResizeTransform(height, width)
        batch_size = 1 if spec.get("batch_dependent") else int(spec.get("native_batch_size", 1))
    elif transform_name == "unified":
        height, width = config.vpr.image_height, config.vpr.image_width
        transform = VPRTransform(height, width)
        batch_size = 1 if spec.get("batch_dependent") else max(1, config.vpr.batch_size * 2)
    else:
        raise ValueError("transform must be native or unified")
    yaws = _parse_yaws(yaw_degrees)
    output_path = Path(output) if output else (
        Path("runs/reference") / model_name / "robotcar_grid" /
        transform_name / database_encoder / "summary.json"
    )
    payload = {
        "model": model_name,
        "database_encoder": database_encoder,
        "transform": transform_name,
        "input_size": [height, width],
        "descriptor_dim": int(spec["descriptor_dim"]),
        "conditions": {},
    }
    for condition in conditions:
        dataset = "robotcar" if condition == "night" else "robotcar_day"
        database, queries = suite_rows(config, dataset, "test")
        database_descriptors = extract_descriptors(
            database_model, database, transform, device, batch_size, config.vpr.workers
        )
        query_descriptors = extract_descriptors(
            query_model, queries, transform, device, batch_size, config.vpr.workers
        )
        cells = {}
        for distance in distances:
            for yaw in yaws:
                positives = _positives(database, queries, float(distance), yaw)
                metrics = retrieval_metrics(
                    database_descriptors, query_descriptors, positives,
                    config.evaluation.recalls,
                )
                metrics.update({
                    "positive_radius_m": float(distance), "positive_yaw_deg": yaw,
                    "database": len(database), "queries": len(queries),
                    "excluded_queries": len(queries) - metrics["valid_queries"],
                    "positive_count_by_query": [int(len(items)) for items in positives],
                    "protocol_fingerprint": protocol_fingerprint(database, queries, float(distance), yaw),
                })
                cells[_cell_key(float(distance), yaw)] = metrics
        payload["conditions"][condition] = {
            "database": len(database), "queries": len(queries),
            "descriptor_extractions": {"database": 1, "queries": 1}, "cells": cells,
        }
    write_json(payload, output_path)
    return output_path


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted p-values."""
    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def _paired_bootstrap_with_pvalue(candidate, baseline, samples: int, seed: int):
    first = np.asarray(candidate, dtype=np.float64)
    second = np.asarray(baseline, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or not len(first):
        raise ValueError("Grid paired-bootstrap vectors must be matching and non-empty")
    differences = first - second
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        count = min(256, samples - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        draws[start:start + count] = differences[indices].mean(axis=1)
    return {
        "delta": float(differences.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "one_sided_p": float((np.count_nonzero(draws <= 0.0) + 1) / (samples + 1)),
        "samples": int(samples),
    }


def compare_robotcar_grids(
    candidate: str | Path,
    baseline: str | Path,
    output: str | Path,
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> Path:
    first = json.loads(Path(candidate).read_text(encoding="utf-8"))
    second = json.loads(Path(baseline).read_text(encoding="utf-8"))
    comparisons = {}
    for condition in sorted(set(first["conditions"]) & set(second["conditions"])):
        first_cells = first["conditions"][condition]["cells"]
        second_cells = second["conditions"][condition]["cells"]
        condition_results = {}
        raw_p = {}
        for index, name in enumerate(sorted(set(first_cells) & set(second_cells))):
            left, right = first_cells[name], second_cells[name]
            if left["protocol_fingerprint"] != right["protocol_fingerprint"]:
                raise ValueError(f"Protocol fingerprint mismatch for {condition}/{name}")
            valid = np.asarray(left["valid_by_query"], dtype=bool)
            if left["valid_by_query"] != right["valid_by_query"]:
                raise ValueError(f"Valid-query mask mismatch for {condition}/{name}")
            result = _paired_bootstrap_with_pvalue(
                np.asarray(left["success_by_k"]["1"])[valid],
                np.asarray(right["success_by_k"]["1"])[valid],
                samples, seed + index,
            )
            condition_results[name] = result
            raw_p[name] = result["one_sided_p"]
        adjusted = holm_adjust(raw_p)
        for name, value in adjusted.items():
            condition_results[name]["holm_adjusted_p"] = value
            condition_results[name]["holm_significant_0_05"] = value < 0.05
        comparisons[condition] = condition_results
    output_path = Path(output)
    write_json({
        "candidate": str(Path(candidate).resolve()),
        "baseline": str(Path(baseline).resolve()),
        "comparisons": comparisons,
        "correction": "Holm within each condition over the configured grid cells",
    }, output_path)
    return output_path
