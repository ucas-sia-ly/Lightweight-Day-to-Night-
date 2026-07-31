from __future__ import annotations

import json
from pathlib import Path

import torch

from .checkpointing import file_sha256, write_json
from .config import ExperimentConfig
from .reference_models import (
    FixedResizeTransform,
    _registry,
    audit_reference_models,
    build_reference_model,
    evaluation_fingerprint,
    evaluate_dual_encoder,
    query_protocol_fingerprint,
    suite_rows,
)
from .tracking import TrainingTracker
from .trained_suite import SUITE_PROTOCOLS, expand_suite_datasets
from .transforms import VPRTransform
from .evaluation import paired_bootstrap


def _tracking_values(payload: dict) -> dict:
    values = {
        "reference_suite/model": payload["model"],
        "reference_suite/database_encoder": payload["database_encoder"],
        "reference_suite/transform": payload["transform"],
        "reference_suite/query_checkpoint_sha256": payload["query_checkpoint_sha256"],
    }
    for name, metrics in payload["datasets"].items():
        namespace = SUITE_PROTOCOLS[name][4]
        prefix = f"{namespace}/{name}"
        for recall, value in metrics["recalls"].items():
            values[f"{prefix}/recall_at_{recall}"] = value
        values[f"{prefix}/mrr"] = metrics["mrr"]
        values[f"{prefix}/valid_queries"] = metrics["valid_queries"]
        query_fingerprint = metrics.get(
            "query_protocol_fingerprint", metrics.get("protocol_fingerprint")
        )
        if query_fingerprint:
            values[f"{prefix}/query_protocol_fingerprint"] = query_fingerprint
        if metrics.get("evaluation_fingerprint"):
            values[f"{prefix}/evaluation_fingerprint"] = metrics[
                "evaluation_fingerprint"
            ]
    return values


def evaluate_reference_suite(
    config: ExperimentConfig,
    registry_path: str | Path,
    *,
    model_name: str,
    database_encoder: str = "original",
    transform_name: str = "unified",
    datasets: tuple[str, ...] = ("all",),
    output_dir: str | Path | None = None,
    batch_size_override: int | None = None,
    allow_batch_dependent: bool = False,
) -> Path:
    selected = expand_suite_datasets(datasets)
    unknown = sorted(set(selected) - set(SUITE_PROTOCOLS))
    if unknown:
        raise ValueError(f"Unknown reference-suite datasets: {unknown}")
    registry = _registry(registry_path)
    if model_name not in registry:
        raise ValueError(f"Unknown reference model: {model_name}")
    audit = audit_reference_models(registry_path)
    if not audit["models"][model_name]["ready"]:
        raise FileNotFoundError(json.dumps(audit["models"][model_name], indent=2))
    spec = registry[model_name]
    if database_encoder == "original" and not spec.get("database_checkpoint"):
        raise ValueError(f"{model_name} has no original database checkpoint")
    if spec.get("checkpoint"):
        query_checkpoint = Path(spec["checkpoint"])
    elif spec.get("loader") == "eigenplaces_hub":
        from .vpr.model import eigenplaces_checkpoint
        query_checkpoint = eigenplaces_checkpoint(int(spec["descriptor_dim"]))
        spec["_resolved_checkpoint"] = str(query_checkpoint)
    else:
        raise ValueError(f"{model_name} does not declare or resolve a checkpoint")
    if not query_checkpoint.is_file():
        raise FileNotFoundError(f"Resolved query checkpoint is missing: {query_checkpoint}")
    database_checkpoint = Path(
        spec["database_checkpoint"]
        if database_encoder == "original" else query_checkpoint
    )
    root = Path(output_dir or (
        Path("runs/reference") / model_name / "final_suite" / transform_name / database_encoder
    ))
    summary_path = root / "summary.json"
    configured_batch_size = (
        int(spec.get("native_batch_size", 1))
        if transform_name == "native" else max(1, config.vpr.batch_size * 2)
    )
    if spec.get("batch_dependent"):
        configured_batch_size = 1
    if batch_size_override is not None:
        if batch_size_override < 1:
            raise ValueError("Reference suite batch size must be positive")
        if spec.get("batch_dependent") and batch_size_override > 1 and not allow_batch_dependent:
            raise ValueError("Batch-dependent reference evaluation requires --allow-batch-dependent")
        configured_batch_size = int(batch_size_override)
    identity = {
        "model": model_name,
        "database_encoder": database_encoder,
        "transform": transform_name,
        "query_checkpoint_sha256": file_sha256(query_checkpoint),
        "database_checkpoint_sha256": file_sha256(database_checkpoint),
        "input_size": (
            list(spec["native_size"])
            if transform_name == "native"
            else [config.vpr.image_height, config.vpr.image_width]
        ),
        "extraction_batch_size": configured_batch_size,
        "batch_dependent_mode": bool(spec.get("batch_dependent") and configured_batch_size > 1),
    }
    tracker = TrainingTracker(
        config,
        job_type="evaluate-reference-suite",
        default_name=f"suite-{model_name}-{database_encoder}-{transform_name}",
        output_dir=root,
        run_key=f"{model_name}:{database_encoder}:{transform_name}",
    )
    cached_payload = None
    if summary_path.is_file():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if all(cached.get(key) == value for key, value in identity.items()):
            changed = False
            for name in set(selected).intersection(cached.get("datasets", {})):
                dataset, split, radius, yaw, _ = SUITE_PROTOCOLS[name]
                database, queries = suite_rows(config, dataset, split)
                query_fingerprint = query_protocol_fingerprint(
                    database, queries, radius, yaw
                )
                height, width = identity["input_size"]
                padding = "none" if transform_name == "native" else "imagenet-mean-centered"
                eval_fingerprint = evaluation_fingerprint(
                    query_fingerprint,
                    transform="direct-resize" if transform_name == "native" else "fit-and-pad",
                    height=height,
                    width=width,
                    padding=padding,
                )
                metrics = cached["datasets"][name]
                if (
                    metrics.get("query_protocol_fingerprint") != query_fingerprint
                    or metrics.get("evaluation_fingerprint") != eval_fingerprint
                ):
                    metrics["query_protocol_fingerprint"] = query_fingerprint
                    metrics["evaluation_fingerprint"] = eval_fingerprint
                    metrics["protocol_fingerprint"] = query_fingerprint
                    write_json(cached["datasets"][name], root / name / "metrics.json")
                    changed = True
            if set(selected).issubset(cached.get("datasets", {})):
                if changed:
                    write_json(cached, summary_path)
                tracker.summary(_tracking_values(cached))
                tracker.finish()
                return summary_path
            cached_payload = cached

    query_model = build_reference_model(spec)
    database_model = (
        build_reference_model({**spec, "checkpoint": str(database_checkpoint)})
        if database_encoder == "original" else query_model
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_model.eval().to(device)
    database_model.eval().to(device)
    if transform_name == "native":
        height, width = spec["native_size"]
        transform = FixedResizeTransform(height, width)
        batch_size = int(spec.get("native_batch_size", 1))
    else:
        height, width = config.vpr.image_height, config.vpr.image_width
        transform = VPRTransform(height, width)
        batch_size = max(1, config.vpr.batch_size * 2)
    batch_size = configured_batch_size

    payload = cached_payload or {
        **identity,
        "descriptor_dim": int(spec["descriptor_dim"]),
        "parameters": sum(parameter.numel() for parameter in query_model.parameters()),
        "reference_note": spec.get("note", ""),
        "nightcities_available": False,
        "datasets": {},
    }
    for name in selected:
        if name in payload["datasets"]:
            continue
        dataset, split, radius, yaw, _ = SUITE_PROTOCOLS[name]
        database, queries = suite_rows(config, dataset, split)
        query_fingerprint = query_protocol_fingerprint(database, queries, radius, yaw)
        transform_kind = "direct-resize" if transform_name == "native" else "fit-and-pad"
        padding = "none" if transform_name == "native" else "imagenet-mean-centered"
        eval_fingerprint = evaluation_fingerprint(
            query_fingerprint,
            transform=transform_kind,
            height=height,
            width=width,
            padding=padding,
        )
        metrics = evaluate_dual_encoder(
            database_model,
            query_model,
            database,
            queries,
            transform,
            radius,
            yaw,
            config.evaluation.recalls,
            device,
            batch_size,
            config.vpr.workers,
        )
        metrics.update({
            "dataset": dataset,
            "split": split,
            "positive_radius_m": radius,
            "positive_yaw_deg": yaw,
            "database": len(database),
            "queries": len(queries),
            "query_protocol_fingerprint": query_fingerprint,
            "evaluation_fingerprint": eval_fingerprint,
            "protocol_fingerprint": query_fingerprint,
            "input_size": [height, width],
            "transform": transform_name,
            "transform_kind": transform_kind,
            "interpolation": "bicubic",
            "padding": padding,
            "normalization": "imagenet",
            "database_encoder": database_encoder,
            "descriptor_dim": int(spec["descriptor_dim"]),
            "extraction_batch_size": batch_size,
            "batch_dependent_mode": identity["batch_dependent_mode"],
            "query_checkpoint_sha256": identity["query_checkpoint_sha256"],
            "database_checkpoint_sha256": identity["database_checkpoint_sha256"],
        })
        payload["datasets"][name] = metrics
        write_json(metrics, root / name / "metrics.json")
    write_json(payload, summary_path)
    tracker.summary(_tracking_values(payload))
    tracker.finish()
    return summary_path


def compare_suite_bootstrap(
    candidate_path: str | Path, baseline_path: str | Path, output: str | Path,
    *, samples: int = 10_000,
) -> Path:
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    candidate_sets = candidate.get("datasets", {})
    baseline_sets = baseline.get("datasets", {})
    names = sorted(set(candidate_sets) & set(baseline_sets))
    if not names:
        raise ValueError("Suite summaries have no common datasets")
    comparisons = {}
    for name in names:
        first, second = candidate_sets[name], baseline_sets[name]
        first_query = first.get(
            "query_protocol_fingerprint", first.get("protocol_fingerprint")
        )
        second_query = second.get(
            "query_protocol_fingerprint", second.get("protocol_fingerprint")
        )
        if first_query != second_query:
            raise ValueError(f"Query protocol fingerprint mismatch for {name}")
        first_success = first["success_by_k"]["1"]
        second_success = second["success_by_k"]["1"]
        first_valid = first.get("valid_by_query", [1] * len(first_success))
        second_valid = second.get("valid_by_query", [1] * len(second_success))
        if first_valid != second_valid:
            raise ValueError(f"Valid-query mask mismatch for {name}")
        valid_indices = [index for index, valid in enumerate(first_valid) if valid]
        if not valid_indices:
            raise ValueError(f"No valid queries for paired bootstrap on {name}")
        first_valid_success = [first_success[index] for index in valid_indices]
        second_valid_success = [second_success[index] for index in valid_indices]
        comparisons[name] = {
            "query_protocol_fingerprint": first_query,
            "evaluation_fingerprint_candidate": first.get("evaluation_fingerprint"),
            "evaluation_fingerprint_baseline": second.get("evaluation_fingerprint"),
            "strictly_controlled_preprocessing": bool(
                first.get("evaluation_fingerprint")
                and first.get("evaluation_fingerprint") == second.get("evaluation_fingerprint")
            ),
            "protocol_fingerprint": first_query,
            "valid_queries": len(valid_indices),
            "candidate_recall_at_1": first["recalls"]["1"],
            "baseline_recall_at_1": second["recalls"]["1"],
            "paired_bootstrap_r1": paired_bootstrap(
                first_valid_success, second_valid_success, samples,
            ),
        }
    payload = {
        "candidate": str(Path(candidate_path).resolve()),
        "baseline": str(Path(baseline_path).resolve()),
        "bootstrap_samples": int(samples), "datasets": comparisons,
    }
    output = Path(output)
    write_json(payload, output)
    return output
