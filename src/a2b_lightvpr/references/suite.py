from __future__ import annotations

import json
from pathlib import Path

import torch

from ..checkpointing import file_sha256, write_json
from ..config import ExperimentConfig
from .models import (
    FixedResizeTransform,
    _registry,
    audit_reference_models,
    build_reference_model,
    evaluation_fingerprint,
    evaluate_dual_encoder,
    query_protocol_fingerprint,
    suite_rows,
)
from ..trained_suite import SUITE_PROTOCOLS, expand_suite_datasets
from ..transforms import VPRTransform


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
        from ..vpr.model import eigenplaces_checkpoint
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
        "backbone": spec.get("backbone", ""),
        "official_checkpoint": bool(spec.get("official_checkpoint", True)),
        "training_status": spec.get("training_status", "official"),
        "required_artifact_sha256": {
            str(Path(path).expanduser().resolve()): file_sha256(path)
            for path in spec.get("required_files", [])
            if Path(path).expanduser().is_file()
        },
    }
    if spec.get("projection_checkpoint"):
        identity["projection_checkpoint_sha256"] = file_sha256(
            spec["projection_checkpoint"]
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
    return summary_path
