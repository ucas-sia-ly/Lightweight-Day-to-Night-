from __future__ import annotations

import json
from pathlib import Path

import torch

from .checkpointing import file_sha256, write_json
from .config import ExperimentConfig
from .generator.translate import load_translated_database
from .reference_models import (
    evaluate_dual_encoder,
    evaluation_fingerprint,
    query_protocol_fingerprint,
    suite_rows,
)
from .runtime import benchmark_model, load_vpr_model
from .tracking import TrainingTracker
from .transforms import VPRTransform
from .vpr.model import resolve_initial_checkpoint


SUITE_PROTOCOLS = {
    "robotcar_project": ("robotcar", "test", 10.0, 30.0, "test"),
    "robotcar_paper": ("robotcar", "test", 25.0, None, "test"),
    "svox_night": ("svox_night", "test", 25.0, None, "test"),
    "msls_night": ("msls_night", "val", 25.0, None, "validation"),
    "tokyo_night": ("tokyo_night", "test", 25.0, None, "test"),
    "tokyo_sunset": ("tokyo_sunset", "test", 25.0, None, "test"),
    "msls_day_val": ("msls_day_val", "val", 25.0, None, "validation"),
    "svox_day_val": ("svox_day_val", "val", 25.0, None, "validation"),
    "robotcar_day": ("robotcar_day", "test", 10.0, 30.0, "test"),
    "svox_day": ("svox_day", "test", 25.0, None, "test"),
    "svox_sun": ("svox_sun", "test", 25.0, None, "test"),
    "tokyo_day": ("tokyo_day", "test", 25.0, None, "test"),
}
NIGHT_PROTOCOLS = (
    "robotcar_project", "svox_night", "msls_night", "tokyo_night",
    "tokyo_sunset",
)
DAY_PROTOCOLS = (
    "msls_day_val", "svox_day_val", "robotcar_day", "svox_day", "svox_sun",
    "tokyo_day",
)
GENERATED_NIGHT_PROTOCOLS = tuple(name for name in NIGHT_PROTOCOLS if name != "tokyo_sunset")


def expand_suite_datasets(datasets: tuple[str, ...], *, generated_night: bool = False):
    aliases = set(datasets)
    if "all-conditions" in aliases:
        return NIGHT_PROTOCOLS + DAY_PROTOCOLS
    if "all-day" in aliases:
        return DAY_PROTOCOLS
    if "all" in aliases:
        return GENERATED_NIGHT_PROTOCOLS if generated_night else NIGHT_PROTOCOLS
    return datasets


def _evaluation_rows(config: ExperimentConfig, dataset: str, split: str):
    database, queries = suite_rows(config, dataset, split)
    manifest_sha256 = ""
    if config.evaluation.database_domain == "generated_night":
        database, manifest_sha256 = load_translated_database(config, dataset, database)
    return database, queries, manifest_sha256


def _database_manifest_hashes(config: ExperimentConfig, selected: tuple[str, ...]) -> dict:
    if config.evaluation.database_domain == "day":
        return {}
    hashes = {}
    for name in selected:
        dataset, split, _, _, _ = SUITE_PROTOCOLS[name]
        database, _, digest = _evaluation_rows(config, dataset, split)
        del database
        hashes[dataset] = digest
    return hashes


def _suite_summary_values(payload: dict, selected: tuple[str, ...]) -> dict:
    values = {}
    for name in selected:
        metrics = payload["datasets"][name]
        namespace = SUITE_PROTOCOLS[name][4]
        prefix = f"{namespace}/{name}"
        for recall, value in metrics["recalls"].items():
            values[f"{prefix}/recall_at_{recall}"] = value
        values[f"{prefix}/mrr"] = metrics["mrr"]
        values[f"{prefix}/valid_queries"] = metrics["valid_queries"]
        for key in (
            "query_protocol_fingerprint", "evaluation_fingerprint",
            "protocol_fingerprint",
        ):
            if key in metrics:
                values[f"{prefix}/{key}"] = metrics[key]
    benchmark = payload["benchmark"]
    values.update({
        "suite/checkpoint_sha256": payload["checkpoint_sha256"],
        "suite/manifest_sha256": payload["manifest_sha256"],
        "suite/descriptor_dim": payload["descriptor_dim"],
        "suite/initialization": payload["initialization"],
        "suite/encoder": payload["encoder"],
        "suite/input_size": payload["input_size"],
        "suite/database_domain": payload.get("database_domain", "day"),
        "suite/parameters": benchmark["parameters"],
        "suite/latency_p50_ms": benchmark["latency_ms"]["p50"],
        "suite/latency_p95_ms": benchmark["latency_ms"]["p95"],
        "suite/peak_cuda_mib": benchmark["peak_cuda_mib"],
    })
    return values


def evaluate_trained_suite(
    config: ExperimentConfig,
    checkpoint: str | Path,
    datasets: tuple[str, ...] = ("all",),
) -> Path:
    selected = expand_suite_datasets(
        datasets, generated_night=config.evaluation.database_domain == "generated_night"
    )
    if config.evaluation.database_domain == "generated_night" and any(
        name not in GENERATED_NIGHT_PROTOCOLS for name in selected
    ):
        raise ValueError("generated_night database evaluation requires night queries")
    unknown = sorted(set(selected) - set(SUITE_PROTOCOLS))
    if unknown:
        raise ValueError(f"Unknown trained-suite datasets: {unknown}")
    checkpoint_path = Path(checkpoint).expanduser()
    suffix = (
        "final_suite" if config.evaluation.database_domain == "day"
        else "final_suite_generated_night"
    )
    output_dir = Path(config.vpr.output_dir) / suffix
    summary_path = output_dir / "summary.json"
    checkpoint_sha256 = file_sha256(checkpoint_path)
    manifest_sha256 = file_sha256(Path(config.data.generated_root) / "manifest.csv")
    database_manifest_sha256 = _database_manifest_hashes(config, selected)
    tracker = TrainingTracker(
        config,
        job_type="evaluate-trained-suite",
        default_name=f"suite-{config.vpr.model_family}-{config.vpr.descriptor_dim}",
        output_dir=output_dir,
    )
    cached_payload = None
    if summary_path.is_file():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        cache_matches = (
            cached.get("checkpoint_sha256") == checkpoint_sha256
            and cached.get("manifest_sha256") == manifest_sha256
            and cached.get("database_domain", "day") == config.evaluation.database_domain
            and cached.get("database_manifest_sha256", {}) == database_manifest_sha256
            and cached.get("model_family") == config.vpr.model_family
            and cached.get("encoder") == config.vpr.model_family
            and cached.get("aggregator") == config.vpr.aggregator
            and cached.get("descriptor_dim") == config.vpr.descriptor_dim
            and cached.get("input_size") == [
                config.vpr.image_height, config.vpr.image_width
            ]
        )
        if cache_matches:
            changed = False
            for name in set(selected).intersection(cached.get("datasets", {})):
                metrics = cached["datasets"][name]
                dataset, split, radius, yaw, _ = SUITE_PROTOCOLS[name]
                database, queries, _ = _evaluation_rows(config, dataset, split)
                query_fingerprint = query_protocol_fingerprint(
                    database, queries, radius, yaw
                )
                eval_fingerprint = evaluation_fingerprint(
                    query_fingerprint,
                    transform="fit-and-pad",
                    height=config.vpr.image_height,
                    width=config.vpr.image_width,
                    padding="imagenet-mean-centered",
                )
                if (
                    metrics.get("query_protocol_fingerprint") != query_fingerprint
                    or metrics.get("evaluation_fingerprint") != eval_fingerprint
                ):
                    metrics["query_protocol_fingerprint"] = query_fingerprint
                    metrics["evaluation_fingerprint"] = eval_fingerprint
                    metrics["protocol_fingerprint"] = query_fingerprint
                    write_json(metrics, output_dir / name / "metrics.json")
                    changed = True
            if set(selected).issubset(cached.get("datasets", {})):
                if changed:
                    write_json(cached, summary_path)
                tracker.summary(_suite_summary_values(cached, selected))
                tracker.finish()
                return summary_path
            cached_payload = cached

    model, state, checkpoint_path = load_vpr_model(config, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    transform = VPRTransform(config.vpr.image_height, config.vpr.image_width)
    payload = cached_payload or {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "database_domain": config.evaluation.database_domain,
        "database_manifest_sha256": database_manifest_sha256,
        "checkpoint_epoch": int(state.get("epoch", 0)),
        "model_family": config.vpr.model_family,
        "initialization": config.vpr.init_checkpoint or "torchvision_imagenet",
        "encoder": config.vpr.model_family,
        "aggregator": config.vpr.aggregator,
        "descriptor_dim": config.vpr.descriptor_dim,
        "input_size": [config.vpr.image_height, config.vpr.image_width],
        "datasets": {},
    }
    if config.vpr.init_checkpoint:
        initial_checkpoint = Path(resolve_initial_checkpoint(
            config.vpr.init_checkpoint, config.vpr.descriptor_dim
        ))
        payload["initial_checkpoint_sha256"] = file_sha256(initial_checkpoint)
    if "benchmark" not in payload:
        payload["benchmark"] = benchmark_model(
            config, checkpoint_path, warmup=10, iterations=100, track=False
        )
    for name in selected:
        if name in payload["datasets"]:
            continue
        dataset, split, radius, yaw, namespace = SUITE_PROTOCOLS[name]
        database, queries, database_digest = _evaluation_rows(config, dataset, split)
        query_fingerprint = query_protocol_fingerprint(database, queries, radius, yaw)
        eval_fingerprint = evaluation_fingerprint(
            query_fingerprint,
            transform="fit-and-pad",
            height=config.vpr.image_height,
            width=config.vpr.image_width,
            padding="imagenet-mean-centered",
        )
        metrics = evaluate_dual_encoder(
            model, model, database, queries, transform, radius, yaw,
            config.evaluation.recalls, device,
            max(1, config.vpr.batch_size * 2), config.vpr.workers,
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
            "transform": "fit-and-pad",
            "interpolation": "bicubic",
            "padding": "imagenet-mean-centered",
            "normalization": "imagenet",
            "database_descriptor_mib_fp32": (
                len(database) * config.vpr.descriptor_dim * 4 / 1024**2
            ),
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "manifest_sha256": payload["manifest_sha256"],
            "database_domain": payload["database_domain"],
            "database_manifest_sha256": database_digest,
            "initialization": payload["initialization"],
            "initial_checkpoint_sha256": payload.get("initial_checkpoint_sha256", ""),
            "input_size": payload["input_size"],
            "descriptor_dim": config.vpr.descriptor_dim,
            "parameters": payload["benchmark"]["parameters"],
            "fp16_peak_cuda_mib": payload["benchmark"]["peak_cuda_mib"],
            "batch1_latency_p50_ms": payload["benchmark"]["latency_ms"]["p50"],
            "batch1_latency_p95_ms": payload["benchmark"]["latency_ms"]["p95"],
        })
        payload["datasets"][name] = metrics
        write_json(metrics, output_dir / name / "metrics.json")
    write_json(payload, summary_path)
    tracker.summary(_suite_summary_values(payload, selected))
    tracker.finish()
    return summary_path
