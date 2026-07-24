from __future__ import annotations

import json
from pathlib import Path

import torch

from .checkpointing import file_sha256, write_json
from .config import ExperimentConfig
from .reference_models import evaluate_dual_encoder, protocol_fingerprint, suite_rows
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
}


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
        if "protocol_fingerprint" in metrics:
            values[f"{prefix}/protocol_fingerprint"] = metrics[
                "protocol_fingerprint"
            ]
    benchmark = payload["benchmark"]
    values.update({
        "suite/checkpoint_sha256": payload["checkpoint_sha256"],
        "suite/manifest_sha256": payload["manifest_sha256"],
        "suite/descriptor_dim": payload["descriptor_dim"],
        "suite/initialization": payload["initialization"],
        "suite/input_size": payload["input_size"],
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
    selected = tuple(SUITE_PROTOCOLS) if "all" in datasets else datasets
    unknown = sorted(set(selected) - set(SUITE_PROTOCOLS))
    if unknown:
        raise ValueError(f"Unknown trained-suite datasets: {unknown}")
    checkpoint_path = Path(checkpoint).expanduser()
    output_dir = Path(config.vpr.output_dir) / "final_suite"
    summary_path = output_dir / "summary.json"
    checkpoint_sha256 = file_sha256(checkpoint_path)
    manifest_sha256 = file_sha256(Path(config.data.generated_root) / "manifest.csv")
    tracker = TrainingTracker(
        config,
        job_type="evaluate-trained-suite",
        default_name=f"suite-{config.vpr.model_family}-{config.vpr.descriptor_dim}",
        output_dir=output_dir,
    )
    if summary_path.is_file():
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        cache_matches = (
            cached.get("checkpoint_sha256") == checkpoint_sha256
            and cached.get("manifest_sha256") == manifest_sha256
            and cached.get("model_family") == config.vpr.model_family
            and cached.get("backbone") == config.vpr.backbone
            and cached.get("aggregator") == config.vpr.aggregator
            and cached.get("descriptor_dim") == config.vpr.descriptor_dim
            and cached.get("input_size") == [
                config.vpr.image_height, config.vpr.image_width
            ]
            and set(selected).issubset(cached.get("datasets", {}))
        )
        if cache_matches:
            changed = False
            for name in selected:
                metrics = cached["datasets"][name]
                dataset, split, radius, yaw, _ = SUITE_PROTOCOLS[name]
                database, queries = suite_rows(config, dataset, split)
                fingerprint = protocol_fingerprint(database, queries, radius, yaw)
                if metrics.get("protocol_fingerprint") != fingerprint:
                    metrics["protocol_fingerprint"] = fingerprint
                    write_json(metrics, output_dir / name / "metrics.json")
                    changed = True
            if changed:
                write_json(cached, summary_path)
            tracker.summary(_suite_summary_values(cached, selected))
            tracker.finish()
            return summary_path

    model, state, checkpoint_path = load_vpr_model(config, checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    transform = VPRTransform(config.vpr.image_height, config.vpr.image_width)
    payload = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "checkpoint_epoch": int(state.get("epoch", 0)),
        "model_family": config.vpr.model_family,
        "initialization": config.vpr.init_checkpoint or "torchvision_imagenet",
        "backbone": config.vpr.backbone,
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
    payload["benchmark"] = benchmark_model(
        config, checkpoint_path, warmup=10, iterations=100, track=False
    )
    for name in selected:
        dataset, split, radius, yaw, namespace = SUITE_PROTOCOLS[name]
        database, queries = suite_rows(config, dataset, split)
        fingerprint = protocol_fingerprint(database, queries, radius, yaw)
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
            "protocol_fingerprint": fingerprint,
            "database_descriptor_mib_fp32": (
                len(database) * config.vpr.descriptor_dim * 4 / 1024**2
            ),
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "manifest_sha256": payload["manifest_sha256"],
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
