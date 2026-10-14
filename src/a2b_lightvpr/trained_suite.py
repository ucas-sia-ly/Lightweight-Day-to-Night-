from __future__ import annotations

import json
from pathlib import Path

import torch

from .checkpointing import file_sha256, write_json
from .config import ExperimentConfig
from .evaluation import resolve_database_domain
from .references.models import (
    evaluate_dual_encoder,
    evaluation_fingerprint,
    query_protocol_fingerprint,
    suite_rows,
    suite_protocol_metadata,
)
from .runtime import benchmark_model, load_vpr_model
from .transforms import VPRTransform
from .vpr.model import resolve_initial_checkpoint


SUITE_PROTOCOLS = {
    "robotcar_seasons_night": ("robotcar_seasons_night", "test", None, None, "test"),
    "svox_night": ("svox_night", "test", 25.0, None, "test"),
    "msls_night": ("msls_night", "val", 25.0, None, "validation"),
    "tokyo_night": ("tokyo_night", "test", 25.0, None, "test"),
    "tokyo_sunset": ("tokyo_sunset", "test", 25.0, None, "test"),
    "msls_day_val": ("msls_day_val", "val", 25.0, None, "validation"),
    "svox_day_val": ("svox_day_val", "val", 25.0, None, "validation"),
    "robotcar_seasons_day": ("robotcar_seasons_day", "test", None, None, "test"),
    "svox_day": ("svox_day", "test", 25.0, None, "test"),
    "svox_sun": ("svox_sun", "test", 25.0, None, "test"),
    "tokyo_day": ("tokyo_day", "test", 25.0, None, "test"),
}
NIGHT_PROTOCOLS = (
    "robotcar_seasons_night", "svox_night", "msls_night", "tokyo_night",
    "tokyo_sunset",
)
DAY_PROTOCOLS = (
    "msls_day_val", "svox_day_val", "robotcar_seasons_day", "svox_day", "svox_sun",
    "tokyo_day",
)
def expand_suite_datasets(datasets: tuple[str, ...]):
    aliases = set(datasets)
    if "all-conditions" in aliases:
        return NIGHT_PROTOCOLS + DAY_PROTOCOLS
    if "all-day" in aliases:
        return DAY_PROTOCOLS
    if "all" in aliases:
        return NIGHT_PROTOCOLS
    return datasets


def _evaluation_rows(config: ExperimentConfig, dataset: str, split: str):
    database, queries = suite_rows(config, dataset, split)
    database, digest = resolve_database_domain(database, config.evaluation, split, queries)
    return database, queries, digest


def _suite_summary_values(payload: dict, selected: tuple[str, ...]) -> dict:
    values = {}
    for name in selected:
        metrics = payload["datasets"][name]
        namespace = SUITE_PROTOCOLS[name][4]
        prefix = f"{namespace}/{name}"
        for recall, value in (metrics.get("recalls") or {}).items():
            values[f"{prefix}/recall_at_{recall}"] = value
        values[f"{prefix}/mrr"] = metrics["mrr"]
        values[f"{prefix}/valid_queries"] = metrics["valid_queries"]
        values[f"{prefix}/evaluation_status"] = metrics.get("evaluation_status", "measured")
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
        "suite/random_seed": payload.get("random_seed", 0),
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
    selected = expand_suite_datasets(datasets)
    unknown = sorted(set(selected) - set(SUITE_PROTOCOLS))
    if unknown:
        raise ValueError(f"Unknown trained-suite datasets: {unknown}")
    checkpoint_path = Path(checkpoint).expanduser()
    output_dir = Path(config.vpr.output_dir) / "final_suite"
    if config.evaluation.database_domain == "fake_night":
        output_dir = output_dir / "fake_night"
    summary_path = output_dir / "summary.json"
    checkpoint_sha256 = file_sha256(checkpoint_path)
    source_only = config.vpr.model_family == "stable_night_salad" and config.vpr.train_domains in {"day_only", "day_aug"}
    training_manifest = Path(config.data.vpr_source_manifest) if source_only else Path(config.data.generated_root) / "manifest.csv"
    manifest_sha256 = file_sha256(training_manifest)
    database_manifest_sha256 = (
        {"fake_database_manifest": file_sha256(config.evaluation.fake_database_manifest)}
        if config.evaluation.database_domain == "fake_night" else {}
    )
    stable_night = config.vpr.model_family == "stable_night_salad"
    stable_head_options = {
        "version": 1,
        "dustbin_enabled": config.vpr.stable_dustbin_enabled,
        "gamma_max": config.vpr.stable_dustbin_gamma_max,
        "sinkhorn_iterations": config.vpr.stable_sinkhorn_iterations,
    } if stable_night else {}
    if stable_night:
        initialization_checkpoint = config.vpr.init_checkpoint or config.vpr.backbone_checkpoint
    elif config.vpr.model_family in {"salad", "dino_mix"}:
        initialization_checkpoint = config.vpr.backbone_checkpoint
    else:
        initialization_checkpoint = config.vpr.init_checkpoint
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
            and cached.get("random_seed", 0) == config.vpr.random_seed
            and cached.get("evaluation_recalls") == list(config.evaluation.recalls)
            and (not stable_night or (
                cached.get("stable_head_options") == stable_head_options
            ))
        )
        if cache_matches:
            changed = False
            for name in set(cached.get("datasets", {})) - set(SUITE_PROTOCOLS):
                del cached["datasets"][name]
                changed = True
            training_identity = {
                "train_domains": config.vpr.train_domains,
                "cross_domain_aux": config.vpr.cross_domain_aux,
                "relation_weight": config.vpr.relation_weight,
                "relation_day_weight": config.vpr.relation_day_weight,
                "relation_queue_size": config.vpr.relation_queue_size,
                "relation_mask_self": config.vpr.relation_mask_self,
                "relation_queue_scope": config.vpr.relation_queue_scope,
            }
            if cached.get("random_seed") != config.vpr.random_seed:
                cached["random_seed"] = config.vpr.random_seed
                changed = True
            if cached.get("training") != training_identity:
                cached["training"] = training_identity
                changed = True
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
                    or any(metrics.get(key) != value for key, value in suite_protocol_metadata(config, dataset).items())
                ):
                    # Changed rows or transforms require new measurements/rankings.
                    del cached["datasets"][name]
                    changed = True
            if set(selected).issubset(cached.get("datasets", {})):
                if changed:
                    write_json(cached, summary_path)
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
        "initialization": initialization_checkpoint or "torchvision_imagenet",
        "encoder": config.vpr.model_family,
        "aggregator": config.vpr.aggregator,
        "descriptor_dim": config.vpr.descriptor_dim,
        "random_seed": config.vpr.random_seed,
        "training": {
            "train_domains": config.vpr.train_domains,
            "cross_domain_aux": config.vpr.cross_domain_aux,
            "relation_weight": config.vpr.relation_weight,
            "relation_day_weight": config.vpr.relation_day_weight,
            "relation_queue_size": config.vpr.relation_queue_size,
            "relation_mask_self": config.vpr.relation_mask_self,
            "relation_queue_scope": config.vpr.relation_queue_scope,
        },
        "input_size": [config.vpr.image_height, config.vpr.image_width],
        "stable_head_options": stable_head_options,
        "evaluation_recalls": list(config.evaluation.recalls),
        "datasets": {},
    }
    if initialization_checkpoint:
        initial_checkpoint = Path(resolve_initial_checkpoint(
            initialization_checkpoint, config.vpr.descriptor_dim
        ))
        if initial_checkpoint.is_file():
            payload["initial_checkpoint_sha256"] = file_sha256(initial_checkpoint)
        else:
            payload["initial_checkpoint_sha256"] = ""
            payload["initial_checkpoint_missing"] = str(initial_checkpoint)
    if "benchmark" not in payload:
        payload["benchmark"] = benchmark_model(
            config, checkpoint_path, warmup=10, iterations=100
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
            **suite_protocol_metadata(config, dataset),
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
    return summary_path
