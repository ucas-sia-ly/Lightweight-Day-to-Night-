from __future__ import annotations

import json
import resource
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .checkpointing import write_json
from .config import ExperimentConfig
from .vpr.model import build_descriptor_model


def vpr_device(config: ExperimentConfig) -> torch.device:
    available = torch.cuda.is_available()
    if config.vpr.require_cuda and not available:
        raise RuntimeError("This experiment requires CUDA, but PyTorch cannot access a GPU. Run in a GPU-enabled environment; use vpr.require_cuda=false only for intentional CPU checks.")
    return torch.device("cuda" if available else "cpu")


def load_vpr_model(config: ExperimentConfig, checkpoint: str | Path | None = None):
    cfg = config.vpr
    checkpoint = Path(checkpoint or Path(cfg.output_dir) / "model_best.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("kind") != "vpr_inference":
        raise ValueError("Expected an inference-only VPR checkpoint")
    expected_metadata = {
        "image_size": [cfg.image_height, cfg.image_width],
        "input_transform": cfg.input_transform,
        "model_family": cfg.model_family,
        "backbone_name": cfg.backbone_name,
    }
    for name, expected in expected_metadata.items():
        if name in state and state[name] != expected:
            raise ValueError(f"Inference checkpoint {name}={state[name]!r} does not match configuration {expected!r}")
    checkpoint_dim = state.get("descriptor_dim")
    if checkpoint_dim is not None and int(checkpoint_dim) != cfg.descriptor_dim:
        raise ValueError(
            f"Checkpoint descriptor dimension {checkpoint_dim} does not match "
            f"configuration dimension {cfg.descriptor_dim}"
        )
    checkpoint_seed = int(state.get("random_seed", 0))
    if checkpoint_seed != config.vpr.random_seed:
        raise ValueError(
            f"Checkpoint random seed {checkpoint_seed} does not match "
            f"configuration seed {config.vpr.random_seed}"
        )
    model = build_descriptor_model(
        cfg.aggregator,
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        descriptor_dim=cfg.descriptor_dim,
        pretrained=False,
        model_family=cfg.model_family,
        num_clusters=cfg.num_clusters,
        init_checkpoint=(
            cfg.init_checkpoint if cfg.model_family in {"mixvpr", "netvlad"} else ""
        ),
        backbone_name=cfg.backbone_name,
        backbone_repo=cfg.backbone_repo,
        backbone_checkpoint=cfg.backbone_checkpoint,
        backbone_trainable_blocks=cfg.backbone_trainable_blocks,
        aggregator_repo=cfg.aggregator_repo,
        stable_dustbin_enabled=cfg.stable_dustbin_enabled,
        stable_dustbin_gamma_init=cfg.stable_dustbin_gamma_init,
        stable_dustbin_gamma_max=cfg.stable_dustbin_gamma_max,
        stable_sinkhorn_iterations=cfg.stable_sinkhorn_iterations,
        salad_dropout=cfg.salad_dropout,
        # The inference checkpoint contains the complete model state. Avoid
        # requiring an unrelated original initialization file after relocation.
        load_initial_checkpoint=False,
    )
    model.load_state_dict(state["model"])
    return model, state, checkpoint


def benchmark_model(
    config: ExperimentConfig,
    checkpoint: str | Path | None = None,
    *,
    warmup: int = 100,
    iterations: int = 1_000,
) -> dict:
    device = vpr_device(config)
    model, state, checkpoint_path = load_vpr_model(config, checkpoint)
    model.eval().to(device)
    sample = torch.randn(
        1, 3, config.vpr.image_height, config.vpr.image_width, device=device
    )
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast(device.type, enabled=use_amp):
                model.encode(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            with torch.autocast(device.type, enabled=use_amp):
                model.encode(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1_000.0)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "encoder": config.vpr.model_family,
        "aggregator": config.vpr.aggregator,
        "image_size": [config.vpr.image_height, config.vpr.image_width],
        "input_transform": config.vpr.input_transform,
        "descriptor_dim": config.vpr.descriptor_dim,
        "parameters": parameters,
        "fp16_parameter_mib": parameters * 2 / 1024**2,
        "latency_ms": {
            "p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
            "mean": float(statistics.fmean(times)),
        },
        "peak_cuda_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else None
        ),
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "passes_30ms_proxy": bool(np.percentile(times, 95) <= 30.0),
    }
    measured_memory = (
        result["peak_cuda_mib"]
        if result["peak_cuda_mib"] is not None
        else result["process_peak_rss_mib"]
    )
    result["passes_1_5gib_memory_proxy"] = bool(measured_memory <= 1536.0)
    result["passes_deployment_proxy"] = bool(
        result["passes_30ms_proxy"] and result["passes_1_5gib_memory_proxy"]
    )
    output = Path(config.vpr.output_dir) / "benchmark.json"
    write_json(result, output)
    return result
