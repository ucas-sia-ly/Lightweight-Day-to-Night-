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
from .tracking import TrainingTracker
from .vpr.model import build_descriptor_model


def load_vpr_model(config: ExperimentConfig, checkpoint: str | Path | None = None):
    cfg = config.vpr
    checkpoint = Path(checkpoint or Path(cfg.output_dir) / "model_best.pt")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("kind") != "vpr_inference":
        raise ValueError("Expected an inference-only VPR checkpoint")
    model = build_descriptor_model(
        cfg.backbone,
        cfg.aggregator,
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        descriptor_dim=cfg.descriptor_dim,
        pretrained=False,
        model_family=cfg.model_family,
        init_checkpoint=(
            cfg.init_checkpoint if cfg.model_family == "mixvpr" else ""
        ),
    )
    model.load_state_dict(state["model"])
    return model, state, checkpoint


def benchmark_model(
    config: ExperimentConfig,
    checkpoint: str | Path | None = None,
    *,
    warmup: int = 100,
    iterations: int = 1_000,
    track: bool = True,
) -> dict:
    model, state, checkpoint_path = load_vpr_model(config, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        "backbone": config.vpr.backbone,
        "aggregator": config.vpr.aggregator,
        "image_size": [config.vpr.image_height, config.vpr.image_width],
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
    if track:
        tracker = TrainingTracker(
            config,
            job_type="benchmark-vpr",
            default_name=(
                f"benchmark-{config.vpr.backbone}-{config.vpr.aggregator}-"
                f"{config.vpr.image_height}x{config.vpr.image_width}"
            ),
            output_dir=config.vpr.output_dir,
        )
        tracker.summary({f"benchmark/{key}": value for key, value in result.items()})
        tracker.finish()
    return result


def export_onnx(config: ExperimentConfig, checkpoint: str | Path | None = None) -> Path:
    model, _, _ = load_vpr_model(config, checkpoint)
    model.eval()
    sample = torch.randn(1, 3, config.vpr.image_height, config.vpr.image_width)
    output = Path(config.vpr.output_dir) / "model.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        sample,
        output,
        input_names=["images"],
        output_names=["descriptors"],
        dynamic_axes={"images": {0: "batch"}, "descriptors": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    return output
