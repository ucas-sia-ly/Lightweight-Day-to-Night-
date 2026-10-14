"""TensorBoard helpers shared by training routes; no model or optimizer state."""
from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch


@contextmanager
def create_summary_writer(cfg, output_dir: str | Path, purge_step: int | None = None):
    if not cfg.tensorboard_enabled:
        yield None
        return
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError(
            "TensorBoard logging requires tensorboard; install the updated project "
            "or run pip install tensorboard. Set tensorboard_enabled=false to disable logging."
        ) from error
    log_dir = Path(cfg.tensorboard_log_dir).expanduser() if cfg.tensorboard_log_dir else Path(output_dir) / "tensorboard"
    if purge_step is not None:
        # TensorBoard reads files lexicographically. Within one second its
        # unpadded writer IDs can sort 10 before 9, applying the resume marker
        # before the stale events it must purge. Give the new file a later
        # timestamp using the public writer API, without rewriting old logs.
        timestamps = []
        for path in log_dir.glob("events.out.tfevents.*"):
            fields = path.name.split(".")
            if len(fields) > 3 and fields[3].isdigit():
                timestamps.append(int(fields[3]))
        if timestamps:
            delay = max(timestamps) + 1 - time.time()
            if delay > 1.1:
                raise RuntimeError("TensorBoard event filenames are ahead of the current clock; correct the clock or choose a new log directory before resuming")
            if delay > 0:
                time.sleep(delay + 0.005)
    writer = SummaryWriter(
        log_dir=str(log_dir), purge_step=purge_step, flush_secs=cfg.tensorboard_flush_secs,
    )
    print(f"TensorBoard logs: {log_dir.resolve()}")
    try:
        writer.add_text(
            "run/config", "```json\n" + json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n```",
            global_step=max(0, (purge_step or 1) - 1),
        )
        yield writer
    finally:
        try:
            writer.flush()
        finally:
            writer.close()


def log_scalars(writer, values: dict, step: int, prefix: str = "") -> None:
    if writer is None:
        return
    prefix = prefix.rstrip("/")
    for name, value in values.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        value = float(value)
        if math.isfinite(value):
            writer.add_scalar(f"{prefix}/{name}" if prefix else name, value, step)


def log_patch_diagnostics(writer, captures: dict, step: int) -> None:
    """Use already-computed patches only, without a second encoder forward."""
    if writer is None:
        return
    for name, capture in captures.items():
        if name == "reference_target":
            writer.add_images("images/reference_reliability_target", capture, step)
            continue
        mean = capture["images"].new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = capture["images"].new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        images = (capture["images"].float() * std + mean).clamp(0.0, 1.0)
        writer.add_images(f"images/{name}", images, step)
        for key in ("reliability", "dustbin_assignment"):
            values = capture[key].float()
            writer.add_images(f"images/{name}_{key}", values, step)
            valid_values = values[capture["valid_mask"]]
            if valid_values.numel() and torch.isfinite(valid_values).all():
                writer.add_histogram(f"patches/{name}/{key}", valid_values, step, max_bins=32)
