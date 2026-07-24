from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig
from .transforms import IMAGENET_MEAN, IMAGENET_STD, denormalize_generator


class TrainingTracker:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        job_type: str,
        default_name: str,
        output_dir: str | Path,
        run_key: str = "",
    ):
        self.config = config.tracking
        self.run = None
        self._wandb = None
        if not self.config.enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B tracking is enabled but wandb is not installed; "
                "run `pip install -e '.[tracking]'`"
            ) from exc

        stable_key = f"{job_type}:{config.fingerprint()}"
        if run_key:
            stable_key = f"{stable_key}:{run_key}"
        run_id = hashlib.sha1(stable_key.encode("utf-8")).hexdigest()[:16]
        self._wandb = wandb
        self.run = wandb.init(
            project=self.config.project,
            entity=self.config.entity or None,
            group=self.config.group or None,
            name=self.config.name or default_name,
            tags=list(self.config.tags),
            job_type=job_type,
            id=run_id,
            resume="allow",
            mode=self.config.mode,
            dir=str(Path(output_dir).resolve()),
            config=config.as_dict(),
        )
        self.run.config.update(
            {"command": " ".join(sys.argv), "config_fingerprint": config.fingerprint()},
            allow_val_change=True,
        )
        self.run.define_metric("trainer/global_step")
        self.run.define_metric("*", step_metric="trainer/global_step")

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(self, values: dict[str, Any], *, step: int) -> None:
        if self.run is not None:
            self.run.log({"trainer/global_step": int(step), **values})

    def log_generator_images(
        self,
        source: torch.Tensor,
        fake: torch.Tensor,
        target: torch.Tensor,
        *,
        step: int,
    ) -> None:
        if self.run is None:
            return
        count = min(4, source.shape[0])
        values = {}
        for key, batch in (("source_day", source), ("generated_night", fake), ("target_night", target)):
            values[f"images/{key}"] = [
                self._wandb.Image(denormalize_generator(image).detach().cpu())
                for image in batch[:count]
            ]
        self.run.log({"trainer/global_step": int(step), **values})

    def log_vpr_images(
        self, day: torch.Tensor, fake: torch.Tensor, *, step: int
    ) -> None:
        if self.run is None:
            return
        mean = day.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = day.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        count = min(4, day.shape[0])
        values = {}
        for key, batch in (("day", day), ("fake_night", fake)):
            display = (batch[:count] * std + mean).clamp(0.0, 1.0)
            values[f"images/{key}"] = [
                self._wandb.Image(image.detach().cpu()) for image in display
            ]
        self.run.log({"trainer/global_step": int(step), **values})

    def log_checkpoint(self, path: str | Path, *, name: str, aliases: list[str]) -> None:
        if self.run is None or not self.config.upload_checkpoints:
            return
        artifact = self._wandb.Artifact(name=name, type="model")
        artifact.add_file(str(path))
        self.run.log_artifact(artifact, aliases=aliases)

    def summary(self, values: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.summary.update(values)

    def finish(self, exit_code: int = 0) -> None:
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
            self.run = None
