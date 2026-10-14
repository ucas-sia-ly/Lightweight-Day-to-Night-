from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from ..checkpointing import (
    SCHEMA_VERSION,
    atomic_torch_save,
    checkpoint_config_matches,
    data_fingerprints,
    restore_rng_state,
    rng_state,
    write_json,
)
from ..config import ExperimentConfig
from ..data import BalancedGeneratorDataset
from ..training_logging import create_summary_writer, log_scalars
from .networks import (
    PatchDiscriminator,
    ResnetGenerator,
    lsgan_discriminator_loss,
    lsgan_generator_loss,
)
from .objectives import ContentObjective, build_content_objective


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_fixed_random_state() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


def _objective_state(objective: ContentObjective) -> dict:
    return objective.state_dict()


def _linear_decay(step: int, max_steps: int) -> float:
    halfway = max_steps // 2
    if step <= halfway:
        return 1.0
    return max(0.0, 1.0 - (step - halfway) / max(1, max_steps - halfway))


def _checkpoint_payload(
    config: ExperimentConfig,
    step: int,
    generator,
    discriminator,
    objective,
    optimizer_g,
    optimizer_d,
    scheduler_g,
    scheduler_d,
    scaler,
):
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "generator_training_state",
        "step": int(step),
        "config_fingerprint": config.fingerprint(),
        "config": config.as_dict(),
        "data_fingerprints": data_fingerprints(config, purpose="generator"),
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "content_objective": _objective_state(objective),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "scheduler_g": scheduler_g.state_dict(),
        "scheduler_d": scheduler_d.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": rng_state(),
    }


def train_generator(config: ExperimentConfig, *, smoke_steps: int | None = None) -> Path:
    cfg = config.generator
    _set_fixed_random_state()
    device = _device()
    max_steps = int(smoke_steps or cfg.max_steps)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.as_dict(), output_dir / "resolved_config.json")
    generator = ResnetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    objective = build_content_objective(
        cfg.objective, num_patches=cfg.num_patches, feature_dim=cfg.feature_dim
    ).to(device)
    generator_parameters = list(generator.parameters()) + list(objective.parameters())
    optimizer_g = torch.optim.Adam(generator_parameters, cfg.lr, betas=(cfg.beta1, cfg.beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), cfg.lr, betas=(cfg.beta1, cfg.beta2))
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(
        optimizer_g, lambda step: _linear_decay(step, max_steps)
    )
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(
        optimizer_d, lambda step: _linear_decay(step, max_steps)
    )
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_step = 0

    if cfg.resume:
        state = torch.load(cfg.resume, map_location=device, weights_only=False)
        if not checkpoint_config_matches(state, config, ("data", "generator")):
            raise ValueError("Generator resume config does not match the current config")
        current_data = data_fingerprints(config, purpose="generator")
        if any(state.get("data_fingerprints", {}).get(key) != value for key, value in current_data.items()):
            raise ValueError("Generator resume training manifest does not match the checkpoint")
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        objective.load_state_dict(state["content_objective"], strict=False)
        optimizer_g.load_state_dict(state["optimizer_g"])
        optimizer_d.load_state_dict(state["optimizer_d"])
        scheduler_g.load_state_dict(state["scheduler_g"])
        scheduler_d.load_state_dict(state["scheduler_d"])
        scaler.load_state_dict(state["scaler"])
        restore_rng_state(state["rng"])
        start_step = int(state["step"])

    dataset = BalancedGeneratorDataset(config.data, max_steps * cfg.batch_size)
    remaining = Subset(dataset, range(start_step * cfg.batch_size, len(dataset)))
    loader = DataLoader(
        remaining,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    final_path = output_dir / "generator_final.pt"
    with create_summary_writer(
        cfg, output_dir, purge_step=start_step + 1 if cfg.resume else None
    ) as writer, tqdm(
        total=max_steps, initial=start_step, desc=f"generator/{cfg.objective}"
    ) as progress:
        for step, batch in enumerate(loader, start=start_step + 1):
            if step > max_steps:
                break
            source = batch["source"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            with torch.autocast(device.type, enabled=use_amp):
                fake, source_features = generator(source, return_features=True)
                _, fake_features = generator.encode_features(fake)
                identity_features = target_features = None
                if cfg.objective == "qsattn" and cfg.qs_nce_identity:
                    identity, target_features = generator(target, return_features=True)
                    _, identity_features = generator.encode_features(identity)

            optimizer_d.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss_d = lsgan_discriminator_loss(discriminator(target), discriminator(fake.detach()))
            scaler.scale(loss_d).backward()
            scaler.step(optimizer_d)

            discriminator.requires_grad_(False)
            optimizer_g.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss_gan = lsgan_generator_loss(discriminator(fake))
                loss_content_source = objective.generator_loss(
                    source, fake, source_features, fake_features
                )
                loss_content_identity = (
                    objective.identity_loss(target_features, identity_features)
                    if target_features is not None and identity_features is not None else None
                )
                loss_content = (
                    0.5 * (loss_content_source + loss_content_identity)
                    if loss_content_identity is not None else loss_content_source
                )
                loss_g = loss_gan + cfg.content_weight * loss_content
            scaler.scale(loss_g).backward()
            scaler.step(optimizer_g)
            scale_before_update = scaler.get_scale()
            scaler.update()
            discriminator.requires_grad_(True)
            if scaler.get_scale() >= scale_before_update:
                scheduler_g.step()
                scheduler_d.step()

            log_due = (
                step == start_step + 1
                or step % cfg.tensorboard_log_interval == 0
                or step == max_steps
            )
            if writer is not None and log_due:
                metrics = {
                    "loss_g": loss_g,
                    "loss_d": loss_d,
                    "loss_gan": loss_gan,
                    "loss_content": loss_content,
                    "loss_content_source": loss_content_source,
                    "amp_scale": scaler.get_scale(),
                    "amp_step_skipped": float(scaler.get_scale() < scale_before_update),
                }
                if loss_content_identity is not None:
                    metrics["loss_content_identity"] = loss_content_identity
                log_scalars(writer, metrics, step, prefix="train/")
                log_scalars(
                    writer,
                    {
                        "generator": optimizer_g.param_groups[0]["lr"],
                        "discriminator": optimizer_d.param_groups[0]["lr"],
                    },
                    step,
                    prefix="lr/",
                )
            image_due = cfg.tensorboard_image_interval > 0 and (
                step == start_step + 1
                or step % cfg.tensorboard_image_interval == 0
                or step == max_steps
            )
            if writer is not None and image_due:
                # Reuse the current batch and forward results; logging adds no model pass.
                samples = {"day": source, "fake_night": fake, "real_night": target}
                if loss_content_identity is not None:
                    samples["identity_night"] = identity
                for name, images in samples.items():
                    preview = images[:3].detach().float().add(1).mul(0.5).clamp(0, 1).cpu()
                    writer.add_images(f"images/{name}", preview, global_step=step)

            progress.update(1)
            progress.set_postfix(g=f"{loss_g.item():.3f}", d=f"{loss_d.item():.3f}")
            due = step % cfg.checkpoint_every == 0 or step == max_steps
            if due:
                state_path = output_dir / "checkpoints" / "latest.pt"
                atomic_torch_save(
                    _checkpoint_payload(
                        config,
                        step,
                        generator,
                        discriminator,
                        objective,
                        optimizer_g,
                        optimizer_d,
                        scheduler_g,
                        scheduler_d,
                        scaler,
                    ),
                    state_path,
                )
                atomic_torch_save(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "kind": "generator_inference",
                        "step": int(step),
                        "objective": cfg.objective,
                        "config_fingerprint": config.fingerprint(),
                        "data_fingerprints": data_fingerprints(config, purpose="generator"),
                        "generator": generator.state_dict(),
                    },
                    final_path,
                )
    return final_path
