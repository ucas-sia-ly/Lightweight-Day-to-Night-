from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from ..checkpointing import (
    SCHEMA_VERSION,
    atomic_torch_save,
    data_fingerprints,
    restore_rng_state,
    rng_state,
    write_json,
)
from ..config import ExperimentConfig
from ..data import BalancedGeneratorDataset
from ..tracking import TrainingTracker
from .networks import (
    PatchDiscriminator,
    ResnetGenerator,
    lsgan_discriminator_loss,
    lsgan_generator_loss,
)
from .objectives import ContentObjective, build_content_objective


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _content_parameters(objective: ContentObjective):
    adversary_ids = {id(parameter) for parameter in objective.adversary_parameters()}
    return [
        parameter
        for parameter in objective.parameters()
        if parameter.requires_grad and id(parameter) not in adversary_ids
    ]


def _objective_state(objective: ContentObjective) -> dict:
    return objective.state_dict()


def _linear_decay(step: int, max_steps: int) -> float:
    halfway = max_steps // 2
    if step <= halfway:
        return 1.0
    return max(0.0, 1.0 - (step - halfway) / max(1, max_steps - halfway))


def _paired_charbonnier_loss(fake: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    if not bool(mask.any()):
        return fake.sum() * 0.0
    fake, target = fake[mask], target[mask]
    return torch.sqrt((fake - target).square() + 1e-6).mean()


def _checkpoint_payload(
    config: ExperimentConfig,
    step: int,
    generator,
    discriminator,
    objective,
    optimizer_g,
    optimizer_d,
    optimizer_adversary,
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
        "data_fingerprints": data_fingerprints(config),
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "content_objective": _objective_state(objective),
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "optimizer_adversary": optimizer_adversary.state_dict() if optimizer_adversary else None,
        "scheduler_g": scheduler_g.state_dict(),
        "scheduler_d": scheduler_d.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": rng_state(),
    }


def train_generator(config: ExperimentConfig, *, smoke_steps: int | None = None) -> Path:
    cfg = config.generator
    _set_seed(cfg.seed)
    device = _device()
    max_steps = int(smoke_steps or cfg.max_steps)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.as_dict(), output_dir / "resolved_config.json")
    tracker = TrainingTracker(
        config,
        job_type="train-generator",
        default_name=f"generator-{cfg.objective}-seed{cfg.seed}",
        output_dir=output_dir,
    )

    generator = ResnetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    objective = build_content_objective(
        cfg.objective, num_patches=cfg.num_patches, feature_dim=cfg.feature_dim
    ).to(device)
    generator_parameters = list(generator.parameters()) + _content_parameters(objective)
    optimizer_g = torch.optim.Adam(generator_parameters, cfg.lr, betas=(cfg.beta1, cfg.beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), cfg.lr, betas=(cfg.beta1, cfg.beta2))
    adversary_parameters = list(objective.adversary_parameters())
    optimizer_adversary = (
        torch.optim.Adam(adversary_parameters, cfg.lr, betas=(cfg.beta1, cfg.beta2))
        if adversary_parameters
        else None
    )
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
        if state["config_fingerprint"] != config.fingerprint():
            raise ValueError("Generator resume config does not match the current config")
        generator.load_state_dict(state["generator"])
        discriminator.load_state_dict(state["discriminator"])
        objective.load_state_dict(state["content_objective"], strict=False)
        optimizer_g.load_state_dict(state["optimizer_g"])
        optimizer_d.load_state_dict(state["optimizer_d"])
        if optimizer_adversary and state.get("optimizer_adversary"):
            optimizer_adversary.load_state_dict(state["optimizer_adversary"])
        scheduler_g.load_state_dict(state["scheduler_g"])
        scheduler_d.load_state_dict(state["scheduler_d"])
        scaler.load_state_dict(state["scaler"])
        restore_rng_state(state["rng"])
        start_step = int(state["step"])

    dataset = BalancedGeneratorDataset(
        config.data, max_steps * cfg.batch_size, cfg.seed, cfg.paired_probability
    )
    remaining = Subset(dataset, range(start_step * cfg.batch_size, len(dataset)))
    loader = DataLoader(
        remaining,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    progress = tqdm(total=max_steps, initial=start_step, desc=f"generator/{cfg.objective}")
    final_path = output_dir / "generator_final.pt"
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

        adversary_loss = None
        if optimizer_adversary is not None:
            optimizer_adversary.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                adversary_loss = objective.adversary_loss(source_features, fake_features)
            if adversary_loss is not None:
                scaler.scale(adversary_loss).backward(retain_graph=True)
                scaler.step(optimizer_adversary)

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
            paired_mask = batch["paired"].to(device)
            paired_charbonnier = _paired_charbonnier_loss(
                fake, target, paired_mask
            )
            loss_g = (
                loss_gan
                + cfg.content_weight * loss_content
                + cfg.paired_charbonnier_weight * paired_charbonnier
            )
        scaler.scale(loss_g).backward()
        scaler.step(optimizer_g)
        scale_before_update = scaler.get_scale()
        scaler.update()
        discriminator.requires_grad_(True)
        if scaler.get_scale() >= scale_before_update:
            scheduler_g.step()
            scheduler_d.step()

        progress.update(1)
        progress.set_postfix(g=f"{loss_g.item():.3f}", d=f"{loss_d.item():.3f}")
        if step == start_step + 1 or step % config.tracking.log_every_steps == 0:
            log_values = {
                "train/generator_total": float(loss_g.item()),
                "train/gan": float(loss_gan.item()),
                "train/content": float(loss_content.item()),
                "train/content_source": float(loss_content_source.item()),
                "train/paired_count": int(paired_mask.sum().item()),
                "train/paired_fraction": float(paired_mask.float().mean().item()),
                "train/discriminator": float(loss_d.item()),
                "train/lr_generator": float(optimizer_g.param_groups[0]["lr"]),
                "train/lr_discriminator": float(optimizer_d.param_groups[0]["lr"]),
                "train/samples_seen": int(step * cfg.batch_size),
            }
            if adversary_loss is not None:
                log_values["train/negative_generator"] = float(adversary_loss.item())
            if loss_content_identity is not None:
                log_values["train/content_identity"] = float(loss_content_identity.item())
            if bool(paired_mask.any()):
                log_values["train/paired_charbonnier"] = float(paired_charbonnier.item())
            tracker.log(log_values, step=step)
        if step == start_step + 1 or step % config.tracking.image_every_steps == 0:
            tracker.log_generator_images(source, fake, target, step=step)
        due = step % cfg.checkpoint_every == 0 or step == max_steps
        if due:
            state_path = output_dir / "checkpoints" / f"state_{step:07d}.pt"
            atomic_torch_save(
                _checkpoint_payload(
                    config,
                    step,
                    generator,
                    discriminator,
                    objective,
                    optimizer_g,
                    optimizer_d,
                    optimizer_adversary,
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
                    "data_fingerprints": data_fingerprints(config),
                    "generator": generator.state_dict(),
                },
                final_path,
            )
    progress.close()
    tracker.summary(
        {
            "final_step": max_steps,
            "objective": cfg.objective,
            "generator_checkpoint": str(final_path.resolve()),
        }
    )
    tracker.log_checkpoint(
        final_path,
        name=f"generator-{cfg.objective}-seed{cfg.seed}",
        aliases=["final", f"step-{max_steps}"],
    )
    tracker.finish()
    return final_path
