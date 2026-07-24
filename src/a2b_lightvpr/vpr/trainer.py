from __future__ import annotations

import copy
import math
import random
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from ..checkpointing import (
    SCHEMA_VERSION,
    atomic_torch_save,
    data_fingerprints,
    file_sha256,
    restore_rng_state,
    rng_state,
    write_json,
)
from ..config import ExperimentConfig
from ..data import CoverageBalancedSampler, PairedVPRDataset, RollingBalancedSampler
from ..evaluation import evaluate_model
from ..tracking import TrainingTracker
from ..transforms import VPRTransform
from .losses import MarginCosineClassifier, configurable_vpr_loss, vpr_training_loss
from .model import build_descriptor_model, resolve_initial_checkpoint
from .teacher_cache import TeacherDescriptorCache


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def optimizer_base_lrs(cfg) -> list[float]:
    return [cfg.backbone_lr, cfg.aggregator_lr, cfg.classifier_lr]


def build_vpr_scheduler(optimizer, cfg, steps_per_epoch: int):
    if cfg.scheduler == "cosine":
        main_epochs = max(1, cfg.epochs - cfg.classifier_warmup_epochs)
        total_steps = max(1, main_epochs * steps_per_epoch)

        def cosine_factor(step: int) -> float:
            progress = min(max(step, 0), total_steps) / total_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cfg.scheduler_min_lr_ratio + (1.0 - cfg.scheduler_min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[cosine_factor] * len(optimizer.param_groups)
        )
    if cfg.scheduler == "plateau":
        min_lrs = [
            group["lr"] * cfg.scheduler_min_lr_ratio
            for group in optimizer.param_groups
        ]
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            threshold=cfg.scheduler_threshold,
            threshold_mode="abs",
            min_lr=min_lrs,
        )
    return None


def learning_rates_at_minimum(optimizer, cfg) -> bool:
    return all(
        float(group["lr"]) <= base * cfg.scheduler_min_lr_ratio * (1.0 + 1e-6)
        for group, base in zip(optimizer.param_groups, optimizer_base_lrs(cfg))
    )


def accumulation_divisor(batch_index: int, loader_length: int, accumulation_steps: int) -> int:
    group_start = (batch_index // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, loader_length - group_start)


def freeze_batch_norm_statistics(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def configure_training_phase(model, optimizer, cfg, epoch: int) -> bool:
    warmup = epoch < cfg.classifier_warmup_epochs
    official_model = cfg.model_family != "custom"
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(not warmup)
    for parameter in model.aggregator.parameters():
        parameter.requires_grad_(not warmup or not official_model)
    if warmup:
        optimizer.param_groups[0]["lr"] = 0.0
        optimizer.param_groups[1]["lr"] = 0.0 if official_model else cfg.aggregator_lr
        optimizer.param_groups[2]["lr"] = cfg.classifier_warmup_lr
    elif epoch == cfg.classifier_warmup_epochs:
        for group, lr in zip(optimizer.param_groups, optimizer_base_lrs(cfg)):
            group["lr"] = lr
    return warmup


def build_vpr_optimizer(model, classifier, cfg):
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": cfg.backbone_lr},
            {"params": model.aggregator.parameters(), "lr": cfg.aggregator_lr},
            {"params": classifier.parameters(), "lr": cfg.classifier_lr},
        ],
        weight_decay=cfg.weight_decay,
    )


def _inference_checkpoint(
    config,
    epoch,
    score,
    epoch_zero_score,
    model,
    generated_manifest,
    preservation_teacher_fingerprint,
):
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "vpr_inference",
        "epoch": int(epoch),
        "val_recall1": float(score),
        "epoch_zero_recall1": float(epoch_zero_score),
        "config_fingerprint": config.fingerprint(),
        "data_fingerprints": data_fingerprints(config, generated=generated_manifest),
        "model": model.state_dict(),
        "preservation_teacher_fingerprint": preservation_teacher_fingerprint,
    }


def _vpr_checkpoint(
    config,
    epoch,
    best_score,
    epoch_zero_score,
    stale_epochs,
    model,
    classifier,
    optimizer,
    scheduler,
    scaler,
    generated_manifest,
    teacher_cache_fingerprint,
    preservation_teacher_fingerprint,
    optimizer_step,
    samples_seen,
):
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "vpr_training_state",
        "epoch": int(epoch),
        "best_val_recall1": float(best_score),
        "epoch_zero_recall1": float(epoch_zero_score),
        "stale_epochs": int(stale_epochs),
        "optimizer_step": int(optimizer_step),
        "samples_seen": int(samples_seen),
        "warmup_completed": bool(epoch >= config.vpr.classifier_warmup_epochs),
        "config_fingerprint": config.fingerprint(),
        "config": config.as_dict(),
        "data_fingerprints": data_fingerprints(config, generated=generated_manifest),
        "model": model.state_dict(),
        "classifier": classifier.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "rng": rng_state(),
        "teacher_cache_fingerprint": teacher_cache_fingerprint,
        "preservation_teacher_fingerprint": preservation_teacher_fingerprint,
    }


def _evaluate_validation(model, config, transform, device, output_path):
    return evaluate_model(
        model,
        config.data.robotcar_manifest,
        "val",
        transform,
        config.evaluation,
        device,
        batch_size=max(1, config.vpr.batch_size * 2),
        workers=config.vpr.workers,
        output_path=output_path,
    )


def train_vpr(config: ExperimentConfig, *, smoke_epochs: int | None = None) -> Path:
    cfg = config.vpr
    _set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.as_dict(), output_dir / "resolved_config.json")
    tracker = TrainingTracker(
        config,
        job_type="train-vpr",
        default_name=(
            f"vpr-{cfg.backbone}-{cfg.aggregator}-{cfg.image_height}x{cfg.image_width}"
            f"-{cfg.loss_profile}-seed{cfg.seed}"
        ),
        output_dir=output_dir,
    )
    transform = VPRTransform(cfg.image_height, cfg.image_width)
    generated_manifest = Path(config.data.generated_root) / "manifest.csv"
    teacher_cache = (
        TeacherDescriptorCache(config, generated_manifest)
        if cfg.loss_profile == "v3" and cfg.relation_teacher in {"salad", "boq"}
        else None
    )
    dataset = PairedVPRDataset(
        generated_manifest,
        transform,
        min_class_images=cfg.min_class_images,
        min_class_images_by_dataset=cfg.min_class_images_by_dataset,
        dataset_sampling_weights=cfg.dataset_sampling_weights,
        datasets=cfg.datasets,
        teacher_cache=teacher_cache,
    )
    if cfg.sampler == "rolling_balanced":
        sampler = RollingBalancedSampler(dataset.rows, cfg.dataset_epoch_quotas, cfg.seed)
    elif cfg.sampler == "coverage_balanced":
        sampler = CoverageBalancedSampler(dataset.rows, cfg.dataset_epoch_quotas, cfg.seed)
    else:
        sampler = WeightedRandomSampler(
            dataset.balanced_weights(), cfg.samples_per_epoch, replacement=True,
            generator=torch.Generator().manual_seed(cfg.seed),
        )

    initial_checkpoint = resolve_initial_checkpoint(cfg.init_checkpoint, cfg.descriptor_dim)
    model = build_descriptor_model(
        cfg.backbone,
        cfg.aggregator,
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        descriptor_dim=cfg.descriptor_dim,
        pretrained=not bool(initial_checkpoint),
        init_checkpoint=initial_checkpoint,
        init_checkpoint_scope=cfg.init_checkpoint_scope,
        model_family=cfg.model_family,
    ).to(device)
    needs_initial_teacher = (
        cfg.preservation_teacher == "initialization"
        or cfg.loss_profile == "v1"
        or cfg.relation_teacher == "self"
        or cfg.cross_domain_aux == "ikt"
    )
    teacher = (
        copy.deepcopy(model).eval().requires_grad_(False)
        if needs_initial_teacher else None
    )
    preservation_teacher_fingerprint = (
        file_sha256(initial_checkpoint)
        if cfg.preservation_teacher == "initialization" and initial_checkpoint
        else config.fingerprint() if cfg.preservation_teacher == "initialization" else ""
    )
    classifier_device = torch.device("cpu") if cfg.classifier_cpu_offload else device
    classifier = MarginCosineClassifier(
        cfg.descriptor_dim, len(dataset.place_to_label)
    ).to(classifier_device)
    optimizer = build_vpr_optimizer(model, classifier, cfg)

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "sampler": sampler,
        "num_workers": cfg.workers,
        "pin_memory": device.type == "cuda",
        "drop_last": True,
    }
    if cfg.workers > 0:
        loader_kwargs.update(
            multiprocessing_context="spawn",
            persistent_workers=True,
        )
    loader = DataLoader(dataset, **loader_kwargs)
    steps_per_epoch = max(1, len(loader) // cfg.gradient_accumulation_steps)
    scheduler = build_vpr_scheduler(optimizer, cfg, steps_per_epoch)
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch, best_score, epoch_zero_score = 0, -1.0, -1.0
    stale_epochs, saved_optimizer_step, samples_seen = 0, 0, 0
    regression_aborted = False

    if cfg.resume:
        state = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        if state["config_fingerprint"] != config.fingerprint():
            raise ValueError("VPR resume config does not match the current config")
        if state.get("preservation_teacher_fingerprint", "") != preservation_teacher_fingerprint:
            raise ValueError("VPR resume preservation teacher does not match")
        model.load_state_dict(state["model"])
        classifier.load_state_dict(state["classifier"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_rng_state(state["rng"])
        start_epoch = int(state["epoch"])
        best_score = float(state["best_val_recall1"])
        epoch_zero_score = float(state.get("epoch_zero_recall1", -1.0))
        stale_epochs = int(state.get("stale_epochs", 0))
        saved_optimizer_step = int(state.get("optimizer_step", 0))
        samples_seen = int(state.get("samples_seen", start_epoch * cfg.samples_per_epoch))
        cache_fingerprint = teacher_cache.fingerprint if teacher_cache is not None else ""
        if state.get("teacher_cache_fingerprint", "") != cache_fingerprint:
            raise ValueError("VPR resume teacher cache does not match the current cache")

    epochs = int(smoke_epochs or cfg.epochs)
    best_path = output_dir / "model_best.pt"
    global_step = saved_optimizer_step
    print(
        "VPR training budget: "
        f"rows={len(dataset):,}, samples/epoch={cfg.samples_per_epoch:,}, "
        f"batches/epoch={len(loader):,}, optimizer_steps/epoch={steps_per_epoch:,}, "
        f"epochs={epochs}, total_samples={cfg.samples_per_epoch * epochs:,}, "
        f"effective_batch={cfg.batch_size * cfg.gradient_accumulation_steps}"
    )

    if cfg.evaluate_epoch_zero and not cfg.resume:
        model.eval()
        metrics = _evaluate_validation(
            model, config, transform, device,
            output_dir / "validation" / "epoch_000.json",
        )
        epoch_zero_score = float(metrics["recalls"]["1"])
        best_score = epoch_zero_score
        atomic_torch_save(
            _inference_checkpoint(
                config, 0, best_score, epoch_zero_score, model,
                generated_manifest, preservation_teacher_fingerprint,
            ),
            best_path,
        )
        tracker.log(
            {
                "validation/epoch_zero/recall_at_1": epoch_zero_score,
                "validation/epoch_zero/recall_at_5": float(metrics["recalls"].get("5", 0.0)),
                "validation/epoch_zero/recall_at_10": float(metrics["recalls"].get("10", 0.0)),
                "validation/epoch_zero/mrr": float(metrics["mrr"]),
                "validation/epoch_zero/valid_queries": int(metrics["valid_queries"]),
                "validation/best_recall_at_1": best_score,
            },
            step=0,
        )
        tracker.log_checkpoint(
            best_path,
            name=f"vpr-{cfg.backbone}-{cfg.aggregator}-seed{cfg.seed}",
            aliases=["best", "epoch-0"],
        )

    completed_epochs = start_epoch
    for epoch in range(start_epoch, epochs):
        if isinstance(sampler, (CoverageBalancedSampler, RollingBalancedSampler)):
            sampler.set_epoch(epoch)
        model.train()
        warmup = configure_training_phase(model, optimizer, cfg, epoch)
        if cfg.freeze_batch_norm:
            freeze_batch_norm_statistics(model)
        if teacher is not None:
            teacher.eval()

        running = 0.0
        epoch_samples = 0
        epoch_optimizer_steps = 0
        data_time = 0.0
        step_time = 0.0
        epoch_start = time.perf_counter()
        data_wait_start = epoch_start
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(loader, desc=f"vpr/{cfg.backbone}/{cfg.aggregator} epoch {epoch + 1}")
        logged_first_update = False
        for batch_index, batch in enumerate(progress):
            batch_ready = time.perf_counter()
            data_time += batch_ready - data_wait_start
            step_start = batch_ready
            day = batch["day"].to(device, non_blocking=True)
            fake = batch["fake"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                explicit_domains = cfg.train_domains != "legacy"
                need_fake = not explicit_domains or cfg.train_domains in {"fake", "day_fake"}
                need_day = (
                    (not explicit_domains and cfg.loss_profile in {"v2", "v3"})
                    or (explicit_domains and cfg.train_domains in {"day", "day_fake"})
                    or cfg.cross_domain_aux in {"pair", "preserve_day"}
                )
                fake_descriptor = model.encode(fake) if need_fake else None
                day_descriptor = model.encode(day) if need_day else None
                if "teacher_descriptor" in batch:
                    teacher_day = batch["teacher_descriptor"].to(device, non_blocking=True)
                elif teacher is not None:
                    with torch.no_grad():
                        teacher_day = teacher.encode(day)
                else:
                    teacher_day = None
                if explicit_domains:
                    loss, components = configurable_vpr_loss(
                        cfg.train_domains,
                        cfg.cross_domain_aux,
                        day_descriptor,
                        fake_descriptor,
                        labels,
                        classifier,
                        teacher_day=teacher_day,
                        pair_weight=cfg.pair_weight,
                        pair_target=cfg.pair_target,
                        relation_weight=cfg.relation_weight,
                        preservation_weight=cfg.preservation_weight,
                        preservation_target=cfg.preservation_target,
                    )
                else:
                    legacy_day = day_descriptor if day_descriptor is not None else fake_descriptor.detach()
                    loss, components = vpr_training_loss(
                        cfg.loss_profile,
                        legacy_day,
                        fake_descriptor,
                        labels,
                        classifier,
                        teacher_day=teacher_day,
                        pair_weight=cfg.pair_weight,
                        pair_target=cfg.pair_target,
                        relation_weight=cfg.relation_weight,
                    )
            loss_divisor = accumulation_divisor(
                batch_index, len(loader), cfg.gradient_accumulation_steps
            )
            scaler.scale(loss / loss_divisor).backward()
            update_due = (
                (batch_index + 1) % cfg.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            optimizer_stepped = False
            if update_due:
                scale_before_update = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= scale_before_update
                if optimizer_stepped:
                    global_step += 1
                    epoch_optimizer_steps += 1
                    if scheduler is not None and cfg.scheduler == "cosine" and not warmup:
                        scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            batch_samples = int(day.shape[0])
            epoch_samples += batch_samples
            samples_seen += batch_samples
            running += float(loss.item())
            step_time += time.perf_counter() - step_start
            elapsed = max(time.perf_counter() - epoch_start, 1e-9)
            image_multiplier = 3 if teacher is not None else 2
            progress.set_postfix(
                loss=f"{loss.item():.3f}",
                samples=f"{epoch_samples}/{cfg.samples_per_epoch}",
            )
            if optimizer_stepped and (
                not logged_first_update or global_step % config.tracking.log_every_steps == 0
            ):
                log_values = {
                    "train/total": float(loss.item()),
                    "train/lr_backbone": float(optimizer.param_groups[0]["lr"]),
                    "train/lr_aggregator": float(optimizer.param_groups[1]["lr"]),
                    "train/lr_classifier": float(optimizer.param_groups[2]["lr"]),
                    "train/epoch": epoch + (batch_index + 1) / max(1, len(loader)),
                    "train/samples_epoch": epoch_samples,
                    "train/samples_total": samples_seen,
                    "train/batches_epoch": len(loader),
                    "train/optimizer_steps_epoch": epoch_optimizer_steps,
                    "train/samples_per_second": epoch_samples / elapsed,
                    "train/images_per_second": epoch_samples * image_multiplier / elapsed,
                    "train/data_time_ms": data_time * 1000 / max(1, batch_index + 1),
                    "train/step_time_ms": step_time * 1000 / max(1, batch_index + 1),
                    "train/bn_frozen": int(cfg.freeze_batch_norm),
                    "train/warmup_active": int(warmup),
                }
                log_values.update(
                    {f"train/{name}": float(value.item()) for name, value in components.items()}
                )
                tracker.log(log_values, step=global_step)
            if optimizer_stepped and (
                not logged_first_update or global_step % config.tracking.image_every_steps == 0
            ):
                tracker.log_vpr_images(day, fake, step=global_step)
            if optimizer_stepped:
                logged_first_update = True
            data_wait_start = time.perf_counter()

        model.eval()
        metrics = _evaluate_validation(
            model, config, transform, device,
            output_dir / "validation" / f"epoch_{epoch + 1:03d}.json",
        )
        score = float(metrics["recalls"]["1"])
        previous_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if scheduler is not None and cfg.scheduler == "plateau":
            scheduler.step(score)
        current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        lr_reduced = any(after < before for before, after in zip(previous_lrs, current_lrs))
        previous_best = best_score
        improved = score > previous_best
        significant_improvement = score > previous_best + cfg.scheduler_threshold
        if improved:
            best_score = score
        stale_epochs = 0 if significant_improvement else stale_epochs + 1
        at_min_lr = learning_rates_at_minimum(optimizer, cfg)
        epoch_elapsed = max(time.perf_counter() - epoch_start, 1e-9)
        epoch_log = {
            "validation/robotcar_project/recall_at_1": score,
            "validation/robotcar_project/recall_at_5": float(metrics["recalls"].get("5", 0.0)),
            "validation/robotcar_project/recall_at_10": float(metrics["recalls"].get("10", 0.0)),
            "validation/robotcar_project/mrr": float(metrics["mrr"]),
            "validation/robotcar_project/valid_queries": int(metrics["valid_queries"]),
            "validation/best_recall_at_1": best_score,
            "train/epoch_loss_mean": running / max(1, len(loader)),
            "train/lr_backbone": current_lrs[0],
            "train/lr_aggregator": current_lrs[1],
            "train/lr_classifier": current_lrs[2],
            "train/lr_reduced": int(lr_reduced),
            "train/stale_epochs": stale_epochs,
            "train/lr_at_minimum": int(at_min_lr),
            "train/samples_epoch": epoch_samples,
            "train/samples_total": samples_seen,
            "train/batches_epoch": len(loader),
            "train/optimizer_steps_epoch": epoch_optimizer_steps,
            "train/samples_per_second": epoch_samples / epoch_elapsed,
            "train/images_per_second": epoch_samples * (3 if teacher is not None else 2) / epoch_elapsed,
            "train/data_time_ms": data_time * 1000 / max(1, len(loader)),
            "train/step_time_ms": step_time * 1000 / max(1, len(loader)),
            "train/bn_frozen": int(cfg.freeze_batch_norm),
            "train/warmup_active": int(warmup),
            "epoch": epoch + 1,
        }
        if isinstance(sampler, (CoverageBalancedSampler, RollingBalancedSampler)):
            for dataset_name, values in sampler.statistics().items():
                for name, value in values.items():
                    epoch_log[f"coverage/{dataset_name}/{name}"] = value
        tracker.log(epoch_log, step=global_step)

        state = _vpr_checkpoint(
            config,
            epoch + 1,
            best_score,
            epoch_zero_score,
            stale_epochs,
            model,
            classifier,
            optimizer,
            scheduler,
            scaler,
            generated_manifest,
            teacher_cache.fingerprint if teacher_cache is not None else "",
            preservation_teacher_fingerprint,
            global_step,
            samples_seen,
        )
        atomic_torch_save(state, output_dir / "checkpoints" / "latest.pt")
        if improved:
            atomic_torch_save(
                _inference_checkpoint(
                    config, epoch + 1, score, epoch_zero_score, model,
                    generated_manifest, preservation_teacher_fingerprint,
                ),
                best_path,
            )
            tracker.log_checkpoint(
                best_path,
                name=f"vpr-{cfg.backbone}-{cfg.aggregator}-seed{cfg.seed}",
                aliases=["best", f"epoch-{epoch + 1}"],
            )

        completed_epochs = epoch + 1
        if (
            epoch == 0
            and epoch_zero_score >= 0
            and cfg.epoch_one_max_recall_drop > 0
            and score < epoch_zero_score - cfg.epoch_one_max_recall_drop
        ):
            regression_aborted = True
            break
        if cfg.early_stopping and stale_epochs >= cfg.patience and at_min_lr:
            break

    early_stopped = (
        cfg.early_stopping
        and stale_epochs >= cfg.patience
        and learning_rates_at_minimum(optimizer, cfg)
    )
    classifier_parameters = sum(parameter.numel() for parameter in classifier.parameters())
    if regression_aborted:
        convergence_status = "epoch_one_regression_aborted"
    elif completed_epochs >= epochs:
        convergence_status = "completed_fixed_budget"
    elif early_stopped:
        convergence_status = "early_stopped_converged"
    else:
        convergence_status = "incomplete"
    tracker.summary(
        {
            "epoch_zero_recall_at_1": epoch_zero_score,
            "best_val_recall_at_1": best_score,
            "completed_epochs": completed_epochs,
            "early_stopped": early_stopped,
            "regression_aborted": regression_aborted,
            "convergence_status": convergence_status,
            "best_checkpoint": str(best_path.resolve()),
            "training_rows": len(dataset),
            "samples_per_epoch": cfg.samples_per_epoch,
            "planned_total_samples": cfg.samples_per_epoch * epochs,
            "samples_seen": samples_seen,
            "place_classes": len(dataset.place_to_label),
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "classifier_parameters": classifier_parameters,
            "classifier_parameter_mib_fp32": classifier_parameters * 4 / 1024**2,
            "classifier_adam_state_mib_fp32": classifier_parameters * 8 / 1024**2,
            "classifier_cpu_offload": cfg.classifier_cpu_offload,
            "classifier_optimizer_state_device": str(classifier_device),
            "peak_training_cuda_mib": (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda" else None
            ),
            "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "sampler": cfg.sampler,
            "coverage_statistics": (
                sampler.statistics()
                if isinstance(sampler, (CoverageBalancedSampler, RollingBalancedSampler)) else {}
            ),
            "effective_batch_size": cfg.batch_size * cfg.gradient_accumulation_steps,
            "scheduler": cfg.scheduler,
            "freeze_batch_norm": cfg.freeze_batch_norm,
            "classifier_warmup_epochs": cfg.classifier_warmup_epochs,
            "preservation_teacher": cfg.preservation_teacher,
            "preservation_teacher_fingerprint": preservation_teacher_fingerprint,
        }
    )
    tracker.finish(exit_code=1 if regression_aborted else 0)
    return best_path
