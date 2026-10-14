from __future__ import annotations

import math
import json
import random
import resource
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from ..checkpointing import (
    SCHEMA_VERSION,
    atomic_torch_save,
    checkpoint_config_matches,
    data_fingerprints,
    file_sha256,
    restore_rng_state,
    rng_state,
    write_json,
)
from ..config import ExperimentConfig
from ..data import (
    CoverageBalancedSampler,
    PKBalancedBatchSampler,
    PairedVPRDataset,
    RollingBalancedSampler,
    StableNightVPRDataset,
)
from ..transforms import VPRTransform
from ..training_logging import create_summary_writer, log_patch_diagnostics, log_scalars
from .losses import (
    MarginCosineClassifier,
    cross_domain_neighborhood_loss,
    day_classification_loss,
    day_fake_classification_loss,
)
from .model import build_descriptor_model, resolve_initial_checkpoint
from .neighborhood import DatasetNeighborhoodQueue
from .teacher_cache import TeacherDescriptorCache
from .salad import DINOv2SALADBackbone
from .stable_night import stable_night_batch_loss


def _set_random_state(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def optimizer_base_lrs(cfg) -> list[float]:
    return [cfg.encoder_lr, cfg.aggregator_lr, cfg.classifier_lr]


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
    if hasattr(model, "set_backbone_trainable"):
        model.set_backbone_trainable(not warmup)
    else:
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
            {"params": model.backbone.parameters(), "lr": cfg.encoder_lr},
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
):
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "vpr_inference",
        "epoch": int(epoch),
        "val_recall1": float(score),
        "epoch_zero_recall1": float(epoch_zero_score),
        "selection_rule": config.vpr.selection_rule,
        "config_fingerprint": config.fingerprint(),
        "descriptor_dim": int(config.vpr.descriptor_dim),
        "random_seed": int(config.vpr.random_seed),
        "data_fingerprints": data_fingerprints(config, generated=generated_manifest),
        "model": model.state_dict(),
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
    optimizer_step,
    samples_seen,
    sampler_state=None,
    relation_queue_state=None,
):
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "vpr_training_state",
        "epoch": int(epoch),
        "best_val_recall1": float(best_score),
        "epoch_zero_recall1": float(epoch_zero_score),
        "selection_rule": config.vpr.selection_rule,
        "stale_epochs": int(stale_epochs),
        "optimizer_step": int(optimizer_step),
        "samples_seen": int(samples_seen),
        "warmup_completed": bool(epoch >= config.vpr.classifier_warmup_epochs),
        "config_fingerprint": config.fingerprint(),
        "config": config.as_dict(),
        "descriptor_dim": int(config.vpr.descriptor_dim),
        "random_seed": int(config.vpr.random_seed),
        "data_fingerprints": data_fingerprints(config, generated=generated_manifest),
        "model": model.state_dict(),
        "classifier": classifier.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "rng": rng_state(),
        "teacher_cache_fingerprint": teacher_cache_fingerprint,
        "sampler_state": sampler_state,
        "relation_queue_state": relation_queue_state,
    }


def _evaluate_validation(model, config, transform, device, output_path):
    from ..references.models import evaluate_dual_encoder, suite_rows

    results = {}
    for dataset in config.evaluation.datasets:
        if dataset not in {"msls_day_val", "svox_day_val"}:
            raise ValueError("Checkpoint validation requires public day validation datasets")
        database, queries = suite_rows(config, dataset, "val")
        if not database or not queries:
            raise ValueError(f"Incomplete public validation split: {dataset}")
        metrics = evaluate_dual_encoder(
            model, model, database, queries, transform, 25.0, None,
            config.evaluation.recalls, device,
            batch_size=max(1, config.vpr.batch_size * 2), workers=config.vpr.workers,
        )
        if metrics["valid_queries"] == 0:
            raise ValueError(f"No ground-truth positives in public validation split: {dataset}")
        metrics.update(database=len(database), queries=len(queries),
                       positive_radius_m=25.0, positive_yaw_deg=None, split="val")
        results[dataset] = metrics
    # Give each configured validation benchmark the same checkpoint-selection
    # weight, so a larger set does not silently dominate a smaller one.
    summary = {
        "recalls": {str(k): float(np.mean([m["recalls"][str(k)] for m in results.values()]))
                    for k in config.evaluation.recalls},
        "mrr": float(np.mean([m["mrr"] for m in results.values()])),
        "valid_queries": sum(m["valid_queries"] for m in results.values()),
        "total_queries": sum(m["total_queries"] for m in results.values()),
        "datasets": results,
        "split": "val", "database_domain": "day", "positive_radius_m": 25.0,
        "positive_yaw_deg": None,
    }
    if output_path:
        write_json(summary, output_path)
    return summary


def export_vpr_inference_checkpoint(
    config: ExperimentConfig,
    training_checkpoint: str | Path,
    output: str | Path,
) -> Path:
    """Strip all training-only state while preserving model tensors exactly."""
    training_checkpoint = Path(training_checkpoint).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite inference checkpoint: {output}")
    state = torch.load(training_checkpoint, map_location="cpu", weights_only=False)
    if state.get("kind") != "vpr_training_state":
        raise ValueError("Expected a VPR training-state checkpoint")
    if not checkpoint_config_matches(state, config, ("data", "vpr", "evaluation")):
        raise ValueError("Training checkpoint config does not match the current config")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "vpr_inference",
        "epoch": int(state["epoch"]),
        "val_recall1": float(state.get("best_val_recall1", -1.0)),
        "epoch_zero_recall1": float(state.get("epoch_zero_recall1", -1.0)),
        "config_fingerprint": state["config_fingerprint"],
        "descriptor_dim": int(config.vpr.descriptor_dim),
        "random_seed": int(config.vpr.random_seed),
        "data_fingerprints": state.get("data_fingerprints", {}),
        "model": state["model"],
    }
    atomic_torch_save(payload, output)
    return output


def train_vpr(config: ExperimentConfig, *, smoke_epochs: int | None = None) -> Path:
    with ExitStack() as logging_stack:
        return _train_vpr(config, smoke_epochs=smoke_epochs, logging_stack=logging_stack)


def _train_vpr(config: ExperimentConfig, *, smoke_epochs: int | None, logging_stack: ExitStack) -> Path:
    cfg = config.vpr
    stable_night = cfg.model_family == "stable_night_salad"
    _set_random_state(cfg.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.as_dict(), output_dir / "resolved_config.json")
    transform = VPRTransform(cfg.image_height, cfg.image_width)
    generated_manifest = Path(config.data.generated_root) / "manifest.csv"
    training_manifest = generated_manifest
    if stable_night and cfg.train_domains in {"day_only", "day_aug"}:
        training_manifest = Path(config.data.vpr_source_manifest)
        generated_manifest = None
    teacher_cache = (
        TeacherDescriptorCache(config, generated_manifest)
        if cfg.cross_domain_aux == "cross_domain_relation"
        and cfg.relation_teacher in {"salad", "boq"}
        else None
    )
    dataset_kwargs = dict(
        min_class_images=cfg.min_class_images,
        min_class_images_by_dataset=cfg.min_class_images_by_dataset,
        dataset_sampling_weights=cfg.dataset_sampling_weights,
        datasets=cfg.datasets,
    )
    if stable_night:
        dataset = StableNightVPRDataset(
            training_manifest, transform, train_domains=cfg.train_domains,
            random_seed=cfg.random_seed, **dataset_kwargs
        )
    else:
        dataset = PairedVPRDataset(
            training_manifest, transform, teacher_cache=teacher_cache,
            include_fake=cfg.train_domains == "day_fake", **dataset_kwargs
        )
    if cfg.sampler == "pk_balanced":
        sampler = PKBalancedBatchSampler(
            dataset.rows,
            cfg.dataset_epoch_quotas,
            cfg.pk_places_per_batch,
            cfg.pk_images_per_place,
            cfg.random_seed,
        )
    elif cfg.sampler == "rolling_balanced":
        sampler = RollingBalancedSampler(
            dataset.rows, cfg.dataset_epoch_quotas, cfg.random_seed
        )
    elif cfg.sampler == "coverage_balanced":
        sampler = CoverageBalancedSampler(
            dataset.rows, cfg.dataset_epoch_quotas, cfg.random_seed
        )
    else:
        sampler = WeightedRandomSampler(
            dataset.balanced_weights(), cfg.samples_per_epoch, replacement=True,
            generator=torch.Generator().manual_seed(cfg.random_seed),
        )

    initial_checkpoint = resolve_initial_checkpoint(cfg.init_checkpoint, cfg.descriptor_dim)
    stable_model_kwargs = {
        "stable_dustbin_enabled": cfg.stable_dustbin_enabled,
        "stable_dustbin_gamma_init": cfg.stable_dustbin_gamma_init,
        "stable_dustbin_gamma_max": cfg.stable_dustbin_gamma_max,
        "stable_sinkhorn_iterations": cfg.stable_sinkhorn_iterations,
    } if stable_night else {}
    model = build_descriptor_model(
        cfg.aggregator,
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        descriptor_dim=cfg.descriptor_dim,
        pretrained=not bool(initial_checkpoint),
        init_checkpoint=initial_checkpoint,
        model_family=cfg.model_family,
        num_clusters=cfg.num_clusters,
        backbone_name=cfg.backbone_name,
        backbone_repo=cfg.backbone_repo,
        backbone_checkpoint=cfg.backbone_checkpoint,
        backbone_trainable_blocks=cfg.backbone_trainable_blocks,
        aggregator_repo=cfg.aggregator_repo,
        **stable_model_kwargs,
    ).to(device)
    classifier_device = torch.device("cpu") if cfg.classifier_cpu_offload else device
    classifier = (
        nn.Identity() if stable_night else MarginCosineClassifier(
            cfg.descriptor_dim, len(dataset.place_to_label)
        )
    ).to(classifier_device)
    optimizer = build_vpr_optimizer(model, classifier, cfg)
    reference_backbone = None
    if stable_night and (
        cfg.stable_stability_weight > 0
        or (cfg.stable_reliability_weight > 0 and cfg.stable_reliability_target == "augmentation")
    ):
        reference_backbone = DINOv2SALADBackbone(
            cfg.backbone_repo, cfg.backbone_name, cfg.backbone_checkpoint, trainable_blocks=0
        ).eval().to(device)
        reference_backbone.requires_grad_(False)
    training_reference_fingerprint = teacher_cache.fingerprint if teacher_cache is not None else ""
    if reference_backbone is not None:
        training_reference_fingerprint = "stable_reference:" + file_sha256(cfg.backbone_checkpoint)
    metrics_path = output_dir / "training_metrics.jsonl"
    if stable_night and not cfg.resume:
        metrics_path.write_text("", encoding="utf-8")

    loader_kwargs = {
        "num_workers": cfg.workers,
        "pin_memory": device.type == "cuda",
        "generator": torch.Generator().manual_seed(cfg.random_seed),
    }
    if cfg.sampler == "pk_balanced":
        loader_kwargs["batch_sampler"] = sampler
    else:
        loader_kwargs.update(batch_size=cfg.batch_size, sampler=sampler, drop_last=True)
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
    queue_partitions = cfg.datasets if cfg.relation_queue_scope == "dataset" else ("shared",)
    relation_queue = (
        DatasetNeighborhoodQueue(queue_partitions, cfg.relation_queue_size, device)
        if cfg.cross_domain_aux == "cross_domain_relation"
        else None
    )
    start_epoch, best_score, epoch_zero_score = 0, -1.0, -1.0
    stale_epochs, saved_optimizer_step, samples_seen = 0, 0, 0
    regression_aborted = False

    if cfg.resume:
        state = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        if not checkpoint_config_matches(state, config, ("data", "vpr", "evaluation")):
            raise ValueError("VPR resume config does not match the current config")
        if state.get("data_fingerprints") != data_fingerprints(config, generated=generated_manifest):
            raise ValueError("VPR resume training or public validation resources do not match the checkpoint")
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
        if state.get("teacher_cache_fingerprint", "") != training_reference_fingerprint:
            if stable_night:
                raise ValueError("VPR resume frozen reference does not match the current reference")
            raise ValueError("VPR resume teacher cache does not match the current cache")
        if state.get("sampler_state") is not None:
            if not hasattr(sampler, "load_state_dict"):
                raise ValueError("Checkpoint contains sampler state for a non-resumable sampler")
            sampler.load_state_dict(state["sampler_state"])
        if relation_queue is not None:
            if state.get("relation_queue_state") is None:
                raise ValueError("CDND resume checkpoint is missing relation queue state")
            relation_queue.load_state_dict(state["relation_queue_state"])

    epochs = int(smoke_epochs or cfg.epochs)
    best_path = output_dir / "model_best.pt"
    global_step = saved_optimizer_step
    writer = logging_stack.enter_context(create_summary_writer(
        cfg, output_dir, purge_step=saved_optimizer_step + 1 if cfg.resume else None,
    ))
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
            output_dir / "validation" / "latest.json",
        )
        epoch_zero_score = float(metrics["recalls"]["1"])
        best_score = epoch_zero_score
        _log_validation(writer, metrics, global_step, best_score)
        atomic_torch_save(
            _inference_checkpoint(
                config, 0, best_score, epoch_zero_score, model, generated_manifest,
            ),
            best_path,
        )

    completed_epochs = start_epoch
    for epoch in range(start_epoch, epochs):
        if stable_night and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        if isinstance(sampler, (CoverageBalancedSampler, RollingBalancedSampler, PKBalancedBatchSampler)):
            sampler.set_epoch(epoch)
        model.train()
        warmup = configure_training_phase(model, optimizer, cfg, epoch)
        if cfg.freeze_batch_norm:
            freeze_batch_norm_statistics(model)

        running = 0.0
        epoch_samples = 0
        epoch_optimizer_steps = 0
        encoded_student_images = 0
        teacher_images = 0
        data_time = 0.0
        step_time = 0.0
        epoch_start = time.perf_counter()
        data_wait_start = epoch_start
        optimizer.zero_grad(set_to_none=True)
        component_totals: dict[str, float] = {}
        group_totals: dict[str, float] = {}
        group_batches = 0
        epoch_skipped_updates = 0
        progress = tqdm(loader, desc=f"vpr/{cfg.model_family}/{cfg.aggregator} epoch {epoch + 1}")
        for batch_index, batch in enumerate(progress):
            batch_ready = time.perf_counter()
            data_time += batch_ready - data_wait_start
            step_start = batch_ready
            next_step = global_step + 1
            capture_patches = (
                writer is not None and stable_night and cfg.tensorboard_image_interval > 0
                and (next_step == saved_optimizer_step + 1
                     or next_step % cfg.tensorboard_image_interval == 0
                     or batch_index + 1 == len(loader))
            )
            diagnostics = {} if capture_patches else None
            day = batch["day"].to(device, non_blocking=True)
            fake = (
                batch["fake"].to(device, non_blocking=True)
                if cfg.train_domains in {"day_fake", "day_aug_fake"}
                else None
            )
            labels = batch["label"].to(device, non_blocking=True)
            image_keys = batch["image_key"].to(device, non_blocking=True)
            batch_datasets = set(batch["dataset"])
            dataset_name = next(iter(batch_datasets))
            if cfg.cross_domain_aux == "cross_domain_relation" and len(batch_datasets) != 1:
                raise RuntimeError("CDND requires every mini-batch to contain one dataset")
            with torch.autocast(device.type, enabled=use_amp):
                if stable_night:
                    view_images = {"day": day}
                    view_masks = {"day": batch["day_valid_mask"].to(device, non_blocking=True)}
                    if "aug" in batch:
                        view_images["aug"] = batch["aug"].to(device, non_blocking=True)
                        view_masks["aug"] = view_masks["day"]
                    if fake is not None:
                        view_images["fake"] = fake
                        view_masks["fake"] = batch["fake_valid_mask"].to(device, non_blocking=True)
                    loss, components, reference_images = stable_night_batch_loss(
                        model, view_images, view_masks, labels, image_keys, cfg, reference_backbone,
                        diagnostics=diagnostics,
                    )
                    teacher_images += reference_images
                else:
                    day_descriptor = model.encode(day)
                    fake_descriptor = model.encode(fake) if fake is not None else None
                    if "teacher_descriptor" in batch:
                        teacher_day = batch["teacher_descriptor"].to(device, non_blocking=True)
                    else:
                        teacher_day = None
                    if fake_descriptor is None:
                        loss, components = day_classification_loss(
                            day_descriptor, labels, classifier
                        )
                    else:
                        loss, components = day_fake_classification_loss(
                            day_descriptor, fake_descriptor, labels, classifier
                        )
                if cfg.cross_domain_aux == "cross_domain_relation":
                    if teacher_day is None or day_descriptor is None or fake_descriptor is None:
                        raise RuntimeError("CDND requires teacher, day, and fake descriptors")
                    queue_partition = (
                        dataset_name if cfg.relation_queue_scope == "dataset" else "shared"
                    )
                    candidates = relation_queue.candidates(
                        queue_partition,
                        teacher_day,
                        day_descriptor,
                        labels,
                        image_keys,
                    )
                    with torch.autocast(device.type, enabled=False):
                        fake_relation, day_relation, relation_stats = (
                            cross_domain_neighborhood_loss(
                                teacher_day,
                                candidates.teacher,
                                fake_descriptor,
                                day_descriptor,
                                candidates.student_day,
                                labels,
                                candidates.labels,
                                image_keys,
                                candidates.image_keys,
                                relation_temperature=cfg.relation_temperature,
                                mask_self_matches=cfg.relation_mask_self,
                            )
                        )
                    relation = fake_relation + cfg.relation_day_weight * day_relation
                    loss = loss + cfg.relation_weight * relation
                    components.update(
                        relation_fake_to_day=fake_relation,
                        relation_day_to_day=day_relation,
                        relation=relation,
                        relation_valid_anchors=relation_stats["valid_anchors"],
                    )
            loss_divisor = accumulation_divisor(
                batch_index, len(loader), cfg.gradient_accumulation_steps
            )
            scaler.scale(loss / loss_divisor).backward()
            if relation_queue is not None:
                relation_queue.enqueue(
                    queue_partition, teacher_day, day_descriptor, labels, image_keys
                )
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
                if not optimizer_stepped:
                    epoch_skipped_updates += 1
                if optimizer_stepped:
                    global_step += 1
                    epoch_optimizer_steps += 1
                    if scheduler is not None and cfg.scheduler == "cosine" and not warmup:
                        scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            batch_samples = int(day.shape[0])
            epoch_samples += batch_samples
            samples_seen += batch_samples
            encoded_student_images += batch_samples * (
                len(view_images) if stable_night else (2 if fake is not None else 1)
            )
            loss_value = float(loss.item())
            if stable_night or writer is not None:
                component_values = {name: float(value.detach().item()) for name, value in components.items()}
                for name, value in component_values.items():
                    component_totals[name] = component_totals.get(name, 0.0) + value
            if writer is not None:
                group_batches += 1
                batch_values = {**component_values, "loss": loss_value}
                for name, value in batch_values.items():
                    group_totals[name] = group_totals.get(name, 0.0) + value
                if update_due:
                    log_due = optimizer_stepped and (
                        global_step == saved_optimizer_step + 1
                        or global_step % cfg.tensorboard_log_interval == 0
                        or batch_index + 1 == len(loader)
                    )
                    if log_due:
                        log_scalars(writer, {name: value / group_batches for name, value in group_totals.items()}, global_step, "train")
                        log_scalars(writer, _optimizer_lrs(optimizer), global_step, "lr")
                        log_scalars(writer, {"samples_seen": samples_seen}, global_step, "progress")
                        log_scalars(writer, {"scale": scaler.get_scale()}, global_step, "amp")
                    if optimizer_stepped and diagnostics is not None:
                        log_patch_diagnostics(writer, diagnostics, global_step)
                    group_totals.clear()
                    group_batches = 0
            running += loss_value
            step_time += time.perf_counter() - step_start
            elapsed = max(time.perf_counter() - epoch_start, 1e-9)
            progress.set_postfix(
                loss=f"{loss_value:.3f}",
                samples=f"{epoch_samples}/{cfg.samples_per_epoch}",
            )
            data_wait_start = time.perf_counter()

        model.eval()
        if cfg.selection_rule == "fixed_budget_last":
            score = -1.0  # Not measured; never report this sentinel as recall.
        else:
            metrics = _evaluate_validation(
                model, config, transform, device,
                output_dir / "validation" / "latest.json",
            )
            score = float(metrics["recalls"]["1"])
        previous_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if scheduler is not None and cfg.scheduler == "plateau":
            scheduler.step(score)
        current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        lr_reduced = any(after < before for before, after in zip(previous_lrs, current_lrs))
        previous_best = best_score
        improved = cfg.selection_rule == "fixed_budget_last" or score > previous_best
        significant_improvement = score > previous_best + cfg.scheduler_threshold
        if improved:
            best_score = score
        if cfg.selection_rule != "fixed_budget_last":
            _log_validation(writer, metrics, global_step, best_score)
        stale_epochs = 0 if significant_improvement else stale_epochs + 1
        at_min_lr = learning_rates_at_minimum(optimizer, cfg)
        epoch_elapsed = max(time.perf_counter() - epoch_start, 1e-9)

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
            training_reference_fingerprint,
            global_step,
            samples_seen,
            sampler.state_dict() if hasattr(sampler, "state_dict") else None,
            relation_queue.state_dict() if relation_queue is not None else None,
        )
        atomic_torch_save(state, output_dir / "checkpoints" / "latest.pt")
        if improved:
            atomic_torch_save(
                _inference_checkpoint(
                    config, epoch + 1, score, epoch_zero_score, model, generated_manifest,
                ),
                best_path,
            )

        completed_epochs = epoch + 1
        if writer is not None:
            epoch_losses = {name: total / max(1, len(loader)) for name, total in component_totals.items()}
            epoch_losses["loss"] = running / max(1, len(loader))
            log_scalars(writer, epoch_losses, global_step, "epoch_loss")
            log_scalars(writer, {
                "epoch": completed_epochs, "samples": epoch_samples,
                "student_images": encoded_student_images, "reference_images": teacher_images,
                "optimizer_steps": epoch_optimizer_steps, "elapsed_seconds": epoch_elapsed,
                "data_wait_seconds": data_time, "batch_seconds": step_time,
                "samples_per_second": epoch_samples / epoch_elapsed,
                "warmup": int(warmup), "amp_skipped_updates": epoch_skipped_updates,
            }, global_step, "epoch_progress")
            log_scalars(writer, _optimizer_lrs(optimizer), global_step, "lr")
            log_scalars(writer, {"scale": scaler.get_scale()}, global_step, "amp")
            writer.flush()
        if stable_night:
            epoch_metrics = {
                "epoch": completed_epochs,
                "train_domains": cfg.train_domains,
                "components_mean": {
                    name: total / max(1, len(loader)) for name, total in component_totals.items()
                },
                "samples": epoch_samples,
                "student_images": encoded_student_images,
                "reference_images": teacher_images,
                "optimizer_steps": epoch_optimizer_steps,
                "learning_rates": current_lrs,
                "validation_recall1": None if cfg.selection_rule == "fixed_budget_last" else score,
                "elapsed_seconds": epoch_elapsed,
            }
            with metrics_path.open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")
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
    if regression_aborted:
        convergence_status = "epoch_one_regression_aborted"
    elif completed_epochs >= epochs:
        convergence_status = "completed_fixed_budget"
    elif early_stopped:
        convergence_status = "early_stopped_converged"
    else:
        convergence_status = "incomplete"
    return best_path


def _optimizer_lrs(optimizer) -> dict[str, float]:
    return {name: group["lr"] for name, group in zip(
        ("backbone", "aggregator", "classifier"), optimizer.param_groups
    )}


def _log_validation(writer, metrics: dict, step: int, best_score: float) -> None:
    values = {f"recall_at_{key}": value for key, value in metrics.get("recalls", {}).items()}
    values.update(mrr=metrics.get("mrr"), valid_queries=metrics.get("valid_queries"),
                  total_queries=metrics.get("total_queries"), best_recall_at_1=best_score)
    log_scalars(writer, values, step, "validation")
