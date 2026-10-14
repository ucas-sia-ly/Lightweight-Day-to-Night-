"""Training-only StableNight losses; reference features never enter inference state."""
from __future__ import annotations

import torch

from .losses import (
    frozen_patch_reliability_target,
    multi_positive_place_loss,
    patch_reliability_loss,
    patch_validity_mask,
    stable_local_consistency_loss,
)


def stable_night_batch_loss(
    model,
    images: dict[str, torch.Tensor],
    valid_masks: dict[str, torch.Tensor],
    labels: torch.Tensor,
    source_keys: torch.Tensor,
    cfg,
    reference_backbone=None,
    diagnostics: dict | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], int]:
    """Encode all configured views and apply independently weighted objectives.

    The only local correspondence accepted here is Day/Aug, whose augmentation
    preserves geometry.  Generated night views participate in global place
    training and the reliability floor, never in patch correspondence losses.
    """
    encoded = {name: model.encode_with_aux(value) for name, value in images.items()}
    day_descriptor = encoded["day"][0]
    zero = sum((descriptor.sum() * 0.0 for descriptor, _ in encoded.values()), day_descriptor.new_zeros(()))
    components: dict[str, torch.Tensor] = {}
    teacher_images = 0
    # Contrastive similarities, cosine targets and probability losses stay fp32
    # even when the encoders run under the trainer's mixed-precision context.
    with torch.autocast(day_descriptor.device.type, enabled=False):
        loss = zero
        if cfg.stable_place_weight > 0:
            descriptors = torch.cat([value[0] for value in encoded.values()], dim=0)
            place_loss, stats = multi_positive_place_loss(
                descriptors,
                labels.repeat(len(encoded)),
                source_keys.repeat(len(encoded)),
                temperature=cfg.stable_place_temperature,
            )
            if not bool(stats["place_valid_anchors"]):
                raise RuntimeError(
                    "StableNight place training needs distinct-source positives and negatives in every batch"
                )
            loss = loss + cfg.stable_place_weight * place_loss
            components.update(place=place_loss, **stats)

        needs_reference = cfg.stable_stability_weight > 0 or (
            cfg.stable_reliability_weight > 0 and cfg.stable_reliability_target == "augmentation"
        )
        target = None
        day_aux = encoded["day"][1]
        day_valid = patch_validity_mask(valid_masks["day"], day_aux["local_features"])
        if needs_reference:
            if "aug" not in images or reference_backbone is None:
                raise RuntimeError("StableNight local losses require aligned augmentation and a frozen reference")
            reference_backbone.eval()
            with torch.no_grad():
                teacher_day, _ = reference_backbone(images["day"].float())
                teacher_aug, _ = reference_backbone(images["aug"].float())
                target = frozen_patch_reliability_target(
                    teacher_day, teacher_aug, images["day"].float(), day_valid
                )
            teacher_images = 2 * images["day"].shape[0]
            components["reference_target_mean"] = (
                target.sum() / day_valid.sum().clamp_min(1)
            ).detach()
        if cfg.stable_stability_weight > 0:
            local_loss, stats = stable_local_consistency_loss(
                day_aux["local_features"], encoded["aug"][1]["local_features"], target, day_valid
            )
            loss = loss + cfg.stable_stability_weight * local_loss
            components.update(stable=local_loss, **stats)
        if cfg.stable_reliability_weight > 0:
            reliability_losses = []
            for name, (_, aux) in encoded.items():
                valid = patch_validity_mask(valid_masks[name], aux["local_features"])
                view_target = target if cfg.stable_reliability_target == "augmentation" and name in {"day", "aug"} else None
                reliability_loss, stats = patch_reliability_loss(
                    aux["reliability"], valid, view_target, floor=cfg.stable_reliability_floor
                )
                reliability_losses.append(reliability_loss)
                components.update({f"{name}_{key}": value for key, value in stats.items()})
            reliability_loss = torch.stack(reliability_losses).mean()
            loss = loss + cfg.stable_reliability_weight * reliability_loss
            components["reliability"] = reliability_loss
        for name, (_, aux) in encoded.items():
            valid = patch_validity_mask(valid_masks[name], aux["local_features"])
            components[f"{name}_predictor_valid_mean"] = (
                (aux["reliability"].detach().float() * valid).sum() / valid.sum().clamp_min(1)
            )
            components[f"{name}_dustbin_mean"] = aux["dustbin_assignment"].detach().float().mean()
            if "cluster_mass" in aux:
                mass = aux["cluster_mass"].detach().float()
                components[f"{name}_cluster_mass_min"] = mass.min()
                components[f"{name}_cluster_mass_max_error"] = (mass - 1.0).abs().max()
            if diagnostics is not None:
                height, width = aux["local_features"].shape[-2:]
                count = min(3, images[name].shape[0])
                diagnostics[name] = {
                    "images": images[name][:count].detach(),
                    "reliability": aux["reliability"][:count].detach().reshape(count, 1, height, width),
                    "dustbin_assignment": aux["dustbin_assignment"][:count].detach().reshape(count, 1, height, width),
                    "valid_mask": valid[:count].reshape(count, 1, height, width),
                }
                if target is not None and name == "day":
                    diagnostics["reference_target"] = target[:count].detach().reshape(count, 1, height, width)
        if hasattr(model.aggregator, "gamma"):
            components["dustbin_gamma"] = model.aggregator.gamma.detach().float()
        components["loss"] = loss.detach()
    return loss, components, teacher_images
