from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MarginCosineClassifier(nn.Module):
    def __init__(self, descriptor_dim: int, classes: int, scale: float = 30.0, margin: float = 0.4):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, descriptor_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)
        self.margin = float(margin)

    def forward(self, descriptors: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        output_device = descriptors.device
        if descriptors.device != self.weight.device:
            descriptors = descriptors.to(self.weight.device, dtype=self.weight.dtype)
            labels = labels.to(self.weight.device) if labels is not None else None
        logits = F.linear(F.normalize(descriptors, dim=1), F.normalize(self.weight, dim=1))
        if labels is not None:
            one_hot = F.one_hot(labels, num_classes=self.weight.shape[0]).to(logits.dtype)
            logits = logits - one_hot * self.margin
        return (logits * self.scale).to(output_device)


def ikt_loss(
    teacher_descriptors: torch.Tensor,
    fake_descriptors: torch.Tensor,
    classifier: MarginCosineClassifier,
    temperature: float = 4.0,
) -> torch.Tensor:
    """NocPlace reference loss; it is not used by the CDND training path."""
    teacher_logits = classifier(teacher_descriptors.detach()) / temperature
    student_logits = classifier(fake_descriptors) / temperature
    return F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.softmax(teacher_logits, dim=1),
        reduction="batchmean",
    ) * temperature**2


def day_fake_classification_loss(
    day: torch.Tensor,
    fake: torch.Tensor,
    labels: torch.Tensor,
    classifier: MarginCosineClassifier,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    day_loss = F.cross_entropy(classifier(day, labels), labels)
    fake_loss = F.cross_entropy(classifier(fake, labels), labels)
    return 0.5 * (day_loss + fake_loss), {
        "classification_day": day_loss,
        "classification_fake": fake_loss,
    }


def day_classification_loss(
    day: torch.Tensor,
    labels: torch.Tensor,
    classifier: MarginCosineClassifier,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    day_loss = F.cross_entropy(classifier(day, labels), labels)
    return day_loss, {"classification_day": day_loss}


def cross_domain_neighborhood_loss(
    teacher_anchors: torch.Tensor,
    teacher_candidates: torch.Tensor,
    fake_anchors: torch.Tensor,
    day_anchors: torch.Tensor,
    student_day_candidates: torch.Tensor,
    anchor_labels: torch.Tensor,
    candidate_labels: torch.Tensor,
    anchor_image_keys: torch.Tensor,
    candidate_image_keys: torch.Tensor,
    *,
    relation_temperature: float = 0.07,
    mask_self_matches: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Distill SALAD day neighborhoods without confidence gating."""
    if teacher_anchors.shape[0] != fake_anchors.shape[0] or fake_anchors.shape != day_anchors.shape:
        raise ValueError("CDND anchor tensors must share the same batch shape")
    if teacher_candidates.shape[0] != student_day_candidates.shape[0]:
        raise ValueError("Teacher and student candidate counts must match")
    if candidate_labels.shape[0] != teacher_candidates.shape[0]:
        raise ValueError("Candidate metadata does not match candidate descriptors")

    teacher_anchor = F.normalize(teacher_anchors.detach().float(), dim=1)
    teacher_candidate = F.normalize(teacher_candidates.detach().float(), dim=1)
    fake_anchor = F.normalize(fake_anchors.float(), dim=1)
    day_anchor = F.normalize(day_anchors.float(), dim=1)
    student_candidate = F.normalize(student_day_candidates.float(), dim=1)
    valid_candidates = (
        anchor_image_keys[:, None].ne(candidate_image_keys[None, :])
        if mask_self_matches
        else torch.ones(
            (anchor_image_keys.shape[0], candidate_image_keys.shape[0]),
            dtype=torch.bool,
            device=anchor_image_keys.device,
        )
    )
    same_place = anchor_labels[:, None].eq(candidate_labels[None, :]) & valid_candidates
    different_place = anchor_labels[:, None].ne(candidate_labels[None, :]) & valid_candidates
    valid_anchors = same_place.any(dim=1) & different_place.any(dim=1)
    if not bool(valid_anchors.any()):
        zero = (fake_anchors.sum() + day_anchors.sum() + student_day_candidates.sum()) * 0.0
        return zero, zero, {
            "valid_anchors": torch.zeros((), device=zero.device, dtype=torch.long)
        }

    valid_candidates = valid_candidates[valid_anchors]
    teacher_similarity = (teacher_anchor @ teacher_candidate.T)[valid_anchors]
    fake_anchor = fake_anchor[valid_anchors]
    day_anchor = day_anchor[valid_anchors]
    teacher_logits = (teacher_similarity / float(relation_temperature)).masked_fill(
        ~valid_candidates, -torch.inf
    )
    teacher_probability = F.softmax(teacher_logits, dim=1)
    teacher_log_probability = torch.log(teacher_probability.clamp_min(1e-12))

    def relation_kl(student_anchor: torch.Tensor) -> torch.Tensor:
        student_logits = (
            student_anchor @ student_candidate.T / float(relation_temperature)
        ).masked_fill(~valid_candidates, -torch.inf)
        student_log_probability = F.log_softmax(student_logits, dim=1)
        student_log_probability = torch.where(
            valid_candidates, student_log_probability, torch.zeros_like(student_log_probability)
        )
        return (
            teacher_probability * (teacher_log_probability - student_log_probability)
        ).sum(dim=1).mean()

    return relation_kl(fake_anchor), relation_kl(day_anchor), {
        "valid_anchors": valid_anchors.sum().detach()
    }


def multi_positive_place_loss(
    descriptors: torch.Tensor,
    labels: torch.Tensor,
    source_keys: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervised contrastive loss across distinct source images of a place.

    All views of the anchor's source are excluded from the denominator.  Thus a
    day/fake or day/augmentation pair cannot satisfy the positive requirement.
    Each distinct-source positive contributes its own log probability, rather
    than allowing one easy positive to dominate a summed positive probability.
    """
    if descriptors.ndim != 2 or labels.shape != descriptors.shape[:1] or source_keys.shape != labels.shape:
        raise ValueError("Place descriptors and metadata must have matching [N] batch dimensions")
    if not temperature > 0:
        raise ValueError("Place contrastive temperature must be positive")
    candidates = source_keys[:, None].ne(source_keys[None, :])
    same_place = labels[:, None].eq(labels[None, :])
    positives = candidates & same_place
    negatives = candidates & ~same_place
    valid = positives.any(dim=1) & negatives.any(dim=1)
    stats = {
        "place_valid_anchors": valid.sum().detach(),
        "place_positive_pairs": positives[valid].sum().detach(),
    }
    if not bool(valid.any()):
        return descriptors.sum() * 0.0, stats
    value = F.normalize(descriptors.float(), dim=1)
    logits = (value @ value.T / float(temperature))[valid]
    logits = logits.masked_fill(~candidates[valid], -torch.inf)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_probability = torch.where(
        positives[valid], log_probability, torch.zeros_like(log_probability)
    )
    loss = -(positive_log_probability.sum(dim=1) / positives[valid].sum(dim=1)).mean()
    return loss, stats


def patch_validity_mask(
    valid_mask: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    """Flatten a patch mask, or conservatively reduce a pixel mask to patches."""
    batch, _, height, width = features.shape
    if valid_mask.shape == (batch, height * width):
        return valid_mask.bool()
    if valid_mask.ndim != 3 or valid_mask.shape[0] != batch:
        raise ValueError("Validity masks must have shape [B,H,W] or [B,P]")
    if valid_mask.shape[-2:] == (height, width):
        return valid_mask.bool().flatten(1)
    coverage = F.adaptive_avg_pool2d(valid_mask.float().unsqueeze(1), (height, width))
    # A token touching any padding is excluded from local supervision.
    return coverage[:, 0].flatten(1).ge(1.0 - 1e-6)


@torch.no_grad()
def frozen_patch_reliability_target(
    day_features: torch.Tensor,
    augmented_features: torch.Tensor,
    day_images: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Detached stability × information heuristic for geometrically aligned views.

    The reference features must come from a frozen, evaluation-mode DINOv2.
    Feature deviation from the valid-image mean measures local distinctiveness;
    within-patch RGB spatial variation downweights uniform areas.  This is a
    soft heuristic, not a semantic guarantee that a token is a fixed landmark.
    """
    if day_features.shape != augmented_features.shape or day_features.ndim != 4:
        raise ValueError("Frozen reference features must have matching [B,C,H,W] shapes")
    if day_images.ndim != 4 or day_images.shape[:2] != (day_features.shape[0], 3):
        raise ValueError("Day images must have shape [B,3,H,W]")
    valid = patch_validity_mask(valid_mask, day_features)
    value = F.normalize(day_features.detach().float().flatten(2), dim=1)
    augmented = F.normalize(augmented_features.detach().float().flatten(2), dim=1)
    stability = (value * augmented).sum(dim=1).clamp(0.0, 1.0)
    valid_weights = valid[:, None].to(value.dtype)
    image_mean = (value * valid_weights).sum(dim=2) / valid_weights.sum(dim=2).clamp_min(1.0)
    image_mean = F.normalize(image_mean, dim=1)
    distinctiveness = (
        4.0 * (1.0 - (value * image_mean.unsqueeze(-1)).sum(dim=1))
    ).clamp(0.0, 1.0)
    mean = day_images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = day_images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    rgb = (day_images.detach().float() * std.float() + mean.float()).clamp(0.0, 1.0)
    grid = day_features.shape[-2:]
    if rgb.shape[-2] % grid[0] or rgb.shape[-1] % grid[1]:
        raise ValueError("Aligned reference image dimensions must divide evenly into the patch grid")
    patch_height, patch_width = rgb.shape[-2] // grid[0], rgb.shape[-1] // grid[1]
    pixels = rgb.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
    # var is stable for constant patches; E[x²] - E[x]² can leave cancellation
    # noise that gives a uniform sky a spurious nonzero texture target.
    pixel_variance = pixels.var(dim=(-2, -1), unbiased=False)
    texture = (pixel_variance.mean(dim=1).sqrt().flatten(1) / 0.05).clamp(0.0, 1.0)
    return (stability * distinctiveness * texture * valid).detach()


def stable_local_consistency_loss(
    day_features: torch.Tensor,
    augmented_features: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cosine consistency only on aligned, informative reference patches."""
    if day_features.shape != augmented_features.shape:
        raise ValueError("Student local features must share matching geometry")
    valid = patch_validity_mask(valid_mask, day_features)
    if target.shape != valid.shape:
        raise ValueError("Reference target must match the student patch grid")
    weight = target.detach().float().clamp(0.0, 1.0) * valid
    discrepancy = 1.0 - F.cosine_similarity(
        day_features.float().flatten(2), augmented_features.float().flatten(2), dim=1
    )
    loss = (discrepancy.clamp_min(0.0) * weight).sum() / weight.sum().clamp_min(1e-12)
    return loss, {
        "stable_valid_patches": valid.sum().detach(),
        "stable_target_mean": (weight.sum() / valid.sum().clamp_min(1)).detach(),
    }


def patch_reliability_loss(
    reliability: torch.Tensor,
    valid_mask: torch.Tensor,
    target: torch.Tensor | None = None,
    *,
    floor: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Optional detached BCE supervision and a per-image valid-patch mean floor."""
    if reliability.ndim != 2 or valid_mask.shape != reliability.shape:
        raise ValueError("Reliability and validity must have matching [B,P] shapes")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("Reliability floor must lie in [0,1]")
    if target is not None and target.shape != reliability.shape:
        raise ValueError("Reliability target must match [B,P]")
    valid = valid_mask.bool()
    counts = valid.sum(dim=1)
    valid_images = counts.gt(0)
    value = reliability.float()
    means = (value * valid).sum(dim=1) / counts.clamp_min(1)
    zero = value.sum() * 0.0
    mean_floor = (F.relu(float(floor) - means[valid_images]).mean() if bool(valid_images.any()) else zero)
    supervised = zero
    if target is not None and bool(valid.any()):
        supervised = F.binary_cross_entropy(
            value[valid].clamp(1e-6, 1.0 - 1e-6),
            target.detach().float()[valid].clamp(0.0, 1.0),
        )
    return supervised + mean_floor, {
        "reliability_bce": supervised.detach(),
        "reliability_floor_loss": mean_floor.detach(),
        "reliability_valid_mean": (means[valid_images].mean().detach() if bool(valid_images.any()) else zero.detach()),
        "reliability_valid_patches": valid.sum().detach(),
    }
