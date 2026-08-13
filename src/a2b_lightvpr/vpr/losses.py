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
    valid_candidates = anchor_image_keys[:, None].ne(candidate_image_keys[None, :])
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


def route_supervised_contrastive_loss(
    fake_descriptors: torch.Tensor,
    day_descriptors: torch.Tensor,
    coordinates: torch.Tensor,
    yaw_deg: torch.Tensor,
    image_keys: torch.Tensor,
    *,
    positive_radius_m: float = 10.0,
    positive_yaw_deg: float = 30.0,
    negative_radius_m: float = 25.0,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise fake-to-day retrieval with metric RobotCar train metadata only."""
    if fake_descriptors.shape != day_descriptors.shape:
        raise ValueError("Route contrastive descriptors must share a shape")
    valid_metadata = torch.isfinite(coordinates).all(dim=1) & torch.isfinite(yaw_deg)
    different_image = image_keys[:, None].ne(image_keys[None, :])
    distances = torch.cdist(coordinates.float(), coordinates.float())
    yaw_delta = torch.remainder(
        yaw_deg[:, None].float() - yaw_deg[None, :].float() + 180.0, 360.0
    ) - 180.0
    metadata_pairs = valid_metadata[:, None] & valid_metadata[None, :] & different_image
    positives = (
        metadata_pairs
        & distances.le(float(positive_radius_m))
        & yaw_delta.abs().le(float(positive_yaw_deg))
    )
    negatives = metadata_pairs & distances.gt(float(negative_radius_m))
    valid_anchors = positives.any(dim=1) & negatives.any(dim=1)
    if not bool(valid_anchors.any()):
        zero = (fake_descriptors.sum() + day_descriptors.sum()) * 0.0
        return zero, {
            "valid_anchors": torch.zeros((), device=zero.device, dtype=torch.long),
            "positive_pairs": positives.sum().detach(),
            "negative_pairs": negatives.sum().detach(),
        }
    fake = F.normalize(fake_descriptors.float(), dim=1)
    day = F.normalize(day_descriptors.float(), dim=1)
    logits = fake @ day.T / float(temperature)
    candidates = positives | negatives
    logits = logits.masked_fill(~candidates, -torch.inf)
    log_denominator = torch.logsumexp(logits, dim=1)
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    log_positive = torch.logsumexp(positive_logits, dim=1)
    loss = -(log_positive[valid_anchors] - log_denominator[valid_anchors]).mean()
    return loss, {
        "valid_anchors": valid_anchors.sum().detach(),
        "positive_pairs": positives.sum().detach(),
        "negative_pairs": negatives.sum().detach(),
    }


def mined_route_contrastive_loss(
    fake_descriptors: torch.Tensor,
    candidate_descriptors: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Contrast fake anchors against frozen, offline-mined day candidates."""
    if candidate_descriptors.ndim != 3:
        raise ValueError("Mined route candidates must have shape [B,K,D]")
    if candidate_descriptors.shape[:2] != positive_mask.shape:
        raise ValueError("Mined route masks do not match candidate descriptors")
    if positive_mask.shape != negative_mask.shape:
        raise ValueError("Mined route positive and negative masks must match")
    if candidate_descriptors.shape[0] != fake_descriptors.shape[0]:
        raise ValueError("Mined route anchor and candidate batches must match")
    candidates = positive_mask | negative_mask
    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not bool(valid.any()):
        zero = fake_descriptors.sum() * 0.0
        return zero, {
            "valid_anchors": torch.zeros((), device=zero.device, dtype=torch.long),
            "positive_pairs": positive_mask.sum().detach(),
            "negative_pairs": negative_mask.sum().detach(),
        }
    anchors = F.normalize(fake_descriptors.float(), dim=1)
    frozen_candidates = F.normalize(candidate_descriptors.detach().float(), dim=2)
    logits = torch.einsum("bd,bkd->bk", anchors, frozen_candidates) / float(temperature)
    logits = logits.masked_fill(~candidates, -torch.inf)
    positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
    loss = -(
        torch.logsumexp(positive_logits, dim=1)[valid]
        - torch.logsumexp(logits, dim=1)[valid]
    ).mean()
    return loss, {
        "valid_anchors": valid.sum().detach(),
        "positive_pairs": positive_mask.sum().detach(),
        "negative_pairs": negative_mask.sum().detach(),
    }
