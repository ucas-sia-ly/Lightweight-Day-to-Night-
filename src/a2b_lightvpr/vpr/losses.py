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


def pair_hinge(day: torch.Tensor, fake: torch.Tensor, target: float = 0.85) -> torch.Tensor:
    agreement = F.cosine_similarity(day, fake, dim=1)
    return F.relu(float(target) - agreement).mean()


def relation_distillation(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    teacher = F.normalize(teacher.detach(), dim=1)
    student = F.normalize(student, dim=1)
    teacher_relation = teacher @ teacher.T
    student_relation = student @ student.T
    mask = ~torch.eye(teacher.shape[0], dtype=torch.bool, device=teacher.device)
    if not mask.any():
        return student.sum() * 0.0
    return F.smooth_l1_loss(student_relation[mask], teacher_relation[mask])


def preservation_hinge(
    teacher: torch.Tensor, student: torch.Tensor, target: float = 0.95
) -> torch.Tensor:
    agreement = F.cosine_similarity(teacher.detach(), student, dim=1)
    return F.relu(float(target) - agreement).mean()


def ikt_loss(
    teacher_descriptors: torch.Tensor,
    fake_descriptors: torch.Tensor,
    classifier: MarginCosineClassifier,
    temperature: float = 4.0,
) -> torch.Tensor:
    teacher_logits = classifier(teacher_descriptors.detach()) / temperature
    student_logits = classifier(fake_descriptors) / temperature
    return (
        F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean",
        )
        * temperature**2
    )


def vpr_training_loss(
    profile: str,
    day: torch.Tensor,
    fake: torch.Tensor,
    labels: torch.Tensor,
    classifier: MarginCosineClassifier,
    *,
    teacher_day: torch.Tensor | None = None,
    pair_weight: float = 0.5,
    pair_target: float = 0.85,
    relation_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fake_classification = F.cross_entropy(classifier(fake, labels), labels)
    components = {"classification_fake": fake_classification}
    if profile == "v0":
        return fake_classification, components
    if profile == "v1":
        if teacher_day is None:
            raise ValueError("V1 IKT requires teacher descriptors")
        auxiliary = ikt_loss(teacher_day, fake, classifier)
        components["ikt"] = auxiliary
        return fake_classification + auxiliary, components

    day_classification = F.cross_entropy(classifier(day, labels), labels)
    cross_domain = pair_hinge(day, fake, pair_target)
    total = 0.5 * (fake_classification + day_classification) + pair_weight * cross_domain
    components.update(classification_day=day_classification, pair=cross_domain)
    if profile == "v3":
        if teacher_day is None:
            raise ValueError("V3 relation distillation requires teacher descriptors")
        relation = relation_distillation(teacher_day, fake)
        components["relation"] = relation
        total = total + relation_weight * relation
    return total, components


def configurable_vpr_loss(
    train_domains: str,
    auxiliary: str,
    day: torch.Tensor | None,
    fake: torch.Tensor | None,
    labels: torch.Tensor,
    classifier: MarginCosineClassifier,
    *,
    teacher_day: torch.Tensor | None = None,
    pair_weight: float = 0.5,
    pair_target: float = 0.85,
    relation_weight: float = 0.5,
    preservation_weight: float = 1.0,
    preservation_target: float = 0.95,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    components = {}
    classification = []
    if train_domains in {"day", "day_fake"}:
        if day is None:
            raise ValueError("Day descriptors are required by train_domains")
        components["classification_day"] = F.cross_entropy(classifier(day, labels), labels)
        classification.append(components["classification_day"])
    if train_domains in {"fake", "day_fake"}:
        if fake is None:
            raise ValueError("Fake descriptors are required by train_domains")
        components["classification_fake"] = F.cross_entropy(classifier(fake, labels), labels)
        classification.append(components["classification_fake"])
    if not classification:
        raise ValueError("At least one training domain is required")
    total = torch.stack(classification).mean()
    if auxiliary == "pair":
        if day is None or fake is None:
            raise ValueError("Pair auxiliary requires day_fake training")
        components["pair"] = pair_hinge(day, fake, pair_target)
        total = total + pair_weight * components["pair"]
    elif auxiliary == "ikt":
        if fake is None or teacher_day is None:
            raise ValueError("IKT requires fake and frozen day-teacher descriptors")
        components["ikt"] = ikt_loss(teacher_day, fake, classifier)
        total = total + components["ikt"]
    elif auxiliary == "relation":
        if fake is None or teacher_day is None:
            raise ValueError("Relation auxiliary requires fake and teacher descriptors")
        components["relation"] = relation_distillation(teacher_day, fake)
        total = total + relation_weight * components["relation"]
    elif auxiliary == "preserve_day":
        if day is None or teacher_day is None:
            raise ValueError("Day preservation requires day and frozen teacher descriptors")
        components["preservation_day"] = preservation_hinge(
            teacher_day, day, preservation_target
        )
        total = total + preservation_weight * components["preservation_day"]
    elif auxiliary != "none":
        raise ValueError(f"Unsupported auxiliary: {auxiliary}")
    return total, components
