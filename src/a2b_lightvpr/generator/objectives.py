from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def _sample_features(
    feature: torch.Tensor,
    count: int,
    ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels, height, width = feature.shape
    flat = feature.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    if ids is None:
        count = min(int(count), height * width)
        ids = torch.randperm(height * width, device=feature.device)[:count]
    sampled = flat[:, ids, :]
    return sampled, ids


def _batch_patch_nce(query: torch.Tensor, key: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    query = F.normalize(query, dim=-1)
    key = F.normalize(key.detach(), dim=-1)
    positive = (query * key).sum(-1, keepdim=True)
    negatives = torch.bmm(query, key.transpose(1, 2))
    eye = torch.eye(negatives.shape[1], device=query.device, dtype=torch.bool).unsqueeze(0)
    negatives = negatives.masked_fill(eye, -10.0)
    logits = torch.cat((positive, negatives), dim=-1) / temperature
    labels = torch.zeros(logits.shape[:2], dtype=torch.long, device=query.device)
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten())


class ContentObjective(nn.Module, ABC):
    name: str

    @abstractmethod
    def generator_loss(
        self,
        source: torch.Tensor,
        fake: torch.Tensor,
        source_features: list[torch.Tensor],
        fake_features: list[torch.Tensor],
    ) -> torch.Tensor:
        raise NotImplementedError

    def adversary_parameters(self) -> Iterable[nn.Parameter]:
        return ()

    def adversary_loss(self, source_features: list[torch.Tensor], fake_features: list[torch.Tensor]):
        return None

    def identity_loss(
        self, target_features: list[torch.Tensor], identity_features: list[torch.Tensor]
    ) -> torch.Tensor | None:
        return None


class ProjectionHeads(nn.Module):
    def __init__(self, input_dims: tuple[int, ...], output_dim: int):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, output_dim), nn.ReLU(True), nn.Linear(output_dim, output_dim))
            for dim in input_dims
        )

    def forward(self, features: list[torch.Tensor], count: int, ids=None):
        projected, returned_ids = [], []
        for index, (feature, head) in enumerate(zip(features, self.heads)):
            sampled, patch_ids = _sample_features(feature, count, None if ids is None else ids[index])
            projected.append(head(sampled))
            returned_ids.append(patch_ids)
        return projected, returned_ids


class LazyProjectionHeads(nn.Module):
    """QS-Attn netF MLPs are initialized from the tapped feature shapes."""

    def __init__(self, count: int, output_dim: int):
        super().__init__()
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.LazyLinear(output_dim), nn.ReLU(True), nn.Linear(output_dim, output_dim)
            )
            for _ in range(count)
        )


class HardNegativeGenerator(nn.Module):
    def __init__(self, input_dims: tuple[int, ...], feature_dim: int, noise_dim: int = 64):
        super().__init__()
        self.feature_projections = nn.ModuleList(nn.Linear(dim, feature_dim) for dim in input_dims)
        self.generators = nn.ModuleList(
            nn.Sequential(
                nn.Linear(feature_dim + noise_dim, feature_dim),
                nn.ReLU(True),
                nn.Linear(feature_dim, feature_dim),
            )
            for _ in input_dims
        )
        self.noise_dim = noise_dim

    def forward(self, features: list[torch.Tensor], count: int, noise=None):
        outputs, noises = [], []
        for index, (feature, project, generate) in enumerate(
            zip(features, self.feature_projections, self.generators)
        ):
            context = feature.mean(dim=(2, 3))
            context = project(context).unsqueeze(1).expand(-1, count, -1)
            current_noise = (
                torch.randn(context.shape[0], count, self.noise_dim, device=feature.device)
                if noise is None
                else noise[index]
            )
            outputs.append(F.normalize(generate(torch.cat((context, current_noise), dim=-1)), dim=-1))
            noises.append(current_noise)
        return outputs, noises


def _negative_nce(query: torch.Tensor, key: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    query = F.normalize(query, dim=-1)
    key = F.normalize(key.detach(), dim=-1)
    positive = (query * key).sum(-1, keepdim=True)
    negative_logits = torch.bmm(query, negative.transpose(1, 2))
    logits = torch.cat((positive, negative_logits), dim=-1) / 0.07
    labels = torch.zeros(logits.shape[:2], dtype=torch.long, device=query.device)
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten())


class NegCUTObjective(ContentObjective):
    name = "negcut"

    def __init__(self, num_patches: int, feature_dim: int):
        super().__init__()
        input_dims = (64, 128, 256, 256, 256)
        self.num_patches = num_patches
        self.projector = ProjectionHeads(input_dims, feature_dim)
        self.negative_generator = HardNegativeGenerator(input_dims, feature_dim)

    def generator_loss(self, source, fake, source_features, fake_features):
        keys, ids = self.projector(source_features, self.num_patches)
        queries, _ = self.projector(fake_features, self.num_patches, ids)
        negatives, _ = self.negative_generator(source_features, self.num_patches)
        losses = [
            _negative_nce(query, key, negative.detach())
            for query, key, negative in zip(queries, keys, negatives)
        ]
        return torch.stack(losses).mean()

    def adversary_parameters(self):
        return self.negative_generator.parameters()

    def adversary_loss(self, source_features, fake_features):
        with torch.no_grad():
            keys, ids = self.projector(source_features, self.num_patches)
            queries, _ = self.projector(fake_features, self.num_patches, ids)
        detached_source = [feature.detach() for feature in source_features]
        negatives_a, noise_a = self.negative_generator(detached_source, self.num_patches)
        negatives_b, noise_b = self.negative_generator(detached_source, self.num_patches)
        hardness = torch.stack(
            [_negative_nce(q.detach(), k.detach(), n) for q, k, n in zip(queries, keys, negatives_a)]
        ).mean()
        diversity_terms = []
        for first, second, z_first, z_second in zip(negatives_a, negatives_b, noise_a, noise_b):
            feature_distance = (first - second).abs().mean()
            noise_distance = (z_first - z_second).abs().mean().clamp_min(1e-6)
            diversity_terms.append(feature_distance / noise_distance)
        diversity = torch.stack(diversity_terms).mean()
        return -hardness - diversity


class QSAttnObjective(ContentObjective):
    name = "qsattn"

    def __init__(self, num_patches: int, feature_dim: int):
        super().__init__()
        self.num_patches = num_patches
        self.projector = LazyProjectionHeads(5, feature_dim)

    def _selected(self, source: torch.Tensor, fake: torch.Tensor):
        batch, channels, height, width = source.shape
        source_flat = source.flatten(2).transpose(1, 2)
        fake_flat = fake.flatten(2).transpose(1, 2)
        source_norm = F.normalize(source_flat, dim=-1)
        attention = torch.bmm(source_norm, source_norm.transpose(1, 2)).softmax(dim=-1)
        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1)
        count = min(self.num_patches, height * width)
        selected = entropy.argsort(dim=-1)[:, :count]
        gather_index = selected.unsqueeze(-1).expand(-1, -1, channels)
        routed_source = torch.bmm(attention.gather(1, selected.unsqueeze(-1).expand(-1, -1, height * width)), source_flat)
        routed_fake = torch.bmm(attention.gather(1, selected.unsqueeze(-1).expand(-1, -1, height * width)), fake_flat)
        return routed_source, routed_fake

    def _feature_loss(self, source_features, fake_features):
        losses = []
        for index, (source_feature, fake_feature, head) in enumerate(
            zip(source_features, fake_features, self.projector.heads)
        ):
            if index < 3:
                source_routed, ids = _sample_features(
                    source_feature.detach(), self.num_patches
                )
                fake_routed, _ = _sample_features(fake_feature, self.num_patches, ids)
            else:
                source_routed, fake_routed = self._selected(
                    source_feature.detach(), fake_feature
                )
            losses.append(_batch_patch_nce(head(fake_routed), head(source_routed)))
        return torch.stack(losses).mean()

    def generator_loss(self, source, fake, source_features, fake_features):
        del source, fake
        return self._feature_loss(source_features, fake_features)

    def identity_loss(self, target_features, identity_features):
        return self._feature_loss(target_features, identity_features)


def build_content_objective(
    name: str,
    *,
    num_patches: int = 64,
    feature_dim: int = 256,
) -> ContentObjective:
    builders = {
        "negcut": lambda: NegCUTObjective(num_patches, feature_dim),
        "qsattn": lambda: QSAttnObjective(num_patches, feature_dim),
    }
    try:
        return builders[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown content objective: {name}") from exc
