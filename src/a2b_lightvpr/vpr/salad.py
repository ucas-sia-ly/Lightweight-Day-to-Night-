from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def _log_sinkhorn(
    z: torch.Tensor,
    log_mu: torch.Tensor,
    log_nu: torch.Tensor,
) -> torch.Tensor:
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(3):
        u = log_mu - torch.logsumexp(z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(z + u.unsqueeze(2), dim=1)
    return z + u.unsqueeze(2) + v.unsqueeze(1)


def _optimal_transport(scores: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    batch, clusters, patches = scores.shape
    one = scores.new_tensor(1)
    m, n = clusters * one, patches * one
    bins = alpha.expand(batch, 1, patches)
    couplings = torch.cat((scores, bins), dim=1)
    norm = -(m + n).log()
    log_mu = torch.cat((norm.expand(clusters), (n - m).log()[None] + norm))
    log_nu = norm.expand(patches)
    return _log_sinkhorn(
        couplings,
        log_mu[None].expand(batch, -1),
        log_nu[None].expand(batch, -1),
    ) - norm


class CompactSALADHead(nn.Module):
    output_dim = 544

    def __init__(
        self,
        num_channels: int,
        num_clusters: int = 16,
        cluster_dim: int = 32,
        token_dim: int = 32,
    ):
        super().__init__()
        self.num_clusters = int(num_clusters)
        self.cluster_dim = int(cluster_dim)
        self.token_dim = int(token_dim)
        self.token_features = nn.Sequential(
            nn.Linear(num_channels, 512), nn.ReLU(), nn.Linear(512, token_dim)
        )
        self.cluster_features = nn.Sequential(
            nn.Conv2d(num_channels, 512, 1),
            nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(512, cluster_dim, 1),
        )
        self.score = nn.Sequential(
            nn.Conv2d(num_channels, 512, 1),
            nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(512, num_clusters, 1),
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    def forward(self, value: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        features, token = value
        local = self.cluster_features(features).flatten(2)
        scores = self.score(features).flatten(2)
        token = self.token_features(token)
        assignments = torch.exp(_optimal_transport(scores, self.dust_bin))[:, :-1]
        assignments = assignments.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        local = local.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        descriptor = torch.cat(
            (
                F.normalize(token, dim=-1),
                F.normalize((local * assignments).sum(dim=-1), dim=1).flatten(1),
            ),
            dim=-1,
        )
        return F.normalize(descriptor, dim=-1)


def _checkpoint_state(path: str | Path) -> dict[str, torch.Tensor]:
    state = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    if not isinstance(state, dict):
        raise TypeError("DINOv2 checkpoint does not contain a state dict")
    return state


class DINOv2SALADBackbone(nn.Module):
    def __init__(
        self,
        repo: str,
        model_name: str,
        checkpoint: str,
        trainable_blocks: int,
    ):
        super().__init__()
        if not torch.cuda.is_available():
            os.environ.setdefault("XFORMERS_DISABLED", "1")
        root = Path(repo).expanduser().resolve()
        if not (root / "hubconf.py").is_file():
            raise FileNotFoundError(f"Local DINOv2 repository is incomplete: {root}")
        self.model = torch.hub.load(
            str(root), model_name, source="local", pretrained=False
        )
        result = self.model.load_state_dict(_checkpoint_state(checkpoint), strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f"DINOv2 checkpoint mismatch: {result}")
        self.num_channels = int(self.model.embed_dim)
        self.trainable_blocks = int(trainable_blocks)
        if not 0 <= self.trainable_blocks <= len(self.model.blocks):
            raise ValueError("backbone_trainable_blocks is outside the DINOv2 depth")
        self.set_trainable(True)

    def set_trainable(self, enabled: bool) -> None:
        self.model.requires_grad_(False)
        if enabled and self.trainable_blocks:
            for block in self.model.blocks[-self.trainable_blocks :]:
                block.requires_grad_(True)
            self.model.norm.requires_grad_(True)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model.forward_features(images)
        patches = output["x_norm_patchtokens"]
        token = output["x_norm_clstoken"]
        batch, _, height, width = images.shape
        expected_patches = (height // 14) * (width // 14)
        if patches.shape[1] != expected_patches:
            raise RuntimeError(
                f"DINOv2 returned {patches.shape[1]} patch tokens, "
                f"expected {expected_patches}"
            )
        features = patches.reshape(
            batch, height // 14, width // 14, self.num_channels
        ).permute(0, 3, 1, 2).contiguous()
        return features, token


class SALADStudentModel(nn.Module):
    descriptor_dim = 544
    aggregator_name = "salad"

    def __init__(
        self,
        repo: str,
        model_name: str,
        checkpoint: str,
        trainable_blocks: int = 2,
    ):
        super().__init__()
        self.backbone = DINOv2SALADBackbone(
            repo, model_name, checkpoint, trainable_blocks
        )
        self.aggregator = CompactSALADHead(self.backbone.num_channels)

    def set_backbone_trainable(self, enabled: bool) -> None:
        self.backbone.set_trainable(enabled)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        descriptor = self.aggregator(self.backbone(images))
        if descriptor.shape[1] != self.descriptor_dim:
            raise RuntimeError(
                f"Descriptor contract violated: expected {self.descriptor_dim}, "
                f"got {descriptor.shape[1]}"
            )
        return F.normalize(descriptor, p=2, dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)
