from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def _checkpoint_state(path: str | Path) -> dict[str, torch.Tensor]:
    state = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    if not isinstance(state, dict):
        raise TypeError("DINOv2 checkpoint does not contain a state dict")
    return state


class AnyLocSmallModel(nn.Module):
    descriptor_dim = 12_288

    def __init__(
        self,
        repo: str,
        backbone_name: str,
        backbone_checkpoint: str,
        vocabulary: str | Path,
        *,
        layer: int = 11,
        facet: str = "token",
        num_clusters: int = 32,
    ):
        super().__init__()
        if facet != "token":
            raise ValueError("The reproducible AnyLoc-S adapter supports token features only")
        root = Path(repo).expanduser().resolve()
        if not (root / "hubconf.py").is_file():
            raise FileNotFoundError(f"Local DINOv2 repository is incomplete: {root}")
        self.backbone = torch.hub.load(
            str(root), backbone_name, source="local", pretrained=False
        )
        result = self.backbone.load_state_dict(
            _checkpoint_state(backbone_checkpoint), strict=True
        )
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f"DINOv2 checkpoint mismatch: {result}")
        self.backbone.requires_grad_(False)
        self.layer = int(layer)
        if self.layer != len(self.backbone.blocks) - 1:
            raise ValueError("AnyLoc-S currently requires the final DINOv2-S block")
        self.facet = facet
        self.num_clusters = int(num_clusters)
        self.patch_dim = int(self.backbone.embed_dim)
        self.descriptor_dim = self.num_clusters * self.patch_dim
        if self.patch_dim != 384 or self.descriptor_dim != 12_288:
            raise ValueError("AnyLoc-S requires 32 clusters over 384-D DINOv2-S tokens")
        vocabulary_state = torch.load(
            Path(vocabulary).expanduser(), map_location="cpu", weights_only=True
        )
        centers = vocabulary_state.get("centers") if isinstance(vocabulary_state, dict) else None
        if centers is None or tuple(centers.shape) != (self.num_clusters, self.patch_dim):
            raise ValueError("AnyLoc-S vocabulary must have shape [32, 384]")
        self.register_buffer("centers", F.normalize(centers.float(), dim=1))
        self.vocabulary_metadata = vocabulary_state.get("metadata", {})
        expected_metadata = {
            "layer": self.layer,
            "facet": self.facet,
            "num_clusters": self.num_clusters,
            "patch_dim": self.patch_dim,
            "descriptor_dim": self.descriptor_dim,
            "official_checkpoint": False,
        }
        mismatches = {
            key: (self.vocabulary_metadata.get(key), value)
            for key, value in expected_metadata.items()
            if self.vocabulary_metadata and self.vocabulary_metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"AnyLoc-S vocabulary metadata mismatch: {mismatches}")

    def patch_descriptors(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.backbone.get_intermediate_layers(
            images, n=[self.layer], norm=False
        )[0]
        height, width = images.shape[-2:]
        expected = (height // 14) * (width // 14)
        if patches.shape[1] != expected:
            raise RuntimeError(
                f"DINOv2 returned {patches.shape[1]} patch tokens, expected {expected}"
            )
        return F.normalize(patches.float(), dim=-1)

    def encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        patches = F.normalize(patches.float(), dim=-1)
        labels = (patches @ self.centers.T).argmax(dim=-1)
        batch, _, channels = patches.shape
        residuals = patches - self.centers[labels]
        pooled = patches.new_zeros(batch, self.num_clusters, channels)
        pooled.scatter_add_(
            1, labels.unsqueeze(-1).expand(-1, -1, channels), residuals
        )
        pooled = F.normalize(pooled, dim=-1)
        return F.normalize(pooled.flatten(1), dim=1)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_patches(self.patch_descriptors(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)
