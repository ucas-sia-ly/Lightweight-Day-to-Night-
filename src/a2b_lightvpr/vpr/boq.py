from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def _load_dinov2(repo: str, model_name: str) -> nn.Module:
    root = Path(repo).expanduser().resolve()
    if not (root / "hubconf.py").is_file():
        raise FileNotFoundError(f"Local DINOv2 repository is incomplete: {root}")
    return torch.hub.load(str(root), model_name, source="local", pretrained=False)


def _checkpoint_state(path: str | Path) -> dict[str, torch.Tensor]:
    state = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    if not isinstance(state, dict):
        raise TypeError("BoQ checkpoint does not contain a state dict")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def _backbone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("backbone.dino.") for key in state):
        return {key.removeprefix("backbone.dino."): value for key, value in state.items()
                if key.startswith("backbone.dino.")}
    if any(key.startswith("dino.") for key in state):
        return {key.removeprefix("dino."): value for key, value in state.items()
                if key.startswith("dino.")}
    return state


class _BoQBackbone(nn.Module):
    def __init__(
        self,
        repo: str,
        model_name: str = "dinov2_vitb14",
        checkpoint: str | Path = "",
        trainable_blocks: int = 0,
    ):
        super().__init__()
        self.dino = _load_dinov2(repo, model_name)
        if checkpoint:
            result = self.dino.load_state_dict(
                _backbone_state(_checkpoint_state(checkpoint)), strict=True
            )
            if result.missing_keys or result.unexpected_keys:
                raise ValueError(f"DINOv2 checkpoint mismatch: {result}")
        self.num_channels = int(self.dino.embed_dim)
        self.trainable_blocks = int(trainable_blocks)
        if not 0 <= self.trainable_blocks <= len(self.dino.blocks):
            raise ValueError("trainable_blocks is outside the DINOv2 depth")
        self.set_trainable(True)

    def set_trainable(self, enabled: bool) -> None:
        self.dino.requires_grad_(False)
        if enabled and self.trainable_blocks:
            for block in self.dino.blocks[-self.trainable_blocks:]:
                block.requires_grad_(True)
            self.dino.norm.requires_grad_(True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = images.shape
        patches = self.dino.get_intermediate_layers(
            images, n=[len(self.dino.blocks) - 1], norm=False
        )[0]
        expected = (height // 14) * (width // 14)
        if patches.shape[1] != expected:
            raise RuntimeError(
                f"DINOv2 returned {patches.shape[1]} patch tokens, expected {expected}"
            )
        return patches.transpose(1, 2).reshape(
            batch, self.num_channels, height // 14, width // 14
        )


class _BoQBlock(nn.Module):
    def __init__(self, dim: int = 384, num_queries: int = 64):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            dim, max(1, dim // 64), 4 * dim, batch_first=True, dropout=0.0
        )
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        self.self_attn = nn.MultiheadAttention(dim, max(1, dim // 64), batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, max(1, dim // 64), batch_first=True)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.encoder(value)
        queries = self.queries.repeat(value.shape[0], 1, 1)
        queries = self.norm_q(queries + self.self_attn(queries, queries, queries)[0])
        output = self.norm_out(self.cross_attn(queries, value, value)[0])
        return value, output


class _BoQHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 768,
        proj_channels: int = 384,
        num_queries: int = 64,
        num_layers: int = 2,
        row_dim: int = 32,
    ):
        super().__init__()
        self.proj_c = nn.Conv2d(in_channels, proj_channels, 3, padding=1)
        self.norm_input = nn.LayerNorm(proj_channels)
        self.boqs = nn.ModuleList(
            _BoQBlock(proj_channels, num_queries) for _ in range(num_layers)
        )
        self.fc = nn.Linear(num_layers * num_queries, row_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.norm_input(self.proj_c(value).flatten(2).permute(0, 2, 1))
        outputs = []
        for block in self.boqs:
            value, output = block(value)
            outputs.append(output)
        descriptor = self.fc(torch.cat(outputs, dim=1).permute(0, 2, 1)).flatten(1)
        return F.normalize(descriptor, dim=-1)


class BoQReferenceModel(nn.Module):
    def __init__(
        self,
        repo: str,
        checkpoint: str,
        *,
        backbone_name: str = "dinov2_vitb14",
        descriptor_dim: int = 12_288,
        backbone_checkpoint: str = "",
        trainable_blocks: int = 0,
    ):
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        full_state = _checkpoint_state(checkpoint)
        self.backbone = _BoQBackbone(
            repo,
            backbone_name,
            backbone_checkpoint or checkpoint,
            trainable_blocks,
        )
        if self.descriptor_dim % 384:
            raise ValueError("BoQ descriptor_dim must be divisible by 384")
        self.aggregator = _BoQHead(
            in_channels=self.backbone.num_channels,
            row_dim=self.descriptor_dim // 384,
        )
        result = self.load_state_dict(full_state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f"BoQ checkpoint mismatch: {result}")

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.aggregator(self.backbone(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)


class BoQSmallModel(nn.Module):
    """Trainable DINOv2-S/14 + BoQ model without changing the reference repository."""

    descriptor_dim = 12_288

    def __init__(
        self,
        repo: str,
        backbone_name: str,
        backbone_checkpoint: str,
        trainable_blocks: int = 2,
        aggregator_repo: str = "",
    ):
        super().__init__()
        self.backbone = _BoQBackbone(
            repo, backbone_name, backbone_checkpoint, trainable_blocks
        )
        if self.backbone.num_channels != 384:
            raise ValueError("BoQSmallModel requires a 384-dimensional DINOv2-S backbone")
        if aggregator_repo:
            source = Path(aggregator_repo).expanduser().resolve() / "src" / "boq.py"
            if not source.is_file():
                raise FileNotFoundError(f"Third-party BoQ aggregator is missing: {source}")
            spec = importlib.util.spec_from_file_location("_a2b_external_boq", source)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load third-party BoQ aggregator: {source}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.aggregator = module.BoQ(
                in_channels=384, proj_channels=384, num_queries=64,
                num_layers=2, row_dim=32,
            )
        else:
            self.aggregator = _BoQHead(
                in_channels=384, proj_channels=384, row_dim=32
            )
        self.descriptor_dim = 12_288

    def set_backbone_trainable(self, enabled: bool) -> None:
        self.backbone.set_trainable(enabled)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        descriptor = self.aggregator(self.backbone(images))
        if isinstance(descriptor, (tuple, list)):
            descriptor = descriptor[0]
        if descriptor.shape[1] != self.descriptor_dim:
            raise RuntimeError(
                f"Descriptor contract violated: expected {self.descriptor_dim}, "
                f"got {descriptor.shape[1]}"
            )
        return F.normalize(descriptor, dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)


class PCAProjectedReference(nn.Module):
    def __init__(self, model: nn.Module, projection: str | Path, output_dim: int):
        super().__init__()
        state = torch.load(
            Path(projection).expanduser(), map_location="cpu", weights_only=True
        )
        if state.get("kind") != "reference_pca":
            raise ValueError("Unsupported reference projection checkpoint")
        if int(state["input_dim"]) != int(model.descriptor_dim):
            raise ValueError("PCA input dimension does not match the reference model")
        if int(state["output_dim"]) != int(output_dim):
            raise ValueError("PCA output dimension does not match the registry")
        self.model = model
        self.descriptor_dim = int(output_dim)
        self.register_buffer("mean", state["mean"].float())
        self.register_buffer("components", state["components"].float())
        if self.mean.shape != (int(model.descriptor_dim),):
            raise ValueError("PCA mean has the wrong shape")
        if self.components.shape != (int(model.descriptor_dim), self.descriptor_dim):
            raise ValueError("PCA component matrix has the wrong shape")
        self.projection_metadata = state.get("metadata", {})

    def project(self, descriptors: torch.Tensor) -> torch.Tensor:
        descriptors = descriptors.float()
        projected = (descriptors - self.mean) @ self.components
        return F.normalize(projected, dim=1)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.project(self.model.encode(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)
