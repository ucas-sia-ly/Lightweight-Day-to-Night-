"""DINOv2-S adaptation of the local official CricaVPR implementation."""
from __future__ import annotations

import importlib
from pathlib import Path

import torch
import math
import types
from torch import nn


def _dynamic_forward(self, x: torch.Tensor) -> torch.Tensor:
    features = self.backbone(x)
    tokens = features["x_norm_patchtokens"]
    batch, patches, dim = tokens.shape
    side = math.isqrt(patches)
    if side * side != patches:
        raise RuntimeError(f"CricaVPR requires a square patch grid, got {patches}")
    grid = tokens.view(batch, side, side, dim).permute(0, 3, 1, 2)

    def aggregate(rows, cols):
        return self.aggregation(grid[:, :, rows, :][:, :, :, cols])

    parts = [features["x_norm_clstoken"]]
    for row_group in torch.tensor_split(torch.arange(side, device=grid.device), 2):
        for col_group in torch.tensor_split(torch.arange(side, device=grid.device), 2):
            parts.append(aggregate(row_group, col_group))
    for row_group in torch.tensor_split(torch.arange(side, device=grid.device), 3):
        for col_group in torch.tensor_split(torch.arange(side, device=grid.device), 3):
            parts.append(aggregate(row_group, col_group))
    encoded = self.encoder(torch.stack(parts, dim=1)).view(batch, 14 * dim)
    return torch.nn.functional.normalize(encoded, p=2, dim=-1)


def build_crica_small(repo: str, foundation_checkpoint: str = "") -> nn.Module:
    # Use the project's existing import isolation; never edit third-party files.
    from ..references.models import _isolated_repo_import

    with _isolated_repo_import(repo, ("network", "backbone")):
        network = importlib.import_module("network")
        blocks = importlib.import_module("backbone.dinov2.block")
        model = network.CricaVPRNet.__new__(network.CricaVPRNet)
        nn.Module.__init__(model)
        model.backbone = network.vit_small(
            patch_size=14, img_size=518, init_values=1, block_chunks=0
        )
        for block in model.backbone.blocks:
            # Preserve the official adapter's channel ratios at half the width.
            block.adapter = blocks.MulConvAdapter(384, 192, 96, 12, 48, 12, 48)
        model.aggregation = nn.Sequential(
            network.L2Norm(), network.GeM(work_with_tokens=None), network.Flatten()
        )
        model.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=384, nhead=16, dim_feedforward=2048,
                activation="gelu", dropout=0.1, batch_first=False,
            ), num_layers=2,
        )
    model.descriptor_dim = 14 * 384
    model.forward = types.MethodType(_dynamic_forward, model)
    if foundation_checkpoint:
        state = torch.load(Path(foundation_checkpoint), map_location="cpu", weights_only=True)
        state = state.get("model", state)
        foundation_keys = {k for k in model.backbone.state_dict() if ".adapter." not in k}
        if set(state) != foundation_keys:
            raise ValueError("Expected a complete plain DINOv2-S checkpoint without adapters")
        result = model.backbone.load_state_dict(state, strict=False)
        if any(".adapter." not in k for k in result.missing_keys) or result.unexpected_keys:
            raise ValueError(f"DINOv2-S foundation checkpoint mismatch: {result}")
    return model


def initialize_crica_training(model: nn.Module) -> None:
    """Match the official train.py freezing and adapter initialization."""
    for name, parameter in model.backbone.named_parameters():
        parameter.requires_grad_("adapter" in name)
    for block in model.backbone.blocks:
        nn.init.zeros_(block.adapter.D_fc2.weight)
        nn.init.zeros_(block.adapter.D_fc2.bias)
        for module in block.adapter.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.weight, 0.00001)
                nn.init.constant_(module.bias, 0.00001)
