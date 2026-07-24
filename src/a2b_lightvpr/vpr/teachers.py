from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def _load_dinov2(repo: str, model_name: str = "dinov2_vitb14") -> nn.Module:
    root = Path(repo).expanduser().resolve()
    if not (root / "hubconf.py").is_file():
        raise FileNotFoundError(f"Local DINOv2 repository is incomplete: {root}")
    return torch.hub.load(str(root), model_name, source="local", pretrained=False)


def _strict_load(model: nn.Module, checkpoint: str, label: str) -> None:
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"{label} checkpoint does not contain a state dict")
    model.load_state_dict(state, strict=True)


def _log_sinkhorn(z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor) -> torch.Tensor:
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


class _SaladBackbone(nn.Module):
    def __init__(self, repo: str):
        super().__init__()
        self.model = _load_dinov2(repo)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = images.shape
        tokens = self.model.prepare_tokens_with_masks(images)
        for block in self.model.blocks:
            tokens = block(tokens)
        tokens = self.model.norm(tokens)
        features = tokens[:, 1:].reshape(batch, height // 14, width // 14, 768)
        return features.permute(0, 3, 1, 2).contiguous(), tokens[:, 0]


class _SaladHead(nn.Module):
    def __init__(self, num_clusters: int = 16, cluster_dim: int = 32, token_dim: int = 32):
        super().__init__()
        self.num_clusters = int(num_clusters)
        self.cluster_dim = int(cluster_dim)
        self.token_dim = int(token_dim)
        self.token_features = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, token_dim))
        self.cluster_features = nn.Sequential(
            nn.Conv2d(768, 512, 1), nn.Identity(), nn.ReLU(), nn.Conv2d(512, cluster_dim, 1)
        )
        self.score = nn.Sequential(
            nn.Conv2d(768, 512, 1), nn.Identity(), nn.ReLU(), nn.Conv2d(512, num_clusters, 1)
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


class SaladTeacher(nn.Module):
    output_dim = 544

    def __init__(self, repo: str, checkpoint: str):
        super().__init__()
        self.backbone = _SaladBackbone(repo)
        self.aggregator = _SaladHead()
        _strict_load(self, checkpoint, "SALAD")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.aggregator(self.backbone(images))


class _BoQBackbone(nn.Module):
    def __init__(self, repo: str):
        super().__init__()
        self.dino = _load_dinov2(repo)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = images.shape
        tokens = self.dino.prepare_tokens_with_masks(images)
        for block in self.dino.blocks:
            tokens = block(tokens)
        patches = tokens[:, 1:].permute(0, 2, 1)
        return patches.reshape(batch, 768, height // 14, width // 14)


class _BoQBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(384, 6, 1536, batch_first=True, dropout=0.0)
        self.queries = nn.Parameter(torch.randn(1, 64, 384))
        self.self_attn = nn.MultiheadAttention(384, 6, batch_first=True)
        self.norm_q = nn.LayerNorm(384)
        self.cross_attn = nn.MultiheadAttention(384, 6, batch_first=True)
        self.norm_out = nn.LayerNorm(384)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.encoder(value)
        queries = self.queries.repeat(value.shape[0], 1, 1)
        queries = self.norm_q(queries + self.self_attn(queries, queries, queries)[0])
        output = self.norm_out(self.cross_attn(queries, value, value)[0])
        return value, output


class _BoQHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_c = nn.Conv2d(768, 384, 3, padding=1)
        self.norm_input = nn.LayerNorm(384)
        self.boqs = nn.ModuleList((_BoQBlock(), _BoQBlock()))
        self.fc = nn.Linear(128, 32)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.norm_input(self.proj_c(value).flatten(2).permute(0, 2, 1))
        outputs = []
        for block in self.boqs:
            value, output = block(value)
            outputs.append(output)
        descriptor = self.fc(torch.cat(outputs, dim=1).permute(0, 2, 1)).flatten(1)
        return F.normalize(descriptor, dim=-1)


class BoQTeacher(nn.Module):
    output_dim = 12_288

    def __init__(self, repo: str, checkpoint: str):
        super().__init__()
        self.backbone = _BoQBackbone(repo)
        self.aggregator = _BoQHead()
        _strict_load(self, checkpoint, "BoQ")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.aggregator(self.backbone(images))


def build_relation_teacher(name: str, repo: str, checkpoint: str) -> nn.Module:
    constructors = {"salad": SaladTeacher, "boq": BoQTeacher}
    if name not in constructors:
        raise ValueError(f"Strong relation teacher must be salad or boq, got {name!r}")
    model = constructors[name](repo, checkpoint)
    return model.eval().requires_grad_(False)
