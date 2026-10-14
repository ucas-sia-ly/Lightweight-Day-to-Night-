from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def _log_sinkhorn(
    z: torch.Tensor,
    log_mu: torch.Tensor,
    log_nu: torch.Tensor,
    iterations: int = 3,
) -> torch.Tensor:
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(z + u.unsqueeze(2), dim=1)
    return z + u.unsqueeze(2) + v.unsqueeze(1)


def _optimal_transport(
    scores: torch.Tensor, alpha: torch.Tensor, *, iterations: int = 3
) -> torch.Tensor:
    """SALAD transport with scalar or per-patch dustbin logits.

    Keep the original three-iteration/dtype path by default. The reliable head
    explicitly requests FP32 and more iterations. Marginals fix cluster mass to
    one each and dustbin mass to patches - clusters, irrespective of logits.
    """
    batch, clusters, patches = scores.shape
    if patches <= clusters:
        raise ValueError("SALAD transport requires more patches than clusters")
    if iterations < 1:
        raise ValueError("Sinkhorn iterations must be positive")
    one = scores.new_tensor(1)
    m, n = clusters * one, patches * one
    if alpha.ndim == 2:
        alpha = alpha.unsqueeze(1)
    try:
        bins = alpha.expand(batch, 1, patches)
    except RuntimeError as error:
        raise ValueError("Dustbin logits must be scalar, [P], [B,P], or [B,1,P]") from error
    couplings = torch.cat((scores, bins), dim=1)
    norm = -(m + n).log()
    log_mu = torch.cat((norm.expand(clusters), (n - m).log()[None] + norm))
    log_nu = norm.expand(patches)
    return _log_sinkhorn(
        couplings,
        log_mu[None].expand(batch, -1),
        log_nu[None].expand(batch, -1),
        iterations,
    ) - norm


class CompactSALADHead(nn.Module):
    """Compact form of serizba/salad's ``models/aggregators/salad.py``.

    The upstream training recipe uses dropout=0.3. Default to zero to preserve
    the existing project baseline; dropout adds no checkpoint parameters.
    """

    output_dim = 544

    def __init__(
        self,
        num_channels: int,
        num_clusters: int = 16,
        cluster_dim: int = 32,
        token_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not math.isfinite(dropout) or not 0.0 <= dropout <= 1.0:
            raise ValueError("SALAD dropout must be finite and between zero and one")
        self.num_clusters = int(num_clusters)
        self.cluster_dim = int(cluster_dim)
        self.token_dim = int(token_dim)
        self.token_features = nn.Sequential(
            nn.Linear(num_channels, 512), nn.ReLU(), nn.Linear(512, token_dim)
        )
        self.cluster_features = nn.Sequential(
            nn.Conv2d(num_channels, 512, 1),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(512, cluster_dim, 1),
        )
        self.score = nn.Sequential(
            nn.Conv2d(num_channels, 512, 1),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
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
        # Equivalent to upstream repeat/multiply/sum without the [B,C,K,P]
        # intermediates, which become expensive for larger metric batches.
        # Preserve multiply's dtype promotion under AMP rather than allowing
        # autocast to lower a FP32 transport solution inside the contraction.
        aggregation_dtype = torch.promote_types(local.dtype, assignments.dtype)
        with torch.autocast(device_type=features.device.type, enabled=False):
            aggregated = torch.einsum(
                "bcp,bkp->bck", local.to(aggregation_dtype), assignments.to(aggregation_dtype)
            )
        descriptor = torch.cat(
            (
                F.normalize(token, dim=-1),
                F.normalize(aggregated, dim=1).flatten(1),
            ),
            dim=-1,
        )
        return F.normalize(descriptor, dim=-1)


class ReliableSALADHead(CompactSALADHead):
    """Compact SALAD plus a lightweight, opt-in patch reliability branch.

    Dynamic logits redistribute the fixed dustbin capacity; they do not change
    it. Normalizing each real cluster in log space keeps its aggregation mass
    nonzero even under a poorly converged or extreme transport solution.
    """

    def __init__(
        self,
        num_channels: int,
        *,
        dustbin_enabled: bool = True,
        gamma_init: float = 1.0,
        gamma_max: float = 4.0,
        sinkhorn_iterations: int = 20,
    ):
        super().__init__(num_channels)
        if not math.isfinite(gamma_max) or not 0 < gamma_init < gamma_max:
            raise ValueError("Dustbin gamma requires 0 < gamma_init < finite gamma_max")
        if sinkhorn_iterations < 1:
            raise ValueError("Sinkhorn iterations must be positive")
        self.dustbin_enabled = bool(dustbin_enabled)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self._gamma_max_setting = float(gamma_max)
        self.register_buffer("gamma_max", torch.tensor(float(gamma_max)))
        self.gamma_logit = nn.Parameter(torch.tensor(math.log(gamma_init / (gamma_max - gamma_init))))
        self.reliability_predictor = nn.Sequential(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    @property
    def gamma(self) -> torch.Tensor:
        return self.gamma_max * self.gamma_logit.sigmoid()

    def get_extra_state(self):
        # These settings affect retrieval but are not trainable tensors. Persist
        # them inside model state so export and strict loading preserve behavior.
        return {
            "version": 1,
            "dustbin_enabled": self.dustbin_enabled,
            "gamma_max": self._gamma_max_setting,
            "sinkhorn_iterations": self.sinkhorn_iterations,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("StableNight head settings do not match the checkpoint")

    def forward(
        self,
        value: tuple[torch.Tensor, torch.Tensor],
        *,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features, token = value
        reliability = self.reliability_predictor(features.flatten(2).transpose(1, 2)).squeeze(-1).sigmoid()
        local = self.cluster_features(features).flatten(2)
        scores = self.score(features).flatten(2)
        token = self.token_features(token)
        # AMP must not lower the precision of logsumexp or weighted aggregation.
        with torch.autocast(device_type=features.device.type, enabled=False):
            bins = self.dust_bin.float()
            if self.dustbin_enabled:
                bins = bins + self.gamma.float() * (1.0 - reliability.float())
            log_assignment = _optimal_transport(
                scores.float(), bins, iterations=self.sinkhorn_iterations
            )
            cluster_weights = log_assignment[:, :-1].softmax(dim=-1)
            aggregated = torch.einsum("bcp,bkp->bck", local.float(), cluster_weights)
            descriptor = F.normalize(torch.cat((
                F.normalize(token.float(), dim=-1),
                F.normalize(aggregated, dim=1).flatten(1),
            ), dim=-1), dim=-1)
        if not return_aux:
            return descriptor
        return descriptor, {
            "reliability": reliability,
            "dustbin_assignment": log_assignment[:, -1].exp(),
            "cluster_mass": log_assignment[:, :-1].exp().sum(dim=-1),
            "local_features": features,
        }


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
        *,
        dropout: float = 0.0,
    ):
        super().__init__()
        if model_name not in {"dinov2_vits14", "dinov2_vits14_reg"}:
            raise ValueError("Compact SALAD requires a DINOv2-S/14 backbone")
        self.backbone = DINOv2SALADBackbone(
            repo, model_name, checkpoint, trainable_blocks
        )
        if self.backbone.num_channels != 384:
            raise ValueError("Compact SALAD requires 384-channel DINOv2-S features")
        self.aggregator = CompactSALADHead(self.backbone.num_channels, dropout=dropout)

    def set_backbone_trainable(self, enabled: bool) -> None:
        self.backbone.set_trainable(enabled)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if (images.ndim != 4 or images.shape[1] != 3
                or any(value <= 0 or value % 14 for value in images.shape[-2:])
                or (images.shape[-2] // 14) * (images.shape[-1] // 14) <= self.aggregator.num_clusters):
            raise ValueError("Compact SALAD images require RGB BCHW, multiples of 14, and more patches than clusters")
        descriptor = self.aggregator(self.backbone(images))
        if descriptor.shape[1] != self.descriptor_dim:
            raise RuntimeError(
                f"Descriptor contract violated: expected {self.descriptor_dim}, "
                f"got {descriptor.shape[1]}"
            )
        return F.normalize(descriptor, p=2, dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)


class StableNightSALADModel(SALADStudentModel):
    """Stage-one StableNight-SALAD; reference teachers live only in training."""

    def __init__(
        self, repo: str, model_name: str, checkpoint: str,
        trainable_blocks: int = 0, **head_options,
    ):
        super().__init__(repo, model_name, checkpoint, trainable_blocks)
        self.aggregator = ReliableSALADHead(self.backbone.num_channels, **head_options)

    def encode_with_aux(self, images: torch.Tensor):
        return self.aggregator(self.backbone(images), return_aux=True)
