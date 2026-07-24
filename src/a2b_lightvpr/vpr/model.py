from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = inputs.clamp_min(self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(value, 1).pow(1.0 / self.p)


class GeMFC(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.gem = GeM()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = F.normalize(inputs, dim=1)
        return self.fc(self.gem(value).flatten(1))


class ConvAP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.project = nn.Conv2d(input_dim, 128, 1)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pool(self.project(inputs)).flatten(1)


class FeatureMixer(nn.Module):
    def __init__(self, token_count: int, channels: int):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(token_count),
            nn.Linear(token_count, token_count),
            nn.ReLU(True),
            nn.Linear(token_count, token_count),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.mix(inputs)


class MixVPR(nn.Module):
    def __init__(self, input_dim: int, grid: tuple[int, int], output_dim: int = 512):
        super().__init__()
        dimensions = {512: (256, 2), 4096: (1024, 4)}
        if output_dim not in dimensions:
            raise ValueError("Official MixVPR output must be 512 or 4096")
        output_channels, output_rows = dimensions[output_dim]
        tokens = int(grid[0] * grid[1])
        self.grid = grid
        self.mix = nn.Sequential(*(FeatureMixer(tokens, input_dim) for _ in range(4)))
        self.channel_proj = nn.Linear(input_dim, output_channels)
        self.row_proj = nn.Linear(tokens, output_rows)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = F.adaptive_avg_pool2d(inputs, self.grid).flatten(2)
        value = self.mix(value)
        value = self.channel_proj(value.transpose(1, 2)).transpose(1, 2)
        value = self.row_proj(value)
        return value.flatten(1)


def _feature_grid(backbone: nn.Module, height: int, width: int) -> tuple[int, int]:
    training = backbone.training
    backbone.eval()
    with torch.no_grad():
        output = backbone(torch.zeros(1, 3, height, width))
    backbone.train(training)
    return int(output.shape[-2]), int(output.shape[-1])


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int, int]:
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        network = models.resnet50(weights=weights)
        return nn.Sequential(*list(network.children())[:-2]), 2048, 32
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        network = models.efficientnet_b0(weights=weights)
        return network.features, 1280, 32
    if name in {"regnet_y_800mf", "regnet_y_1_6gf"}:
        constructor, weights_type = {
            "regnet_y_800mf": (models.regnet_y_800mf, models.RegNet_Y_800MF_Weights),
            "regnet_y_1_6gf": (models.regnet_y_1_6gf, models.RegNet_Y_1_6GF_Weights),
        }[name]
        network = constructor(weights=weights_type.IMAGENET1K_V2 if pretrained else None)
        return nn.Sequential(network.stem, network.trunk_output), network.fc.in_features, 32
    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        network = models.convnext_tiny(weights=weights)
        return network.features, network.classifier[-1].in_features, 32
    raise ValueError(f"Unsupported backbone: {name}")


class DescriptorModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        aggregator: nn.Module,
        descriptor_dim: int = 512,
        backbone_name: str = "",
        aggregator_name: str = "",
    ):
        super().__init__()
        self.backbone = backbone
        self.aggregator = aggregator
        self.descriptor_dim = int(descriptor_dim)
        self.backbone_name = backbone_name
        self.aggregator_name = aggregator_name

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        value = self.aggregator(self.backbone(images))
        if value.shape[1] != self.descriptor_dim:
            raise RuntimeError(
                f"Descriptor contract violated: expected {self.descriptor_dim}, got {value.shape[1]}"
            )
        return F.normalize(value, dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)


def build_descriptor_model(
    backbone_name: str,
    aggregator_name: str,
    *,
    image_height: int,
    image_width: int,
    descriptor_dim: int = 512,
    pretrained: bool = True,
    init_checkpoint: str = "",
    init_checkpoint_scope: str = "full",
    model_family: str = "custom",
) -> DescriptorModel:
    if model_family == "mixvpr":
        if (image_height, image_width) != (320, 320):
            raise ValueError("Official MixVPR requires 320x320 input")
        model = OfficialMixVPRModel(descriptor_dim)
        if not init_checkpoint:
            raise ValueError("Official MixVPR requires a checkpoint")
        _strict_load_state(model, init_checkpoint)
        return model
    backbone, channels, _ = _build_backbone(backbone_name, pretrained)
    if aggregator_name == "gem":
        aggregator = GeMFC(channels, descriptor_dim)
    elif aggregator_name == "convap":
        aggregator = ConvAP(channels)
    elif aggregator_name == "mixvpr":
        grid = _feature_grid(backbone, image_height, image_width)
        aggregator = MixVPR(channels, grid, descriptor_dim)
    else:
        raise ValueError(f"Unsupported aggregator: {aggregator_name}")
    model = DescriptorModel(
        backbone, aggregator, descriptor_dim, backbone_name, aggregator_name
    )
    if init_checkpoint:
        result = load_initial_weights(model, init_checkpoint, init_checkpoint_scope)
        if init_checkpoint_scope == "full" and (result["missing"] or result["unexpected"]):
            raise ValueError(f"Strict initial checkpoint mismatch: {result}")
    return model


class CroppedResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(weights=None)
        self.model.avgpool = nn.Identity()
        self.model.fc = nn.Identity()
        self.model.layer4 = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = self.model.conv1(inputs)
        value = self.model.bn1(value)
        value = self.model.relu(value)
        value = self.model.maxpool(value)
        value = self.model.layer1(value)
        value = self.model.layer2(value)
        return self.model.layer3(value)


class OfficialMixVPRModel(nn.Module):
    def __init__(self, descriptor_dim: int):
        super().__init__()
        self.backbone = CroppedResNet50()
        self.aggregator = MixVPR(1024, (20, 20), descriptor_dim)
        self.descriptor_dim = int(descriptor_dim)
        self.backbone_name = "resnet50"
        self.aggregator_name = "mixvpr"

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        value = self.aggregator(self.backbone(images))
        if value.shape[1] != self.descriptor_dim:
            raise RuntimeError(
                f"Descriptor contract violated: expected {self.descriptor_dim}, got {value.shape[1]}"
            )
        return F.normalize(value, dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)


def _checkpoint_state(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dict")
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def _strict_load_state(model: nn.Module, checkpoint: str | Path) -> None:
    result = model.load_state_dict(_checkpoint_state(checkpoint), strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(f"Strict checkpoint mismatch: {result}")


def eigenplaces_checkpoint(descriptor_dim: int) -> Path:
    if descriptor_dim not in {512, 2048}:
        raise ValueError("EigenPlaces adaptation supports 512 or 2048 dimensions")
    filename = f"ResNet50_{descriptor_dim}_eigenplaces.pth"
    url = f"https://github.com/gmberton/EigenPlaces/releases/download/v1.0/{filename}"
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "github.com" or not parsed.path.startswith(
        "/gmberton/EigenPlaces/releases/download/v1.0/"
    ):
        raise ValueError("Refusing a non-official EigenPlaces checkpoint URL")
    state = torch.hub.load_state_dict_from_url(url, map_location="cpu", file_name=filename)
    cache_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not cache_path.is_file() or not isinstance(state, dict):
        raise RuntimeError("EigenPlaces checkpoint download did not complete")
    digest = hashlib.sha256()
    with cache_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {
        "url": url,
        "file": str(cache_path.resolve()),
        "size_bytes": cache_path.stat().st_size,
        "sha256": digest.hexdigest(),
    }
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)
    return cache_path


def resolve_initial_checkpoint(value: str, descriptor_dim: int) -> str:
    if value == "eigenplaces://resnet50":
        return str(eigenplaces_checkpoint(descriptor_dim))
    return value


def load_initial_weights(
    model: DescriptorModel, checkpoint: str | Path, scope: str = "full"
) -> dict[str, list[str]]:
    state = _checkpoint_state(checkpoint)
    mapped = {}
    for key, value in state.items():
        if key.startswith("aggregation.1.") and model.aggregator_name == "gem":
            mapped["aggregator.gem." + key.removeprefix("aggregation.1.")] = value
        elif key.startswith("aggregation.3.") and model.aggregator_name == "gem":
            mapped["aggregator.fc." + key.removeprefix("aggregation.3.")] = value
        elif key.startswith("backbone."):
            mapped[key] = value
        elif key.startswith("aggregator.") and scope == "full":
            mapped[key] = value
    result = model.load_state_dict(mapped, strict=False)
    return {"missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)}
