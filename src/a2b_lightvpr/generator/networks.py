from __future__ import annotations

import functools

import torch
from torch import nn


def _norm(channels: int) -> nn.Module:
    return nn.InstanceNorm2d(channels, affine=False, track_running_stats=False)


class ResnetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            _norm(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            _norm(channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class ResnetGenerator(nn.Module):
    """Shared fully-convolutional ResNet-9 generator with feature taps."""

    feature_layers = (0, 1, 2, 5, 8)

    def __init__(self, channels: int = 64, blocks: int = 9):
        super().__init__()
        modules: list[nn.Module] = [
            nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(3, channels, 7), _norm(channels), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(channels, channels * 2, 3, 2, 1), _norm(channels * 2), nn.ReLU(True)),
            nn.Sequential(nn.Conv2d(channels * 2, channels * 4, 3, 2, 1), _norm(channels * 4), nn.ReLU(True)),
        ]
        modules.extend(ResnetBlock(channels * 4) for _ in range(blocks))
        self.encoder = nn.ModuleList(modules)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels * 4, channels * 2, 3),
            _norm(channels * 2),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels * 2, channels, 3),
            _norm(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, 3, 7),
            nn.Tanh(),
        )
        self.apply(init_weights)

    def encode_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features: list[torch.Tensor] = []
        value = inputs
        for index, layer in enumerate(self.encoder):
            value = layer(value)
            if index in self.feature_layers:
                features.append(value)
        return value, features

    def forward(self, inputs: torch.Tensor, *, return_features: bool = False):
        latent, features = self.encode_features(inputs)
        output = self.decoder(latent)
        return (output, features) if return_features else output


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator used by all content objectives."""

    def __init__(self, channels: int = 64, layers: int = 3):
        super().__init__()
        sequence: list[nn.Module] = [nn.Conv2d(3, channels, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        multiplier = 1
        for layer_index in range(1, layers):
            previous = multiplier
            multiplier = min(2**layer_index, 8)
            sequence.extend(
                [
                    nn.Conv2d(channels * previous, channels * multiplier, 4, 2, 1, bias=False),
                    nn.InstanceNorm2d(channels * multiplier),
                    nn.LeakyReLU(0.2, True),
                ]
            )
        previous = multiplier
        multiplier = min(2**layers, 8)
        sequence.extend(
            [
                nn.Conv2d(channels * previous, channels * multiplier, 4, 1, 1, bias=False),
                nn.InstanceNorm2d(channels * multiplier),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(channels * multiplier, 1, 4, 1, 1),
            ]
        )
        self.model = nn.Sequential(*sequence)
        self.apply(init_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def lsgan_discriminator_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((real - 1.0).square().mean() + fake.square().mean())


def lsgan_generator_loss(fake: torch.Tensor) -> torch.Tensor:
    return (fake - 1.0).square().mean()

