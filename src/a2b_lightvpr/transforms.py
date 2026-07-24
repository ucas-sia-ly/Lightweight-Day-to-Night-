from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _scaled_size(width: int, height: int, scale: float) -> tuple[int, int]:
    return max(1, round(height * scale)), max(1, round(width * scale))


def resize_short_side(image: Image.Image, short_side: int) -> Image.Image:
    width, height = image.size
    scale = float(short_side) / float(min(width, height))
    size = _scaled_size(width, height, scale)
    return TF.resize(image, size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True)


def nocplace_paper_crop(
    image: Image.Image, short_side: int = 640, crop_size: int = 512
) -> Image.Image:
    """Deterministic counterpart of NocPlace's short-side resize and 512 crop."""
    image = resize_short_side(image.convert("RGB"), short_side)
    width, height = image.size
    if min(width, height) < crop_size:
        raise ValueError("Crop exceeds resized image")
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return TF.crop(image, top, left, crop_size, crop_size)


@dataclass(frozen=True)
class GeneratorTrainTransform:
    load_short_side: int = 352
    crop_size: int = 320

    def __call__(self, image: Image.Image, *, rng: random.Random | None = None) -> torch.Tensor:
        rng = rng or random
        image = resize_short_side(image.convert("RGB"), self.load_short_side)
        width, height = image.size
        if min(width, height) < self.crop_size:
            raise ValueError("Generator crop is larger than the resized image")
        top = rng.randint(0, height - self.crop_size)
        left = rng.randint(0, width - self.crop_size)
        image = TF.crop(image, top, left, self.crop_size, self.crop_size)
        if rng.random() < 0.5:
            image = TF.hflip(image)
        return TF.normalize(TF.to_tensor(image), (0.5,) * 3, (0.5,) * 3)

    def paired(self, first: Image.Image, second: Image.Image, *, rng: random.Random | None = None):
        """Apply identical geometry to an aligned day/night pair."""
        rng = rng or random
        first = resize_short_side(first.convert("RGB"), self.load_short_side)
        second = TF.resize(
            second.convert("RGB"), [first.height, first.width],
            interpolation=TF.InterpolationMode.BICUBIC, antialias=True,
        )
        top = rng.randint(0, first.height - self.crop_size)
        left = rng.randint(0, first.width - self.crop_size)
        flip = rng.random() < 0.5
        outputs = []
        for image in (first, second):
            image = TF.crop(image, top, left, self.crop_size, self.crop_size)
            if flip:
                image = TF.hflip(image)
            outputs.append(TF.normalize(TF.to_tensor(image), (0.5,) * 3, (0.5,) * 3))
        return tuple(outputs)


def resize_to_fit(image: Image.Image, max_height: int, max_width: int, multiple: int = 4) -> Image.Image:
    width, height = image.size
    scale = min(float(max_width) / width, float(max_height) / height)
    new_height, new_width = _scaled_size(width, height, scale)
    new_height = max(multiple, new_height // multiple * multiple)
    new_width = max(multiple, new_width // multiple * multiple)
    return TF.resize(
        image.convert("RGB"),
        [new_height, new_width],
        interpolation=TF.InterpolationMode.BICUBIC,
        antialias=True,
    )


def generator_inference_tensor(image: Image.Image, max_height: int, max_width: int) -> torch.Tensor:
    image = resize_to_fit(image, max_height, max_width)
    return TF.normalize(TF.to_tensor(image), (0.5,) * 3, (0.5,) * 3)


@dataclass(frozen=True)
class VPRTransform:
    height: int
    width: int

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        source_width, source_height = image.size
        scale = min(self.width / source_width, self.height / source_height)
        resized_height, resized_width = _scaled_size(source_width, source_height, scale)
        tensor = TF.to_tensor(
            TF.resize(
                image,
                [resized_height, resized_width],
                interpolation=TF.InterpolationMode.BICUBIC,
                antialias=True,
            )
        )
        mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        canvas = mean.expand(3, self.height, self.width).clone()
        top = (self.height - resized_height) // 2
        left = (self.width - resized_width) // 2
        canvas[:, top : top + resized_height, left : left + resized_width] = tensor
        return TF.normalize(canvas, IMAGENET_MEAN, IMAGENET_STD)


def denormalize_generator(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor.float() + 1.0) * 0.5).clamp(0.0, 1.0)


def normalized_generator_resize(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(tensor, (height, width), mode="bilinear", align_corners=False, antialias=True)
