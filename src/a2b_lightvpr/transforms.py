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
    resize_mode: str = "fit_pad"

    def __post_init__(self) -> None:
        if self.resize_mode not in {"fit_pad", "resize"}:
            raise ValueError("VPR resize_mode must be fit_pad or resize")

    @property
    def transform_kind(self) -> str:
        return "direct-resize" if self.resize_mode == "resize" else "fit-and-pad"

    @property
    def padding(self) -> str:
        return "none" if self.resize_mode == "resize" else "imagenet-mean-centered"

    def _fit_canvas(
        self, image: Image.Image, return_mask: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        image = image.convert("RGB")
        if self.resize_mode == "resize":
            tensor = TF.to_tensor(TF.resize(
                image, [self.height, self.width],
                interpolation=TF.InterpolationMode.BICUBIC, antialias=True,
            ))
            valid = torch.ones((self.height, self.width), dtype=torch.bool) if return_mask else None
            return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD), valid
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
        valid = None
        if return_mask:
            valid = torch.zeros((self.height, self.width), dtype=torch.bool)
            valid[top : top + resized_height, left : left + resized_width] = True
        return TF.normalize(canvas, IMAGENET_MEAN, IMAGENET_STD), valid

    def __call__(self, image: Image.Image) -> torch.Tensor:
        # The default preserves the existing tensor-only fit-and-pad behavior.
        return self._fit_canvas(image)[0]

    def transform_with_mask(
        self, image: Image.Image, patch_size: int | None = 14
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the same image plus valid pixels or wholly unpadded patches.

        This mask describes canvas geometry only. It does not certify spatial
        correspondence between a source image and a generator's output.
        """
        tensor, valid = self._fit_canvas(image, return_mask=True)
        assert valid is not None
        return tensor, valid if patch_size is None else patch_valid_mask(valid, patch_size)


def patch_valid_mask(valid_pixels: torch.Tensor, patch_size: int = 14) -> torch.Tensor:
    """Exclude every patch that contains padding, including partial boundaries."""
    if patch_size < 1:
        raise ValueError("patch_size must be positive")
    if valid_pixels.ndim != 2:
        raise ValueError("valid_pixels must have shape [H, W]")
    if min(valid_pixels.shape) < patch_size:
        raise ValueError("Image canvas must contain at least one full patch")
    fractions = F.avg_pool2d(valid_pixels.float()[None, None], patch_size, patch_size)
    return fractions[0, 0] == 1.0


@dataclass(frozen=True)
class PhotometricVPR:
    """Moderate appearance variation without cropping or moving scene content.

    Operates on an ImageNet-normalized fitted canvas. Optional local Python and
    torch RNGs support reproducible dataset augmentation without advancing global
    state. Only the content rectangle changes; padding is restored exactly.
    """

    exposure_stops: tuple[float, float] = (-0.8, 0.5)
    brightness: tuple[float, float] = (0.8, 1.15)
    contrast: tuple[float, float] = (0.8, 1.2)
    temperature: float = 0.12
    max_noise_std: float = 0.02
    blur_probability: float = 0.25

    def __call__(
        self,
        tensor: torch.Tensor,
        valid_pixels: torch.Tensor,
        *,
        rng: random.Random | None = None,
        torch_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        rng = rng if rng is not None else random
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError("PhotometricVPR expects a [3, H, W] image")
        if valid_pixels.shape != tensor.shape[-2:] or valid_pixels.dtype != torch.bool:
            raise ValueError("PhotometricVPR expects a boolean [H, W] validity mask")
        if not valid_pixels.any():
            return tensor.clone()
        # Working on the fitted content alone keeps blur/noise out of padding.
        positions = valid_pixels.nonzero(as_tuple=False)
        top, left = positions.amin(dim=0).tolist()
        bottom, right = (positions.amax(dim=0) + 1).tolist()
        mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
        content = (tensor[:, top:bottom, left:right].float() * std + mean).clamp(0, 1)
        content = content * (2.0 ** rng.uniform(*self.exposure_stops))
        content = content * rng.uniform(*self.brightness)
        center = content.mean(dim=(-2, -1), keepdim=True)
        content = center + rng.uniform(*self.contrast) * (content - center)
        warmth = rng.uniform(-self.temperature, self.temperature)
        gains = content.new_tensor((1.0 + warmth, 1.0, 1.0 - warmth)).view(3, 1, 1)
        content = content * gains
        if min(content.shape[-2:]) > 1 and rng.random() < self.blur_probability:
            content = TF.gaussian_blur(content, 3, sigma=rng.uniform(0.1, 0.8))
        noise_std = rng.uniform(0.0, self.max_noise_std)
        if noise_std:
            noise = torch.randn(content.shape, dtype=content.dtype, device=content.device,
                                generator=torch_generator)
            content = content + noise * noise_std
        normalized = ((content.clamp(0, 1) - mean) / std).to(tensor.dtype)
        output = tensor.clone()
        output[:, top:bottom, left:right] = normalized
        # This also handles masks with internal invalid pixels without changing
        # the input tensor or overwriting their original values.
        return torch.where(valid_pixels[None], output, tensor)


def denormalize_generator(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor.float() + 1.0) * 0.5).clamp(0.0, 1.0)


def normalized_generator_resize(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(tensor, (height, width), mode="bilinear", align_corners=False, antialias=True)
