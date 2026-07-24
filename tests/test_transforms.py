import random

import numpy as np
import torch
from PIL import Image

from a2b_lightvpr.transforms import GeneratorTrainTransform, VPRTransform, resize_to_fit


def image(width=160, height=90):
    return Image.fromarray(np.full((height, width, 3), 127, dtype=np.uint8))


def test_fit_pad_preserves_full_aspect_ratio():
    tensor = VPRTransform(256, 384)(image())
    assert tensor.shape == (3, 256, 384)
    # A 16:9 image fills the width and is padded vertically, never cropped.
    assert tensor[:, 0].abs().max().item() < 1e-5


def test_generator_resize_is_unpadded_and_multiple_of_four():
    resized = resize_to_fit(image(1280, 960), 320, 480)
    assert resized.size == (424, 320)
    assert resized.width % 4 == resized.height % 4 == 0


def test_paired_generator_transform_shares_geometry():
    pixels = np.arange(20 * 12 * 3, dtype=np.uint8).reshape(12, 20, 3)
    source = Image.fromarray(pixels)
    first, second = GeneratorTrainTransform(12, 8).paired(
        source, source, rng=random.Random(7)
    )
    assert torch.equal(first, second)
