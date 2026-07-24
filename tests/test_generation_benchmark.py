from PIL import Image

from a2b_lightvpr.transforms import nocplace_paper_crop


def test_paper_crop_resizes_short_side_then_center_crops():
    image = Image.new("RGB", (1280, 720))
    assert nocplace_paper_crop(image, short_side=640, crop_size=512).size == (512, 512)
