import csv
import math

from PIL import Image

from a2b_lightvpr.data import PairedVPRDataset, optional_float
from a2b_lightvpr.transforms import VPRTransform


def test_day_only_dataset_does_not_open_generated_image(tmp_path):
    day = tmp_path / "day.jpg"
    Image.new("RGB", (16, 12), "white").save(day)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "dataset", "split", "place_id", "image_id", "source_path", "generated_path",
        ])
        writer.writeheader()
        writer.writerow({
            "dataset": "msls",
            "split": "train",
            "place_id": "p0",
            "image_id": "i0",
            "source_path": str(day),
            "generated_path": str(tmp_path / "missing-fake.jpg"),
        })
    dataset = PairedVPRDataset(
        manifest,
        VPRTransform(8, 8),
        min_class_images=1,
        datasets=("msls",),
        include_fake=False,
    )
    sample = dataset[0]
    assert tuple(sample["day"].shape) == (3, 8, 8)
    assert "fake" not in sample


def test_optional_pose_values_accept_svox_direction_codes():
    assert optional_float("12.5") == 12.5
    assert math.isnan(optional_float(""))
    assert math.isnan(optional_float(None))
    assert math.isnan(optional_float("B"))
    assert math.isnan(optional_float("F"))
