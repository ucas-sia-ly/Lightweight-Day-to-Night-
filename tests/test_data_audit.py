import csv
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from a2b_lightvpr.config import DataConfig
from a2b_lightvpr.data import PairedVPRDataset, RollingBalancedSampler, audit_data
from a2b_lightvpr.prepare import EXPECTED_VPR_ELIGIBLE, _parse_svox_train_path
from a2b_lightvpr.transforms import VPRTransform


FIELDS = [
    "split", "role", "protocol_role", "domain", "traverse", "timestamp",
    "easting", "northing", "yaw_rad", "yaw_deg", "source_path", "output_path",
]


def write_fixture(root: Path, leak=False):
    robot = root / "robot.csv"
    with robot.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, split in enumerate(("train", "val", "test")):
            row = dict.fromkeys(FIELDS, "")
            row.update(
                split=split,
                role="database",
                domain="day",
                traverse="same" if leak else split,
                timestamp=str(index),
                easting="0",
                northing="0",
                yaw_deg="0",
                source_path=__file__,
            )
            writer.writerow(row)
    external = root / "external.csv"
    with external.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "dataset", "split", "route", "paired", "source_path", "target_path"])
        writer.writeheader()
        writer.writerow({"pair_id": "1", "dataset": "x", "split": "train", "route": "r", "paired": "0", "source_path": __file__, "target_path": __file__})
    protocol = root / "protocol.json"
    protocol.write_text(json.dumps({"dataset": "fixture"}))
    return DataConfig(str(robot), str(protocol), str(external))


def test_audit_accepts_disjoint_traversals(tmp_path):
    result = audit_data(write_fixture(tmp_path))
    assert result["generator_train_rows"] == 1


def test_audit_rejects_traversal_leakage(tmp_path):
    with pytest.raises(ValueError, match="leakage"):
        audit_data(write_fixture(tmp_path, leak=True))


def test_vpr_training_rejects_non_train_generated_rows(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image_path)
    manifest = tmp_path / "generated.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "place_id", "source_path", "generated_path", "image_id"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "val",
                "place_id": "0:0:0",
                "source_path": image_path,
                "generated_path": image_path,
                "image_id": "x",
            }
        )
    with pytest.raises(ValueError, match="non-train"):
        PairedVPRDataset(manifest, VPRTransform(8, 8), min_class_images=1)


def test_vpr_dataset_sampling_weights_match_config(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image_path)
    manifest = tmp_path / "generated.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "dataset", "place_id", "source_path", "generated_path", "image_id"],
        )
        writer.writeheader()
        for dataset in ("msls", "robotcar", "svox"):
            for index in range(4):
                writer.writerow({
                    "split": "train", "dataset": dataset, "place_id": "place",
                    "source_path": image_path, "generated_path": image_path,
                    "image_id": f"{dataset}-{index}",
                })
    expected = {"msls": 1 / 3, "robotcar": 1 / 3, "svox": 1 / 3}
    dataset = PairedVPRDataset(
        manifest, VPRTransform(8, 8), min_class_images=2,
        dataset_sampling_weights=expected,
    )
    weights = dataset.balanced_weights()
    draws = torch.multinomial(weights, 100000, replacement=True)
    counts = {name: 0 for name in expected}
    for index in draws.tolist():
        counts[dataset.rows[index]["dataset"]] += 1
    for name, probability in expected.items():
        assert counts[name] / 100000 == pytest.approx(probability, abs=0.01)


def test_frozen_extended_place_statistics():
    assert EXPECTED_VPR_ELIGIBLE == {
        "msls": {"places": 18960, "rows": 75840},
        "robotcar": {"places": 1464, "rows": 8947},
        "svox": {"places": 9436, "rows": 33526},
    }


@pytest.mark.parametrize("direction", ["F", "B"])
def test_svox_filename_records_camera_direction(direction):
    path = Path(f"@0621503.83@5736826.91@201205@panorama@{direction}@.jpg")
    assert _parse_svox_train_path(path) == (621503.83, 5736826.91, direction)


def test_rolling_sampler_is_balanced_unique_and_resume_deterministic():
    rows = []
    for dataset, count in (("a", 4), ("b", 2)):
        for index in range(count):
            rows.append({"dataset": dataset, "global_place_id": f"{dataset}:p{index // 2}"})
    sampler = RollingBalancedSampler(rows, {"a": 3, "b": 2})
    first = list(sampler)
    assert len(first) == 5
    assert len(first) == len(set(first))
    assert first == list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first != second
    assert len(second) == len(set(second))
    resumed = RollingBalancedSampler(rows, {"a": 3, "b": 2})
    resumed.set_epoch(1)
    assert second == list(resumed)
    stats = sampler.statistics()
    assert stats["a"]["quota"] == 3
    assert stats["b"]["image_coverage_ratio"] == 1.0


def test_rolling_sampler_repeats_only_after_small_dataset_is_covered():
    rows = [
        {"dataset": "small", "global_place_id": f"small:p{index}"}
        for index in range(3)
    ]
    sampler = RollingBalancedSampler(rows, {"small": 7})
    window = sampler._window("small", 0)
    values = list(sampler)
    assert len(values) == 7
    assert set(window[:3]) == {0, 1, 2}
    assert set(values) == {0, 1, 2}
    assert sampler.statistics()["small"]["image_coverage_ratio"] == 1.0


def test_rolling_sampler_windows_stay_unique_across_many_cycle_boundaries():
    rows = [
        {"dataset": "svox", "global_place_id": f"svox:p{index // 2}"}
        for index in range(34)
    ]
    sampler = RollingBalancedSampler(rows, {"svox": 20})
    for epoch in range(20):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        assert len(indices) == 20
        assert len(set(indices)) == 20

        resumed = RollingBalancedSampler(rows, {"svox": 20})
        resumed.set_epoch(epoch)
        assert indices == list(resumed)
