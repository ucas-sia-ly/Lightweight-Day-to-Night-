import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from a2b_lightvpr.config import DataConfig
from a2b_lightvpr.data import PairedVPRDataset, RollingBalancedSampler, audit_data
from a2b_lightvpr.transforms import VPRTransform


def write_fixture(root: Path, *, split="train"):
    generator = root / "generator.csv"
    with generator.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "dataset", "split", "route", "paired", "source_path", "target_path"])
        writer.writeheader()
        writer.writerow({"pair_id": "1", "dataset": "robotcar", "split": "train", "route": "train-route", "paired": "0", "source_path": __file__, "target_path": __file__})
    sources = root / "source.csv"
    with sources.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "dataset", "split", "place_id", "source_path"])
        writer.writeheader()
        writer.writerow({"image_id": "i", "dataset": "robotcar", "split": split, "place_id": "p", "source_path": __file__})
    return DataConfig(robotcar_seasons_root=str(root / "missing-public-test"),
                      generator_manifest=str(generator), vpr_source_manifest=str(sources),
                      generated_root=str(root / "generated"))


def test_training_audit_does_not_require_any_robotcar_test_protocol(tmp_path):
    result = audit_data(write_fixture(tmp_path))
    assert result["generator_train_rows"] == result["vpr_train_rows"] == 1
    assert result["generator_datasets"] == {"robotcar": 1}
    assert not result["robotcar_seasons_available"]
    assert "protocol_dataset" not in result and "robotcar_counts" not in result


def test_source_only_training_audit_does_not_require_generator_metadata(tmp_path):
    config = replace(write_fixture(tmp_path), generator_manifest="")
    (tmp_path / "generator.csv").unlink()
    result = audit_data(config)
    assert result["generator_train_rows"] == 0
    assert result["generator_datasets"] == {}
    assert result["vpr_train_rows"] == 1
    assert result["vpr_datasets"] == {"robotcar": 1}


def test_training_audit_rejects_non_train_source_rows(tmp_path):
    with pytest.raises(ValueError, match="non-train"):
        audit_data(write_fixture(tmp_path, split="test"))


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
