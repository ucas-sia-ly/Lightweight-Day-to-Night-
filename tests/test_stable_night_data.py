import csv
import random

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from a2b_lightvpr.data import PairedVPRDataset, PKBalancedBatchSampler, StableNightVPRDataset
from a2b_lightvpr.transforms import IMAGENET_MEAN, IMAGENET_STD, PhotometricVPR, VPRTransform


def write_manifest(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def source_row(path, image_id="source0", place="place0", **extra):
    return {
        "dataset": "msls", "split": "train", "place_id": place,
        "image_id": image_id, "source_path": str(path), **extra,
    }


@pytest.mark.parametrize("mode", ["day_only", "day_aug", "day_fake", "day_aug_fake"])
def test_stable_night_four_modes_and_file_access(tmp_path, monkeypatch, mode):
    day_path, fake_path = tmp_path / "day.png", tmp_path / "fake.png"
    Image.new("RGB", (70, 28), (120, 145, 160)).save(day_path)
    fake_mode = mode in {"day_fake", "day_aug_fake"}
    row = source_row(day_path)
    if fake_mode:
        Image.new("RGB", (70, 56), (40, 55, 70)).save(fake_path)
        row["generated_path"] = str(fake_path)
    # Source-only modes need neither a generated CSV nor a generated_path field.
    manifest = write_manifest(tmp_path / ("generated.csv" if fake_mode else "source.csv"), [row])
    opened = []
    original_open = Image.open

    def record_open(path, *args, **kwargs):
        opened.append(str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", record_open)
    dataset = StableNightVPRDataset(manifest, VPRTransform(56, 70), min_class_images=1, train_domains=mode)
    sample = dataset[0]
    assert opened == [str(day_path)] + ([str(fake_path)] if fake_mode else [])
    assert sample["day"].shape == (3, 56, 70)
    assert sample["day_valid_mask"].dtype == torch.bool
    assert sample["day_valid_mask"].shape == (4, 5)
    assert not sample["day_valid_mask"][[0, 3]].any()
    assert sample["day_valid_mask"][1:3].all()
    assert ("aug" in sample) == (mode in {"day_aug", "day_aug_fake"})
    assert ("fake" in sample) == fake_mode
    if "aug" in sample:
        assert sample["aug"].shape == sample["day"].shape
        assert torch.equal(sample["aug"][:, :14], sample["day"][:, :14])
        assert torch.equal(sample["aug"][:, 42:], sample["day"][:, 42:])
    if fake_mode:
        assert sample["fake_valid_mask"].all()
        assert not torch.equal(sample["day_valid_mask"], sample["fake_valid_mask"])


def test_valid_mask_excludes_partial_padding_and_legacy_tensor_is_identical():
    image = Image.new("RGB", (56, 42), (90, 100, 110))
    transform = VPRTransform(56, 56)
    day, patches = transform.transform_with_mask(image)
    _, pixels = transform.transform_with_mask(image, patch_size=None)
    assert torch.equal(day, transform(image))
    assert pixels[7:49].all()
    assert not pixels[:7].any()
    assert not pixels[49:].any()
    assert patches.shape == (4, 4)
    assert patches[1:3].all()
    assert not patches[[0, 3]].any()


def test_photometric_changes_keep_pixel_locations_padding_and_input():
    pixels = np.arange(56 * 28 * 3, dtype=np.uint8).reshape(28, 56, 3)
    transform = VPRTransform(56, 56)
    tensor, valid = transform.transform_with_mask(Image.fromarray(pixels), patch_size=None)
    original = tensor.clone()
    augmentation = PhotometricVPR(
        exposure_stops=(0.0, 0.0), brightness=(0.85, 0.85), contrast=(1.0, 1.0),
        temperature=0.0, max_noise_std=0.0, blur_probability=0.0,
    )
    output = augmentation(tensor, valid)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(3, 1, 1)
    expected = ((tensor * std + mean).clamp(0, 1) * 0.85 - mean) / std
    assert torch.allclose(output[:, valid], expected[:, valid], atol=2e-6)
    assert torch.equal(output[:, ~valid], original[:, ~valid])
    assert torch.equal(tensor, original)
    assert output.data_ptr() != tensor.data_ptr()


def test_photometric_uses_reproducible_global_rng_for_noise_and_blur():
    tensor, valid = VPRTransform(56, 56).transform_with_mask(
        Image.new("RGB", (56, 28), (120, 130, 140)), patch_size=None
    )
    augmentation = PhotometricVPR(blur_probability=1.0)
    random.seed(17)
    torch.manual_seed(17)
    first = augmentation(tensor, valid)
    random.seed(17)
    torch.manual_seed(17)
    second = augmentation(tensor, valid)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert not torch.equal(first[:, valid], tensor[:, valid])
    assert torch.equal(first[:, ~valid], tensor[:, ~valid])


def test_photometric_local_rng_does_not_advance_global_states():
    tensor, valid = VPRTransform(56, 56).transform_with_mask(
        Image.new("RGB", (56, 28), (120, 130, 140)), patch_size=None
    )
    python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    augmentation = PhotometricVPR(blur_probability=1.0)
    first = augmentation(tensor, valid, rng=random.Random(123),
                         torch_generator=torch.Generator().manual_seed(123))
    second = augmentation(tensor, valid, rng=random.Random(123),
                          torch_generator=torch.Generator().manual_seed(123))
    assert torch.equal(first, second)
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_source_augmentation_is_deterministic_by_seed_source_and_epoch(tmp_path):
    day = tmp_path / "day.png"
    Image.new("RGB", (56, 28), (120, 130, 140)).save(day)
    alias = tmp_path / "alias.png"
    alias.symlink_to(day)
    manifest = write_manifest(tmp_path / "source.csv", [source_row(day)])
    alias_manifest = write_manifest(tmp_path / "alias.csv", [source_row(alias, image_id="changed-id")])
    arguments = dict(transform=VPRTransform(56, 56), min_class_images=1, train_domains="day_aug", random_seed=17)
    dataset = StableNightVPRDataset(manifest, **arguments)
    canonical_alias = StableNightVPRDataset(alias_manifest, **arguments)
    first = dataset[0]["aug"]
    random.seed(999)
    torch.manual_seed(999)
    python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    assert torch.equal(first, dataset[0]["aug"])
    assert torch.equal(first, canonical_alias[0]["aug"])
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    different_seed = StableNightVPRDataset(manifest, **{**arguments, "random_seed": 18})
    assert not torch.equal(first, different_seed[0]["aug"])
    dataset.set_epoch(1)
    second = dataset[0]["aug"]
    assert not torch.equal(first, second)
    # A freshly resumed dataset reproduces the same epoch without worker RNG
    # snapshots, and restoring an earlier epoch restores its augmented view.
    resumed = StableNightVPRDataset(manifest, **arguments)
    resumed.set_epoch(1)
    assert torch.equal(second, resumed[0]["aug"])
    dataset.set_epoch(0)
    assert torch.equal(first, dataset[0]["aug"])


def test_shared_epoch_reaches_spawned_persistent_data_worker(tmp_path):
    day = tmp_path / "day.png"
    Image.new("RGB", (56, 28), (120, 130, 140)).save(day)
    manifest = write_manifest(tmp_path / "source.csv", [source_row(day)])
    dataset = StableNightVPRDataset(manifest, VPRTransform(56, 56), min_class_images=1,
                                    train_domains="day_aug", random_seed=17)
    assert dataset._epoch.is_shared()
    loader = DataLoader(dataset, batch_size=1, num_workers=1, multiprocessing_context="spawn",
                        persistent_workers=True, prefetch_factor=1)
    try:
        first = next(iter(loader))["aug"][0]
        dataset.set_epoch(1)
        second = next(iter(loader))["aug"][0]
        assert not torch.equal(first, second)
        torch.testing.assert_close(second, dataset[0]["aug"], atol=1e-6, rtol=1e-6)
        dataset.set_epoch(0)
        restored = next(iter(loader))["aug"][0]
        assert torch.equal(first, restored)
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_multigenerator_sources_count_once_before_place_filter(tmp_path):
    first, second = tmp_path / "day0.png", tmp_path / "day1.png"
    for path in (first, second):
        Image.new("RGB", (28, 28), "white").save(path)
    alias = tmp_path / "day0_alias.png"
    alias.symlink_to(first)
    rows = [
        source_row(first, generated_path=str(tmp_path / "fake_a.png"), generator_sha256="a"),
        source_row(alias, image_id="another-generator-id", generated_path=str(tmp_path / "fake_b.png"), generator_sha256="b"),
        # Distinct sources remain distinct even if a generator reuses image_id.
        source_row(second, generated_path=str(tmp_path / "fake_c.png"), generator_sha256="a"),
    ]
    manifest = write_manifest(tmp_path / "multi_generator.csv", rows)
    dataset = StableNightVPRDataset(manifest, VPRTransform(28, 28), min_class_images=2, train_domains="day_fake")
    assert len(dataset) == 2
    assert dataset.rows[0]["generated_path"] == str(tmp_path / "fake_a.png")
    assert dataset.rows[0]["source_path"] == str(first.resolve())
    assert torch.equal(dataset.balanced_weights(), torch.tensor([0.5, 0.5], dtype=torch.double))
    sampler = PKBalancedBatchSampler(dataset.rows, {"msls": 2}, places_per_batch=1, images_per_place=2)
    batch = next(iter(sampler))
    assert len({dataset.rows[index]["source_path"] for index in batch}) == 2
    with pytest.raises(ValueError, match="place-class filter"):
        StableNightVPRDataset(manifest, VPRTransform(28, 28), min_class_images=3, train_domains="day_fake")
    # Legacy baselines retain their original row-based counting semantics.
    legacy = PairedVPRDataset(manifest, VPRTransform(28, 28), min_class_images=3, include_fake=False)
    assert len(legacy) == 3


def test_canonical_source_keys_ignore_generators_and_preserve_dataset_namespace(tmp_path):
    first, second = tmp_path / "day0.png", tmp_path / "day1.png"
    for path in (first, second):
        Image.new("RGB", (28, 28), "white").save(path)
    alias = tmp_path / "alias.png"
    alias.symlink_to(first)
    manifest_a = write_manifest(tmp_path / "a.csv", [source_row(first)])
    manifest_b = write_manifest(tmp_path / "b.csv", [source_row(alias, image_id="generator-changed-id")])
    manifest_c = write_manifest(tmp_path / "c.csv", [source_row(second)])
    manifest_d = write_manifest(tmp_path / "d.csv", [source_row(first, dataset="svox")])
    keys = [StableNightVPRDataset(path, VPRTransform(28, 28), min_class_images=1)[0]["image_key"].item()
            for path in (manifest_a, manifest_b, manifest_c, manifest_d)]
    assert keys[0] == keys[1]
    assert len({keys[0], keys[2], keys[3]}) == 3


@pytest.mark.parametrize("conflict", ["place", "split", "generator", "source"])
def test_stable_night_rejects_conflicting_identity_metadata(tmp_path, conflict):
    first = source_row(tmp_path / "day.png", generated_path=str(tmp_path / "fake.png"), generator_sha256="a")
    second = dict(first)
    if conflict == "place":
        second["place_id"] = "other"
    elif conflict == "split":
        second["split"] = "test"
    elif conflict == "generator":
        second["generator_sha256"] = "b"
    else:
        second["source_path"] = str(tmp_path / "different_day.png")
    manifest = write_manifest(tmp_path / "invalid.csv", [first, second])
    with pytest.raises(ValueError, match="[Cc]onflict|multiple source"):
        StableNightVPRDataset(manifest, VPRTransform(28, 28), min_class_images=1, train_domains="day_fake")
