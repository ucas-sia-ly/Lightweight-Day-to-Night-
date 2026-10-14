import csv
import random
from pathlib import Path

import pytest
import torch
from PIL import Image

from a2b_lightvpr.checkpointing import file_sha256
from a2b_lightvpr.gsv_metric import MetricCityBatchSampler, MetricVPRDataset, prepare_gsv_metric
from a2b_lightvpr.references.training import _gsv_image_name
from a2b_lightvpr.transforms import VPRTransform


def gsv_fixture(root, counts=(4, 4, 3)):
    (root / "Dataframes").mkdir(parents=True)
    records = []
    for place, count in enumerate(counts, 1):
        for view in range(count):
            row = {"city_id": "Fixture", "place_id": str(place), "year": str(2015 + view),
                   "month": "1", "northdeg": "380", "lat": str(13 + place / 1000),
                   "lon": "100.0", "panoid": f"p{place}-v{view}"}
            path = root / "Images/Fixture" / _gsv_image_name("Fixture", place, row)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12, 8), (100 + view * 20, 80, 60)).save(path)
            records.append(row)
    write_csv(root / "Dataframes/Fixture.csv", records)
    return records


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_preparation_preserves_gsv_labels_gps_and_four_distinct_sources(tmp_path):
    root, output = tmp_path / "gsv", tmp_path / "prepared"
    records = gsv_fixture(root)
    # A duplicate metadata row cannot supply an extra positive image.
    write_csv(root / "Dataframes/Fixture.csv", records + [records[0]])
    before = file_sha256(root / "Dataframes/Fixture.csv")
    audit = prepare_gsv_metric(root, output)
    rows = read_csv(output / "vpr_train.csv")
    assert audit["raw_metadata_rows"] == 12 and audit["raw_places"] == 3
    assert audit["rows"] == audit["distinct_sources"] == len(rows) == 8
    assert audit["places"] == 2 and audit["excluded_places"] == 1
    assert audit["duplicate_source_rows_removed"] == 1
    assert audit["images_per_place_histogram"] == {4: 2}
    assert {row["place_id"] for row in rows} == {"Fixture:1", "Fixture:2"}
    assert {row["dataset"] for row in rows} == {"gsv_cities"}
    assert {row["split"] for row in rows} == {"train"}
    assert {row["yaw"] for row in rows} == {"20.0"}
    assert {row["latitude"] for row in rows} == {"13.001", "13.002"}
    assert all(Path(row["source_path"]).is_file() for row in rows)
    assert len({row["image_id"] for row in rows}) == 8
    assert audit["metadata_csv_sha256"] == {"Fixture.csv": before}
    assert audit["benchmark_splits_read"] == [] and not audit["generated_images_read"]
    assert file_sha256(root / "Dataframes/Fixture.csv") == before
    manifest_before = (output / "vpr_train.csv").read_bytes()
    prepare_gsv_metric(root, output)
    assert (output / "vpr_train.csv").read_bytes() == manifest_before


def test_preparation_fails_missing_sources_before_writing_manifest(tmp_path):
    root = tmp_path / "gsv"
    records = gsv_fixture(root)
    (root / "Images/Fixture" / _gsv_image_name("Fixture", 1, records[0])).unlink()
    with pytest.raises(FileNotFoundError, match="Missing 1 GSV"):
        prepare_gsv_metric(root, tmp_path / "prepared")
    assert not (tmp_path / "prepared/vpr_train.csv").exists()


def test_preparation_rejects_same_canonical_source_with_conflicting_places(tmp_path):
    root = tmp_path / "gsv"
    records = gsv_fixture(root)
    first = root / "Images/Fixture" / _gsv_image_name("Fixture", 1, records[0])
    second = root / "Images/Fixture" / _gsv_image_name("Fixture", 2, records[4])
    second.unlink()
    second.symlink_to(first)
    with pytest.raises(ValueError, match="conflicting place labels"):
        prepare_gsv_metric(root, tmp_path / "prepared")


def test_preparation_cannot_write_into_original_dataset(tmp_path):
    root = tmp_path / "gsv"
    gsv_fixture(root)
    with pytest.raises(ValueError, match="overwrite the GSV source"):
        prepare_gsv_metric(root, root / "Dataframes/output")


@pytest.fixture
def manifest(tmp_path):
    root = tmp_path / "gsv"
    gsv_fixture(root)
    output = tmp_path / "prepared"
    prepare_gsv_metric(root, output)
    return output / "vpr_train.csv"


def test_metric_dataset_uses_one_source_view_and_keeps_rng_and_padding(manifest):
    transform = VPRTransform(14, 14)
    dataset = MetricVPRDataset(manifest, transform, train_domains="day_aug", augmentation_probability=1.0)
    python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
    first = dataset[0]
    second = dataset[0]
    assert set(first) & {"day", "fake", "aug"} == set()
    assert first["image"].shape == (3, 14, 14) and bool(first["is_augmented"])
    assert torch.equal(first["image"], second["image"])
    assert random.getstate() == python_state and torch.equal(torch.get_rng_state(), torch_state)
    with Image.open(dataset.rows[0]["source_path"]) as image:
        original, valid = transform.transform_with_mask(image, patch_size=None)
    assert not torch.equal(first["image"][:, valid], original[:, valid])
    assert torch.equal(first["image"][:, ~valid], original[:, ~valid])
    dataset.set_epoch(1)
    assert not torch.equal(first["image"], dataset[0]["image"])
    dataset.set_epoch(0)
    assert torch.equal(first["image"], dataset[0]["image"])


def test_day_only_never_augments_even_with_probability_one(manifest):
    dataset = MetricVPRDataset(manifest, VPRTransform(14, 14), train_domains="day_only", augmentation_probability=1.0)
    original = dataset[0]
    dataset.set_epoch(2)
    assert not bool(original["is_augmented"])
    assert torch.equal(original["image"], dataset[0]["image"])
    assert len(dataset.rows) == 8 and len(dataset.place_to_label) == 2
    assert len({dataset[index]["image_key"].item() for index in range(len(dataset))}) == 8


def test_metric_duplicates_do_not_satisfy_four_source_requirement(manifest):
    row = read_csv(manifest)[0]
    write_csv(manifest, [{**row, "image_id": f"alias-{index}"} for index in range(4)])
    with pytest.raises(ValueError, match="four-distinct-source class filter"):
        MetricVPRDataset(manifest, VPRTransform(14, 14))


def test_metric_conflicting_alias_and_test_rows_are_rejected(manifest):
    rows = read_csv(manifest)
    write_csv(manifest, rows + [{**rows[0], "place_id": "different-place"}])
    with pytest.raises(ValueError, match="conflicting place labels"):
        MetricVPRDataset(manifest, VPRTransform(14, 14))
    rows[0]["split"] = "test"
    write_csv(manifest, rows)
    with pytest.raises(ValueError, match="non-train"):
        MetricVPRDataset(manifest, VPRTransform(14, 14))


def sampler_rows():
    return [{"dataset": "gsv_cities", "route": city, "global_place_id": f"gsv_cities:{city}:{place}",
             "source_path": f"/{city}/p{place}-v{view}.jpg"}
            for city, count in (("A", 5), ("B", 3)) for place in range(count) for view in range(4)]


def test_city_sampler_covers_every_place_and_samples_distinct_images():
    rows = sampler_rows()
    sampler = MetricCityBatchSampler(rows, {"gsv_cities": 40}, places_per_batch=2, images_per_place=4)
    batches = list(sampler)
    assert len(batches) == 5 and sampler.minimum_coverage_samples == 40
    assert {rows[index]["global_place_id"] for batch in batches for index in batch} == {row["global_place_id"] for row in rows}
    for batch in batches:
        assert len(batch) == len(set(batch)) == 8
        assert len({rows[index]["route"] for index in batch}) == 1
        assert len({rows[index]["global_place_id"] for index in batch}) == 2
        assert len({rows[index]["source_path"] for index in batch}) == 8
    assert sampler.statistics()["gsv_cities"]["place_coverage_ratio"] == 1.0


def test_city_sampler_resume_and_city_partition_validation():
    rows = sampler_rows()
    sampler = MetricCityBatchSampler(rows, {"gsv_cities": 64}, places_per_batch=2, images_per_place=4)
    iterator = iter(sampler)
    next(iterator)
    state = sampler.state_dict()
    expected = list(iterator)
    resumed = MetricCityBatchSampler(rows, {"gsv_cities": 64}, places_per_batch=2, images_per_place=4)
    resumed.load_state_dict(state)
    assert list(resumed) == expected
    changed = [{**row, "route": "C" if row["route"] == "A" else "B"} for row in rows]
    wrong = MetricCityBatchSampler(changed, {"gsv_cities": 64}, places_per_batch=2, images_per_place=4)
    with pytest.raises(ValueError, match="city/source partition"):
        wrong.load_state_dict(state)
    resumed.set_epoch(1)
    assert len(list(resumed)) == 8


def test_city_sampler_rejects_budget_below_place_coverage():
    with pytest.raises(ValueError, match="at least 40 samples"):
        MetricCityBatchSampler(sampler_rows(), {"gsv_cities": 32}, places_per_batch=2, images_per_place=4)


@pytest.mark.parametrize("places", [33, 35])
def test_city_sampler_keeps_all_tail_places_at_a_32_place_cycle_boundary(places):
    rows = [{"dataset": "gsv_cities", "route": "A", "global_place_id": f"gsv_cities:A:{place}",
             "source_path": f"/A/p{place}-v{view}.jpg"} for place in range(places) for view in range(4)]
    for seed in range(20):
        sampler = MetricCityBatchSampler(rows, {"gsv_cities": 256}, random_seed=seed)
        batches = list(sampler)
        assert len(batches) == 2
        assert {rows[index]["global_place_id"] for batch in batches for index in batch} == {row["global_place_id"] for row in rows}
        assert all(len({rows[index]["global_place_id"] for index in batch}) == 32 for batch in batches)
