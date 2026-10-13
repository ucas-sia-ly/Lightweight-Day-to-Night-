import csv
from dataclasses import replace

import pytest
from PIL import Image

from a2b_lightvpr.config import EvalConfig
from a2b_lightvpr.data import RobotCarRow
from a2b_lightvpr.evaluation import positive_indices, resolve_database_domain


def row(path, role="database"):
    return RobotCarRow("test", role, "day" if role == "database" else "night",
                       "route", "1", 0., 0., 0., str(path), "")


def write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "generated_path", "split"])
        writer.writeheader()
        writer.writerows(rows)


def test_fake_database_preserves_order_pose_and_positives_and_query(tmp_path):
    day_paths = [tmp_path / f"day{i}.png" for i in range(2)]
    fake_paths = [tmp_path / f"fake{i}.png" for i in range(2)]
    for image in day_paths + fake_paths:
        Image.new("RGB", (16, 16)).save(image)
    database = [row(day_paths[0]), replace(row(day_paths[1]), easting=100., timestamp="2")]
    query = row(tmp_path / "real_night.png", "query")
    manifest = tmp_path / "fake.csv"
    # A reordered CSV does not reorder the retrieval database.
    write_manifest(manifest, [{"source_path": source.name, "generated_path": fake.name, "split": "test"}
                              for source, fake in reversed(list(zip(day_paths, fake_paths)))])
    cfg = EvalConfig(database_domain="fake_night", fake_database_manifest=str(manifest))
    mapped, digest = resolve_database_domain(database, cfg, "test")
    assert len(digest) == 64
    assert [r.source_path for r in mapped] == list(map(str, fake_paths))
    assert [(r.easting, r.timestamp) for r in mapped] == [(r.easting, r.timestamp) for r in database]
    before = positive_indices(database, [query], 10., 30.)
    after = positive_indices(mapped, [query], 10., 30.)
    assert before[0].tolist() == after[0].tolist() == [0]
    assert query.source_path == str(tmp_path / "real_night.png")
    assert database[0].source_path == str(day_paths[0])
    assert resolve_database_domain(database, EvalConfig(), "test") == (database, "")


@pytest.mark.parametrize("problem", ["missing", "split", "ambiguous", "same", "absent_file"])
def test_fake_database_fails_incomplete_or_invalid_mapping(tmp_path, problem):
    day, fake = tmp_path / "day.png", tmp_path / "fake.png"
    Image.new("RGB", (16, 16)).save(fake)
    entry = {"source_path": str(day), "generated_path": str(fake), "split": "test"}
    entries = [entry]
    if problem == "missing":
        entries = []
    elif problem == "split":
        entry["split"] = "train"
    elif problem == "ambiguous":
        entries += [{**entry, "generated_path": str(tmp_path / "other.png")}]
    elif problem == "same":
        entry["generated_path"] = str(day)
    elif problem == "absent_file":
        fake.unlink()
    manifest = tmp_path / "fake.csv"
    write_manifest(manifest, entries)
    cfg = EvalConfig(database_domain="fake_night", fake_database_manifest=str(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_database_domain([row(day)], cfg, "test")


def test_fake_mapping_rejects_real_query_and_shared_outputs(tmp_path):
    query_path = tmp_path / "query.png"
    fake = tmp_path / "fake.png"
    for path in (query_path, fake):
        Image.new("RGB", (16, 16)).save(path)
    day1, day2 = tmp_path / "day1.png", tmp_path / "day2.png"
    manifest = tmp_path / "fake.csv"
    cfg = EvalConfig(database_domain="fake_night", fake_database_manifest=str(manifest))
    write_manifest(manifest, [{"source_path": str(day1), "generated_path": str(query_path), "split": "test"}])
    with pytest.raises(ValueError, match="real Database or Query"):
        resolve_database_domain([row(day1)], cfg, "test", [row(query_path, "query")])
    write_manifest(manifest, [{"source_path": str(day), "generated_path": str(fake), "split": "test"}
                              for day in (day1, day2)])
    with pytest.raises(ValueError, match="distinct generated"):
        resolve_database_domain([row(day1), row(day2)], cfg, "test")
