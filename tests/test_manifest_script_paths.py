import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from a2b_lightvpr import paths
from a2b_lightvpr.checkpointing import file_sha256


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts/data"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_path_test_{name}", SCRIPT_ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def migrated_fixture(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    datasets = tmp_path / "datasets"
    monkeypatch.chdir(project)
    monkeypatch.setattr(paths, "PROJECT_ROOT", project)
    monkeypatch.setenv("A2B_DATASETS_ROOT", str(datasets))
    rows = []
    for dataset in ("robotcar", "svox", "msls"):
        day = datasets / dataset / "day.png"
        day.parent.mkdir(parents=True)
        day.write_bytes(b"day fixture")
        fake = project / "data/generated/multidataset_qsattn" / dataset / "fake.png"
        fake.parent.mkdir(parents=True)
        fake.write_bytes(b"fake fixture")
        checkpoint = project / f"runs/generator/qsattn_{dataset}_30000_seed0/generator_final.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"generator fixture")
        digest = file_sha256(checkpoint)
        row = {
            "dataset": dataset,
            "image_id": dataset,
            "source_path": str(paths.LEGACY_DATASETS_ROOT / dataset / "day.png"),
            "generated_path": str(paths.LEGACY_ROOT / "A2B-LightVPR" / fake.relative_to(project)),
            "place_id": "p",
            "split": "train",
            "generator_sha256": digest,
        }
        rows.append(row)
        write_manifest(fake.parent / "manifest.csv", [row])
        write_manifest(project / f"data/manifests_multidataset_qsattn/{dataset}_vpr_train.csv", [row])
        (fake.parent / "translation_metadata.json").write_text(json.dumps({
            "checkpoint": str(paths.LEGACY_ROOT / "A2B-LightVPR" / checkpoint.relative_to(project)),
            "checkpoint_sha256": digest,
            "checkpoint_step": 30000,
            "smoke_limit": None,
        }))
    combined = project / "data/generated/multidataset_qsattn/manifest.csv"
    write_manifest(combined, rows)
    return project, datasets, rows, combined


def test_multigenerator_migration_checks_preserve_raw_manifest_and_hash(tmp_path, monkeypatch):
    _, _, rows, combined = migrated_fixture(tmp_path, monkeypatch)
    before = combined.read_bytes()
    before_hash = file_sha256(combined)
    script = load_script("prepare_multidataset_qsattn")
    script.merge_manifests()
    assert combined.read_bytes() == before
    assert file_sha256(combined) == before_hash
    with combined.open() as handle:
        assert list(csv.DictReader(handle)) == rows


def test_multigenerator_missing_source_keeps_output_and_counts(tmp_path, monkeypatch):
    _, datasets, _, combined = migrated_fixture(tmp_path, monkeypatch)
    before = combined.read_bytes()
    (datasets / "robotcar/day.png").unlink()
    script = load_script("prepare_multidataset_qsattn")
    with pytest.raises(ValueError, match="missing or incorrectly routed image"):
        script.merge_manifests()
    assert combined.read_bytes() == before
    assert len(list(csv.DictReader(combined.open()))) == 3


def test_single_dataset_migration_preserves_existing_raw_output(tmp_path, monkeypatch):
    project, _, rows, _ = migrated_fixture(tmp_path, monkeypatch)
    monkeypatch.syspath_prepend(str(SCRIPT_ROOT))
    output = project / "data/generated/msls_only_qsattn30k/manifest.csv"
    write_manifest(output, [rows[-1]])
    before = output.read_bytes()
    script = load_script("prepare_single_dataset")
    script.prepare("msls")
    assert output.read_bytes() == before
    with output.open() as handle:
        assert list(csv.DictReader(handle)) == [rows[-1]]


def test_robotcar_preparation_uses_only_existing_training_manifests(tmp_path, monkeypatch):
    project, _, rows, _ = migrated_fixture(tmp_path, monkeypatch)
    script = load_script("prepare_robotcar")
    source = project / "data/manifests_clean_night"
    robot = [row for row in rows if row["dataset"] == "robotcar"]
    # Two training views are required for a usable place class.
    second = project / "second-view.png"
    second.write_bytes(b"second source")
    robot.append({**robot[0], "image_id": "second-view", "source_path": str(second)})
    write_manifest(source / "generator_train.csv", [{**row, "target_path": row["source_path"]} for row in robot])
    write_manifest(source / "vpr_train.csv", robot)
    config = SimpleNamespace(data=SimpleNamespace(
        generator_manifest=str(project / "data/manifests_robotcar_only/generator_train.csv"),
        vpr_source_manifest=str(project / "data/manifests_robotcar_only/vpr_train.csv"),
    ))
    monkeypatch.setattr(script, "load_config", lambda _: config)
    script.prepare()
    audit = json.loads((project / "data/manifests_robotcar_only/audit.json").read_text())
    assert audit["vpr_images"] == 2 and audit["split"] == "train"
    assert not any("protocol" in key for key in audit["hashes"])
    assert not (project / "data/protocols").exists()
