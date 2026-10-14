import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from a2b_lightvpr.config import DataConfig, ExperimentConfig, VPRConfig
from a2b_lightvpr.resource_audit import audit_resources


def write_manifest(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture_config(tmp_path, mode="day_aug", *, source_duplicates=False):
    repo = tmp_path / "dino"
    repo.mkdir()
    (repo / "hubconf.py").write_text("# fixture", encoding="utf-8")
    weights = tmp_path / "dino.pt"
    weights.write_bytes(b"not-a-loadable-checkpoint")
    rows = []
    for place in range(2):
        for view in range(2):
            source, fake = tmp_path / f"day{place}{view}.png", tmp_path / f"fake{place}{view}.png"
            source.write_bytes(b"not-a-decodable-image")
            fake.write_bytes(b"not-a-decodable-image")
            rows.append({"dataset": "msls", "split": "train", "place_id": str(place),
                         "image_id": f"image{place}{view}", "source_path": str(source), "generated_path": str(fake)})
    if source_duplicates:
        rows = [rows[0], {**rows[0], "image_id": "generator-duplicate-id"}, rows[2], rows[3]]
    manifest = tmp_path / "source.csv"
    write_manifest(manifest, rows)
    generated = tmp_path / "generated"
    if mode in {"day_fake", "day_aug_fake"}:
        generated.mkdir()
        write_manifest(generated / "manifest.csv", rows)
    config = ExperimentConfig(
        DataConfig(robotcar_seasons_root=str(tmp_path / "missing_seasons"),
                   vpr_source_manifest=str(manifest), generated_root=str(generated),
                   generator_manifest=str(tmp_path / "missing_generator.csv")),
        vpr=VPRConfig(model_family="stable_night_salad", aggregator="salad", descriptor_dim=544,
                      train_domains=mode, backbone_repo=str(repo), backbone_checkpoint=str(weights),
                      datasets=("msls",), min_class_images=2, sampler="pk_balanced",
                      pk_places_per_batch=2, pk_images_per_place=2,
                      selection_rule="fixed_budget_last", evaluate_epoch_zero=False),
    )
    return config, manifest, rows


@pytest.mark.parametrize("mode", ["day_only", "day_aug"])
def test_source_only_fixed_budget_readiness_does_not_require_generator_or_test_metadata(tmp_path, monkeypatch, mode):
    config, manifest, _ = fixture_config(tmp_path, mode)
    original = manifest.read_bytes()
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("audit must not load weights"))
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: pytest.fail("audit must not decode images"))
    output = tmp_path / "reports" / "resources.json"
    report = audit_resources(config, output)
    assert report["training_ready"]
    assert report["assets"]["training_manifest"]["path"] == str(manifest)
    assert report["assets"]["robotcar_seasons_root"]["required_for"] == "evaluation"
    assert "robotcar_seasons_root" in report["evaluation_gaps"]
    assert report["assets"]["generator_manifest"]["required_for"] == "generator"
    assert not report["assets"]["generator_manifest"]["available"]
    assert report["datasets"]["msls"]["eligible_places"] == 2
    assert report["datasets"]["msls"]["eligible_rows"] == 4
    assert report["datasets"]["msls"]["source_missing"] == 0
    assert report["datasets"]["msls"]["unique_fake_paths"] == 0
    assert manifest.read_bytes() == original
    assert not (tmp_path / "generated").exists()
    assert json.loads(output.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize("control", ["epoch_zero", "validation_best"])
def test_public_validation_metadata_required_for_active_training_validation(tmp_path, control):
    config, _, _ = fixture_config(tmp_path)
    overrides = {"evaluate_epoch_zero": True} if control == "epoch_zero" else {"selection_rule": "validation_best"}
    config = replace(config, vpr=replace(config.vpr, **overrides))
    report = audit_resources(config)
    assert not report["training_ready"]
    assert report["assets"]["msls_root"]["required_for"] == "training"
    assert report["assets"]["validation_msls_cph_query"]["required_for"] == "training"
    assert report["assets"]["robotcar_seasons_root"]["required_for"] == "evaluation"
    assert "robotcar_manifest" not in report["assets"]


@pytest.mark.parametrize("mode", ["day_fake", "day_aug_fake"])
def test_audit_counts_missing_sources_and_fake_images_without_removing_rows(tmp_path, mode):
    config, _, rows = fixture_config(tmp_path, mode=mode)
    original = (tmp_path / "generated/manifest.csv").read_bytes()
    Path(rows[0]["source_path"]).unlink()
    Path(rows[1]["generated_path"]).unlink()
    report = audit_resources(config)
    info = report["datasets"]["msls"]
    assert not report["training_ready"]
    assert info["selected_rows"] == info["unique_sources"] == info["unique_fake_paths"] == 4
    assert info["source_missing"] == info["fake_missing"] == 1
    assert info["eligible_rows"] == 4
    assert info["missing_examples"]["source"] == [rows[0]["source_path"]]
    assert info["missing_examples"]["fake"] == [rows[1]["generated_path"]]
    assert (tmp_path / "generated/manifest.csv").read_bytes() == original


def test_audit_enforces_distinct_source_class_and_pk_counts(tmp_path):
    config, _, _ = fixture_config(tmp_path, source_duplicates=True)
    report = audit_resources(config)
    info = report["datasets"]["msls"]
    assert not report["training_ready"]
    assert info["selected_rows"] == 4 and info["unique_sources"] == 3
    assert info["duplicate_source_rows"] == 1
    assert info["eligible_places"] == 1 and info["eligible_rows"] == 2
    assert not info["pk_ready"]
    # Legacy place filtering remains based on row count, while PK readiness is
    # stricter and prevents a generator duplicate from satisfying K sources.
    generated = tmp_path / "legacy_generated"
    generated.mkdir()
    generated.joinpath("manifest.csv").write_bytes((tmp_path / "source.csv").read_bytes())
    legacy = replace(config, data=replace(config.data, generated_root=str(generated)),
                     vpr=replace(config.vpr, model_family="custom", train_domains="day"))
    legacy_info = audit_resources(legacy)["datasets"]["msls"]
    assert legacy_info["eligible_places"] == 2
    assert legacy_info["pk_undersized_places"] == 1
    assert not legacy_info["pk_ready"]


def test_audit_filters_inactive_datasets_but_rejects_selected_invalid_splits(tmp_path):
    config, manifest, rows = fixture_config(tmp_path)
    unrelated = {**rows[0], "dataset": "robotcar", "source_path": str(tmp_path / "missing.png"), "split": "test"}
    write_manifest(manifest, rows + [unrelated])
    assert audit_resources(config)["training_ready"]
    rows[0]["split"] = "test"
    write_manifest(manifest, rows + [unrelated])
    report = audit_resources(config)
    assert not report["training_ready"]
    assert report["selected_rows"] == 4
    assert report["invalid_splits"] == {"test": 1}


def test_skip_images_reports_unknown_missing_counts_and_limited_scope(tmp_path):
    config, _, rows = fixture_config(tmp_path)
    Path(rows[0]["source_path"]).unlink()
    report = audit_resources(config, scan_images=False)
    assert report["training_ready"]
    assert report["images_scanned"] is False
    assert report["readiness_scope"] == "assets_manifest_and_sampling_only"
    assert report["datasets"]["msls"]["source_missing"] is None
    assert report["datasets"]["msls"]["fake_missing"] is None


def test_dino_entrypoint_init_weights_and_cdnd_cache_sidecars_are_required(tmp_path):
    config, _, _ = fixture_config(tmp_path)
    repo = tmp_path / "dino"
    (repo / "hubconf.py").unlink()
    missing = audit_resources(config)
    assert not missing["assets"]["backbone_repo"]["available"]
    assert not missing["training_ready"]
    (repo / "hubconf.py").touch()
    teacher = tmp_path / "teacher.pt"
    teacher.touch()
    cache = tmp_path / "teacher.npy"
    cache.touch()
    cdnd = replace(config, vpr=replace(config.vpr, cross_domain_aux="cross_domain_relation",
        relation_teacher="salad", relation_teacher_repo=str(repo), relation_teacher_checkpoint=str(teacher),
        relation_teacher_cache=str(cache), init_checkpoint=str(tmp_path / "missing_init.pt")))
    report = audit_resources(cdnd)
    assert not report["training_ready"]
    for name in ("init_weights", "relation_teacher_cache_metadata", "relation_teacher_cache_ids"):
        assert not report["assets"][name]["available"]
        assert report["assets"][name]["required_for"] == "training"


def test_audit_output_cannot_overwrite_its_source_manifest(tmp_path):
    config, manifest, _ = fixture_config(tmp_path)
    original = manifest.read_bytes()
    with pytest.raises(ValueError, match="overwrite an input"):
        audit_resources(config, manifest)
    assert manifest.read_bytes() == original


def test_audit_resolves_migrated_csv_images_without_rewriting_provenance(tmp_path, monkeypatch):
    from a2b_lightvpr import paths
    config, _, rows = fixture_config(tmp_path, mode="day_fake")
    datasets = tmp_path / "moved_datasets"
    msls = datasets / "MSLS"
    msls.mkdir(parents=True)
    for row in rows:
        source = Path(row["source_path"])
        moved = msls / source.name
        moved.write_bytes(source.read_bytes())
        row["source_path"] = str(paths.LEGACY_DATASETS_ROOT / "MSLS" / source.name)
        row["generated_path"] = str(paths.LEGACY_ROOT / "A2B-LightVPR" / Path(row["generated_path"]).name)
    monkeypatch.setattr(paths, "DATASETS_ROOT", datasets)
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("A2B_DATASETS_ROOT", raising=False)
    manifest = tmp_path / "generated/manifest.csv"
    write_manifest(manifest, rows)
    original = manifest.read_bytes()
    report = audit_resources(config)
    assert report["training_ready"]
    assert report["datasets"]["msls"]["source_missing"] == 0
    assert report["datasets"]["msls"]["fake_missing"] == 0
    assert manifest.read_bytes() == original


def test_missing_examples_are_capped_and_source_labels_cannot_conflict(tmp_path):
    config, manifest, rows = fixture_config(tmp_path)
    for index in range(10):
        rows.append({**rows[0], "image_id": f"missing{index}", "source_path": str(tmp_path / f"missing{index}.png")})
    rows.append({**rows[0], "image_id": "conflicting_label", "place_id": "other_place"})
    write_manifest(manifest, rows)
    report = audit_resources(config)
    assert not report["training_ready"]
    assert report["datasets"]["msls"]["source_missing"] == 10
    assert len(report["datasets"]["msls"]["missing_examples"]["source"]) == 5
    assert report["datasets"]["msls"]["conflicting_source_labels"] == 1


@pytest.mark.parametrize("problem", ["ownership", "generator_hash"])
def test_audit_rejects_stable_fake_provenance_conflicts(tmp_path, problem):
    config, _, rows = fixture_config(tmp_path, mode="day_fake")
    if problem == "ownership":
        rows[1]["generated_path"] = rows[0]["generated_path"]
    else:
        rows[0]["generator_sha256"] = "generator_a"
        rows.append({**rows[0], "image_id": "duplicate", "generator_sha256": "generator_b"})
    write_manifest(tmp_path / "generated/manifest.csv", rows)
    report = audit_resources(config)
    info = report["datasets"]["msls"]
    assert not report["training_ready"]
    assert info["pk_ready"]
    assert info["source_missing"] == info["fake_missing"] == 0
    assert info["conflicting_fake_ownership"] == (problem == "ownership")
    assert info["conflicting_generator_hashes"] == (problem == "generator_hash")
    assert len(info["fake_metadata_conflict_examples"]) == 1


def test_audit_accepts_consistent_duplicate_fake_metadata_and_retains_known_hash(tmp_path):
    config, _, rows = fixture_config(tmp_path, mode="day_fake")
    rows[0]["generator_sha256"] = "generator_a"
    rows.extend([{**rows[0], "image_id": "duplicate1", "generator_sha256": ""},
                 {**rows[0], "image_id": "duplicate2"}])
    write_manifest(tmp_path / "generated/manifest.csv", rows)
    report = audit_resources(config)
    info = report["datasets"]["msls"]
    assert report["training_ready"]
    assert info["selected_rows"] == 6 and info["unique_sources"] == info["eligible_rows"] == 4
    assert info["duplicate_source_rows"] == 2
    assert info["conflicting_fake_ownership"] == info["conflicting_generator_hashes"] == 0


@pytest.mark.parametrize("weights,ready", [({"msls": 1.0}, True), ({"svox": 1.0}, False),
                                          ({"msls": 1.0, "svox": 1.0}, False)])
def test_audit_checks_dataset_sampling_weight_coverage(tmp_path, weights, ready):
    config, _, _ = fixture_config(tmp_path)
    config = replace(config, vpr=replace(config.vpr, dataset_sampling_weights=weights))
    report = audit_resources(config)
    assert report["training_ready"] is ready
    if not ready:
        assert any("sampling weights" in error for error in report["training_errors"])
