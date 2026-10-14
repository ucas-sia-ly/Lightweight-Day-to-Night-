import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from a2b_lightvpr import paths
from a2b_lightvpr.checkpointing import checkpoint_config_matches, data_fingerprints, file_sha256
from a2b_lightvpr.config import DataConfig, ExperimentConfig, VPRConfig, load_config, validate_config
from a2b_lightvpr.data import read_generated_manifest, read_generator_manifest
from a2b_lightvpr.generator import translate
from a2b_lightvpr.runtime import load_vpr_model
from a2b_lightvpr.vpr import model as model_module


@pytest.fixture
def layout(tmp_path, monkeypatch):
    project, legacy = tmp_path / "project", tmp_path / "legacy"
    project.mkdir()
    monkeypatch.setattr(paths, "PROJECT_ROOT", project)
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(paths, "DATASETS_ROOT", tmp_path / "datasets")
    monkeypatch.setattr(paths, "TORCH_HUB_ROOT", tmp_path / "torch_hub")
    monkeypatch.setattr(paths, "LEGACY_ROOT", legacy)
    monkeypatch.setattr(paths, "LEGACY_DATASETS_ROOT", legacy / "CycleGAN-turbo_NetVLAD/dataset")
    monkeypatch.delenv("A2B_DATASETS_ROOT", raising=False)
    return project, legacy


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"resource")
    return path


def test_aliases_are_exact_and_never_replace_missing_dates_or_weights(layout):
    project, legacy = layout
    moved = touch(paths.DATASETS_ROOT / "MSLS/train_val/frame.jpg")
    old = paths.LEGACY_DATASETS_ROOT / "MSLS/train_val/frame.jpg"
    assert paths.resolve_resource_path(old) == moved
    missing = paths.LEGACY_DATASETS_ROOT / "RobotCar_RGB/day/missing-date/frame.png"
    touch(paths.DATASETS_ROOT / "RobotCar/day/another-date/frame.png")
    assert paths.resolve_resource_path(missing) == missing
    backbone = touch(paths.TORCH_HUB_ROOT / "checkpoints/dinov2_vits14_reg4_pretrain.pth")
    old_backbone = legacy / "reference_code/salad/dinov2_vits14_reg4_pretrain.pth"
    assert paths.resolve_resource_path(old_backbone) == backbone
    missing_weight = legacy / "reference_code/weights/other_model.pth"
    assert paths.resolve_resource_path(missing_weight) == missing_weight
    # Explicit existing original resources take precedence over any alias.
    touch(old)
    assert paths.resolve_resource_path(old) == old


def test_readers_resolve_paths_without_changing_csv_hash_or_ids(layout):
    project, legacy = layout
    source = touch(paths.DATASETS_ROOT / "MSLS/frame.jpg")
    fake = touch(project / "data/generated/images/frame.png")
    old_source = paths.LEGACY_DATASETS_ROOT / "MSLS/frame.jpg"
    old_fake = legacy / "A2B-LightVPR/data/generated/images/frame.png"
    manifest = project / "data/generated/manifest.csv"
    write_csv(manifest, [dict(image_id="fixed-id", place_id="p", split="train",
                             source_path=str(old_source), generated_path=str(old_fake))])
    before = manifest.read_bytes(), file_sha256(manifest)
    rows = read_generated_manifest(manifest)
    assert rows[0]["source_path"] == str(source) and rows[0]["generated_path"] == str(fake)
    assert rows[0]["image_id"] == "fixed-id"
    assert before == (manifest.read_bytes(), file_sha256(manifest))
    generator = project / "generator.csv"
    write_csv(generator, [dict(split="train", source_path=str(old_source), target_path=str(old_source))])
    assert read_generator_manifest(generator)[0]["target_path"] == str(source)


def test_resume_compares_relocated_resource_identity_and_preserves_hyperparameters(layout):
    project, legacy = layout
    source = touch(project / "data/source.csv")
    config = ExperimentConfig(DataConfig(vpr_source_manifest=str(source)))
    saved = config.as_dict()
    saved["data"]["vpr_source_manifest"] = str(legacy / "A2B-LightVPR/data/source.csv")
    assert checkpoint_config_matches({"config": saved}, config)
    changed = replace(config, vpr=replace(config.vpr, encoder_lr=2e-5))
    assert not checkpoint_config_matches({"config": saved}, changed)


def test_translation_reuses_unchanged_metadata_and_manifest_after_move(layout, monkeypatch):
    project, legacy = layout
    source = touch(paths.DATASETS_ROOT / "MSLS/day.png")
    source_manifest = project / "data/source.csv"
    old_source = paths.LEGACY_DATASETS_ROOT / "MSLS/day.png"
    write_csv(source_manifest, [dict(image_id="i", split="train", dataset="msls", place_id="p", source_path=str(old_source))])
    generator_checkpoint = project / "runs/generator/generator_final.pt"
    generator_checkpoint.parent.mkdir(parents=True)
    torch.save(dict(kind="generator_inference", objective="qsattn", step=30000,
                    config_fingerprint="original", generator={}), generator_checkpoint)
    output = project / "data/generated/test"
    fake = touch(output / "images/msls/i.png")
    output_manifest = output / "manifest.csv"
    old_fake = legacy / "A2B-LightVPR/data/generated/test/images/msls/i.png"
    write_csv(output_manifest, [dict(image_id="i", source_id="i", source_path=str(old_source),
        generated_path=str(old_fake), split="train", domain="fake_night", role="database", traverse="",
        utm_e="", utm_n="", yaw="", place_id="p", generator_sha256=file_sha256(generator_checkpoint), dataset="msls")])
    metadata = dict(checkpoint=str(legacy / "A2B-LightVPR/runs/generator/generator_final.pt"),
                    checkpoint_sha256=file_sha256(generator_checkpoint), checkpoint_step=30000,
                    checkpoint_config_fingerprint="original",
                    source_manifest=str(legacy / "A2B-LightVPR/data/source.csv"),
                    source_manifest_sha256=file_sha256(source_manifest), output_size=[320, 480],
                    datasets=["msls"], smoke_limit=None)
    metadata_path = output / "translation_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    before = output_manifest.read_bytes(), metadata_path.read_bytes()
    monkeypatch.setattr(translate, "ResnetGenerator", lambda: torch.nn.Identity())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = ExperimentConfig(DataConfig(vpr_source_manifest=str(source_manifest),
                                         generated_root=str(output)))
    assert translate.translate_training_set(config, generator_checkpoint) == output_manifest
    assert (output_manifest.read_bytes(), metadata_path.read_bytes()) == before
    assert fake.is_file() and source.is_file()
    assert not translate.translation_metadata_matches(metadata, {**metadata, "checkpoint_sha256": "changed"})


def test_fixed_budget_source_training_fingerprints_only_active_training_assets(tmp_path):
    manifest = touch(tmp_path / "source.csv")
    config = ExperimentConfig(DataConfig(vpr_source_manifest=str(manifest)),
        vpr=VPRConfig(model_family="stable_night_salad", selection_rule="fixed_budget_last", evaluate_epoch_zero=False))
    assert data_fingerprints(config) == {"vpr_source_manifest": file_sha256(manifest)}


def test_available_dataset_profile_is_explicit_and_keeps_legacy_dataset_constraint():
    config = load_config("configs/experiments/students/stable_night_salad_msls_svox_day_aug_544.yaml")
    assert config.vpr.datasets == ("msls", "svox")
    assert config.vpr.selection_rule == "fixed_budget_last"
    assert config.vpr.samples_per_epoch == sum(config.vpr.dataset_epoch_quotas.values()) == 40000
    legacy = ExperimentConfig(DataConfig(), vpr=VPRConfig(datasets=("msls", "svox")))
    with pytest.raises(ValueError, match="Formal VPR training"):
        validate_config(legacy)


def test_full_inference_load_does_not_require_original_mixvpr_initialization(tmp_path, monkeypatch):
    class TinyOfficial(torch.nn.Module):
        descriptor_dim = 512

        def __init__(self, *args):
            super().__init__()
            self.backbone = torch.nn.Linear(3, 3)
            self.aggregator = torch.nn.Linear(3, 512)

    monkeypatch.setattr(model_module, "OfficialMixVPRModel", TinyOfficial)
    cfg = VPRConfig(model_family="mixvpr", aggregator="mixvpr", descriptor_dim=512,
                    image_height=320, image_width=320, init_checkpoint=str(tmp_path / "missing_initial.ckpt"))
    config = ExperimentConfig(DataConfig(), vpr=cfg)
    original = TinyOfficial()
    checkpoint = tmp_path / "inference.pt"
    torch.save(dict(kind="vpr_inference", descriptor_dim=512, random_seed=0, model=original.state_dict()), checkpoint)
    loaded, _, _ = load_vpr_model(config, checkpoint)
    assert all(torch.equal(v, loaded.state_dict()[k]) for k, v in original.state_dict().items())
    with pytest.raises(ValueError, match="requires a checkpoint"):
        model_module.build_descriptor_model("mixvpr", image_height=320, image_width=320,
                                            descriptor_dim=512, model_family="mixvpr")


def test_prepare_refuses_to_overwrite_incomplete_existing_robotcar_training_identities(layout, monkeypatch):
    from a2b_lightvpr import prepare
    project, _ = layout
    monkeypatch.setattr(prepare, "PROJECT_ROOT", project)
    output = project / "data/manifests"
    manifest = output / "vpr_train.csv"
    write_csv(manifest, [dict(dataset="robotcar", split="train", source_path="missing.png")])
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="refusing to overwrite"):
        prepare.build_manifests(paths.DATASETS_ROOT, output)
    assert manifest.read_bytes() == before


def test_prepare_preserves_existing_robotcar_training_ids_without_test_assets(layout, monkeypatch):
    from a2b_lightvpr import prepare
    project, _ = layout
    monkeypatch.setattr(prepare, "PROJECT_ROOT", project)
    output = project / "data/manifests"
    rows = [dict(dataset="robotcar", split="train", source_path=str(touch(project / f"day{i}.png")),
                 image_id=f"frozen-id-{i}", place_id="frozen-place", route="training-route") for i in range(2)]
    manifest = output / "vpr_train.csv"
    write_csv(manifest, rows)
    summary = prepare.build_manifests(paths.DATASETS_ROOT, output)
    with manifest.open() as handle:
        retained = list(csv.DictReader(handle))
    assert [(row["image_id"], row["place_id"], row["route"]) for row in retained] == [
        (row["image_id"], row["place_id"], row["route"]) for row in rows]
    assert summary["vpr"] == {"robotcar": 2}
    assert not (project / "data/protocols").exists()
