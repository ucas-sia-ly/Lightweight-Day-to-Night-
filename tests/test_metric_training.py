import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn

from a2b_lightvpr.checkpointing import checkpoint_config_matches
from a2b_lightvpr.config import DataConfig, ExperimentConfig, apply_overrides, load_config
from a2b_lightvpr.vpr import trainer
from a2b_lightvpr.runtime import vpr_device


CONFIG = "configs/experiments/metric/salad_small_gsv_day_aug_544.yaml"


@pytest.mark.parametrize("domain", ["day_only", "day_aug"])
def test_metric_recipes_keep_pretrained_source_and_source_distinct_budget(domain):
    cfg = load_config(f"configs/experiments/metric/salad_small_gsv_{domain}_544.yaml")
    assert cfg.vpr.init_checkpoint.endswith("salad-dinov2-s-544-gsv/last.ckpt")
    assert cfg.vpr.training_objective == "multi_similarity"
    assert cfg.vpr.backbone_name == "dinov2_vits14"
    assert cfg.vpr.descriptor_dim == 544 and cfg.vpr.input_transform == "resize"
    assert cfg.vpr.batch_size == cfg.vpr.pk_places_per_batch * cfg.vpr.pk_images_per_place == 128
    assert cfg.vpr.samples_per_epoch == 251392
    assert cfg.vpr.require_cuda
    assert cfg.vpr.evaluate_epoch_zero and cfg.vpr.selection_rule == "validation_best"
    assert cfg.evaluation.datasets == ("svox_day_val",)


def test_recommended_training_fails_before_data_loading_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = load_config(CONFIG)
    cfg = replace(cfg, vpr=replace(cfg.vpr, output_dir=str(tmp_path / "new_run")))
    with pytest.raises(RuntimeError, match="requires CUDA"):
        trainer.train_vpr(cfg)
    assert vpr_device(replace(cfg, vpr=replace(cfg.vpr, require_cuda=False))).type == "cpu"


@pytest.mark.parametrize("override,match", [
    ("vpr.train_domains=day_fake", "source-only"),
    ("vpr.datasets=[msls]", "GSV-Cities"),
    ("vpr.min_class_images=2", "four distinct"),
    ("vpr.salad_dropout=1.0", "dropout"),
    ("vpr.metric_augmentation_probability=1.1", "probability"),
    ("vpr.metric_within_city=1", "boolean"),
    ("vpr.input_transform=crop", "input_transform"),
    ("evaluation.batch_size=-1", "batch_size"),
    ("vpr.pk_places_per_batch=2.5", "positive integer"),
    ("vpr.pk_images_per_place=true", "positive integer"),
    ("vpr.min_class_images=4.5", "positive integer"),
    ("vpr.dataset_epoch_quotas={gsv_cities: 251392.0}", "positive integer"),
    ("vpr.min_class_images_by_dataset={gsv_cities: 4.5}", "positive integer"),
])
def test_metric_configuration_rejects_incompatible_training(override, match):
    with pytest.raises(ValueError, match=match):
        apply_overrides(load_config(CONFIG), [override])


def test_metric_supports_explicit_fake_database_and_aggregator_only_training():
    cfg = apply_overrides(load_config(CONFIG), [
        "evaluation.database_domain=fake_night",
        "evaluation.fake_database_manifest=explicit_test_mapping.csv",
        "vpr.backbone_trainable_blocks=0",
    ])
    assert cfg.evaluation.database_domain == "fake_night"
    with pytest.raises(ValueError, match="explicit"):
        apply_overrides(cfg, ["evaluation.fake_database_manifest="])


def test_default_migration_does_not_resume_metric_as_classification():
    old = ExperimentConfig(DataConfig(vpr_source_manifest="legacy_source.csv"))
    values = old.as_dict()
    for name in ("training_objective", "input_transform", "salad_dropout",
                 "metric_augmentation_probability", "metric_within_city", "require_cuda"):
        del values["vpr"][name]
    del values["evaluation"]["batch_size"]
    state = {"config": values, "config_fingerprint": "old-schema"}
    assert checkpoint_config_matches(state, old, ("data", "vpr", "evaluation"))
    assert not checkpoint_config_matches(state, load_config(CONFIG), ("data", "vpr", "evaluation"))


class TinyDescriptor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 4, 1)
        self.aggregator = nn.Linear(4, 544)
        self.descriptor_dim = 544

    def encode(self, images):
        return F.normalize(self.aggregator(self.backbone(images).mean((2, 3))), dim=1)


def test_real_metric_dataset_training_resume_logging_and_pretrained_best(tmp_path, monkeypatch):
    rows = []
    for place in range(4):
        for source in range(4):
            image = tmp_path / f"p{place}s{source}.png"
            Image.new("RGB", (32, 24), (50 + place * 20, 60 + source * 15, 90)).save(image)
            rows.append(dict(image_id=f"{place}-{source}", dataset="gsv_cities", split="train",
                             place_id=f"city:{place}", route="city", source_path=str(image)))
    manifest = tmp_path / "source.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    cfg = load_config(CONFIG)
    cfg = replace(cfg, data=replace(cfg.data, vpr_source_manifest=str(manifest)), vpr=replace(
        cfg.vpr, output_dir=str(tmp_path / "run"), epochs=2, workers=0, batch_size=4,
        pk_places_per_batch=2, pk_images_per_place=2, samples_per_epoch=8,
        dataset_epoch_quotas={"gsv_cities": 8}, gradient_accumulation_steps=2,
        epoch_one_max_recall_drop=0., amp=False,
        require_cuda=False,
    ))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(trainer, "data_fingerprints", lambda *a, **kw: {"source": "actual-test"})
    builds = []

    def build(*args, **kwargs):
        builds.append(kwargs)
        return TinyDescriptor()

    def unused(*args, **kwargs):
        raise AssertionError("Metric source-only training must not construct a classifier or teacher cache")

    monkeypatch.setattr(trainer, "build_descriptor_model", build)
    monkeypatch.setattr(trainer, "MarginCosineClassifier", unused)
    monkeypatch.setattr(trainer, "TeacherDescriptorCache", unused)
    scores = iter([.8, .7, .6])
    monkeypatch.setattr(trainer, "_evaluate_validation", lambda *a, **kw: {
        "recalls": {"1": next(scores), "5": .9, "10": 1.},
        "mrr": .85, "valid_queries": 10, "total_queries": 10,
    })
    best = trainer.train_vpr(cfg, smoke_epochs=1)
    assert builds[0]["init_checkpoint"] == cfg.vpr.init_checkpoint
    assert builds[0]["salad_dropout"] == .3
    assert torch.load(best, weights_only=True)["epoch"] == 0
    resume = replace(cfg, vpr=replace(cfg.vpr, resume=str(Path(cfg.vpr.output_dir) / "checkpoints/latest.pt")))
    trainer.train_vpr(resume)
    state = torch.load(resume.vpr.resume, weights_only=False)
    assert state["epoch"] == 2 and state["optimizer_step"] == 2
    assert state["classifier"] == {}
    assert state["sampler_state"]["kind"] == "metric_city_pk"
    assert torch.load(best, weights_only=True)["epoch"] == 0  # Keep the measured initial model.
    logs = EventAccumulator(str(Path(cfg.vpr.output_dir) / "tensorboard")).Reload()
    assert [item.step for item in logs.Scalars("train/place")] == [1, 2]
    assert "train/mined_positive_pairs" in logs.Tags()["scalars"]
    assert "train/augmentation_fraction" in logs.Tags()["scalars"]
    records = [json.loads(line) for line in (Path(cfg.vpr.output_dir) / "training_metrics.jsonl").read_text().splitlines()]
    assert len(records) == 2 and all(item["student_images"] == 8 for item in records)
    assert all(item["reference_images"] == 0 for item in records)
    moved = replace(cfg, vpr=replace(cfg.vpr, output_dir=str(tmp_path / "moved"), resume=resume.vpr.resume))
    restored_best = trainer.train_vpr(moved)
    assert restored_best.read_bytes() == best.read_bytes()
    assert torch.load(restored_best, weights_only=True)["training_steps"] == 0
