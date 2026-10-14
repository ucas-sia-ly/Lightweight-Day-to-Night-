from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn

from a2b_lightvpr.config import DataConfig, ExperimentConfig, VPRConfig
from a2b_lightvpr.vpr import trainer


class TinyHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(4, dim)
        self.reliable = nn.Conv2d(4, 1, 1)
        self.gamma = nn.Parameter(torch.tensor(1.))

    def forward(self, features):
        return F.normalize(self.fc(features.mean((2, 3))), dim=1)


class TinyModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.backbone = nn.Conv2d(3, 4, 1)
        self.aggregator = TinyHead(dim)
        self.calls = 0

    def encode(self, images):
        self.calls += 1
        features = F.adaptive_avg_pool2d(self.backbone(images), (2, 2))
        return self.aggregator(features)

    def encode_with_aux(self, images):
        self.calls += 1
        features = F.adaptive_avg_pool2d(self.backbone(images), (2, 2))
        reliability = self.aggregator.reliable(features).sigmoid().flatten(1)
        return self.aggregator(features), {"reliability": reliability, "dustbin_assignment": 1-reliability,
            "local_features": features, "cluster_mass": reliability.new_ones(len(images), 16)}


class Images(torch.utils.data.Dataset):
    def __init__(self, *args, train_domains="day", include_fake=False, **kwargs):
        self.rows = [{"dataset": "robotcar", "global_place_id": f"p{i//2}", "image_id": str(i)} for i in range(4)]
        self.place_to_label = {"p0": 0, "p1": 1}
        self.augmentation = "aug" in train_domains
        self.fake = include_fake or "fake" in train_domains

    def __len__(self):
        return 4

    def __getitem__(self, index):
        image = torch.randn(3, 8, 8, generator=torch.Generator().manual_seed(index))
        value = {"day": image, "day_valid_mask": torch.ones(2, 2, dtype=torch.bool),
                 "label": torch.tensor(index//2), "image_key": torch.tensor(index), "dataset": "robotcar"}
        if self.augmentation:
            value["aug"] = image * .7
        if self.fake:
            value.update(fake=image * .4, fake_valid_mask=value["day_valid_mask"])
        return value


@pytest.fixture
def training(tmp_path, monkeypatch):
    models = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(trainer, "PairedVPRDataset", Images)
    monkeypatch.setattr(trainer, "StableNightVPRDataset", Images)
    monkeypatch.setattr(trainer, "data_fingerprints", lambda *a, **kw: {})

    def build_model(*args, descriptor_dim=512, **kwargs):
        model = TinyModel(descriptor_dim)
        models.append(model)
        return model

    monkeypatch.setattr(trainer, "build_descriptor_model", build_model)
    cfg = VPRConfig(output_dir=str(tmp_path / "run"), train_domains="day", epochs=2, workers=0,
        amp=False, batch_size=4, gradient_accumulation_steps=2, sampler="pk_balanced", samples_per_epoch=12,
        datasets=("robotcar",), dataset_epoch_quotas={"robotcar": 12}, pk_places_per_batch=2, pk_images_per_place=2,
        selection_rule="fixed_budget_last", early_stopping=False, tensorboard_log_interval=100, tensorboard_image_interval=100)
    return ExperimentConfig(DataConfig(), vpr=cfg), models


def events(config):
    return EventAccumulator(str(Path(config.vpr.output_dir) / "tensorboard")).Reload()


def test_legacy_vpr_accumulated_loss_logs_first_last_update_without_fake_recall(training, monkeypatch):
    config, models = training
    values = iter([1., 3., 10.])

    def loss(descriptor, labels, classifier):
        value = descriptor.sum() * 0 + next(values)
        return value, {"classification_day": value}

    monkeypatch.setattr(trainer, "day_classification_loss", loss)
    trainer.train_vpr(config, smoke_epochs=1)
    logs = events(config)
    assert [(item.step, item.value) for item in logs.Scalars("train/loss")] == [(1, 2.), (2, 10.)]
    assert logs.Scalars("epoch_loss/loss")[0].value == pytest.approx(14./3)
    assert logs.Scalars("epoch_progress/epoch")[0].step == 2
    assert not any(tag.startswith("validation/") for tag in logs.Tags()["scalars"])
    assert models[0].calls == 3  # No extra encoder pass for logging.
    assert len(logs.Scalars("lr/backbone")) >= 1


def test_measured_validation_and_epoch_zero_are_logged_at_optimizer_steps(training, monkeypatch):
    config, _ = training
    config = replace(config, vpr=replace(config.vpr, selection_rule="validation_best", evaluate_epoch_zero=True))
    measured = iter([.4, .6])
    monkeypatch.setattr(trainer, "_evaluate_validation", lambda *a, **kw: {
        "recalls": {"1": next(measured), "5": .9}, "mrr": .5, "valid_queries": 10, "total_queries": 10,
    })
    trainer.train_vpr(config, smoke_epochs=1)
    logs = events(config)
    assert [e.step for e in logs.Scalars("validation/recall_at_1")] == [0, 2]
    assert [e.value for e in logs.Scalars("validation/recall_at_1")] == pytest.approx([.4, .6])
    assert logs.Scalars("validation/best_recall_at_1")[-1].value == pytest.approx(.6)


def test_stable_diagnostics_resume_steps_and_purge_future_events(training):
    config, models = training
    config = replace(config, vpr=replace(config.vpr, model_family="stable_night_salad", descriptor_dim=544,
        train_domains="day_aug", samples_per_epoch=4, dataset_epoch_quotas={"robotcar": 4}))
    trainer.train_vpr(config, smoke_epochs=1)
    logs = events(config)
    assert "train/day_predictor_valid_mean" in logs.Tags()["scalars"]
    assert "train/dustbin_gamma" in logs.Tags()["scalars"]
    assert "patches/day/reliability" in logs.Tags()["histograms"]
    assert "images/day_reliability" in logs.Tags()["images"]
    assert logs.Images("images/day_reliability")[0].height == 2  # Native patch grid.
    assert models[0].calls == 2  # Day/Aug, no diagnostic encoder pass.
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(Path(config.vpr.output_dir) / "tensorboard")) as writer:
        writer.add_scalar("train/loss", 999., 999)
    resume = replace(config, vpr=replace(config.vpr, resume=str(Path(config.vpr.output_dir) / "checkpoints/latest.pt")))
    trainer.train_vpr(resume)
    assert [item.step for item in events(config).Scalars("train/loss")] == [1, 2]


def test_logging_disabled_preserves_model_updates_and_creates_no_events(training):
    config, models = training
    config = replace(config, vpr=replace(config.vpr, model_family="stable_night_salad", descriptor_dim=544,
                                       train_domains="day_aug"))
    trainer.train_vpr(config, smoke_epochs=1)
    off = replace(config, vpr=replace(config.vpr, output_dir=str(Path(config.vpr.output_dir).parent / "off"), tensorboard_enabled=False))
    trainer.train_vpr(off, smoke_epochs=1)
    assert not (Path(off.vpr.output_dir) / "tensorboard").exists()
    for key, value in models[0].state_dict().items():
        assert torch.equal(value, models[1].state_dict()[key])
