"""Exercise CDND's real loss/queue branch with small, deterministic descriptors."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from a2b_lightvpr.vpr import trainer
from test_vpr_tensorboard import Images, events, training


class TeacherImages(Images):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["teacher_descriptor"] = F.normalize(
            torch.randn(8, generator=torch.Generator().manual_seed(index + 100)), dim=0,
        )
        return sample


def cdnd_config(training, monkeypatch):
    config, models = training
    monkeypatch.setattr(trainer, "PairedVPRDataset", TeacherImages)
    monkeypatch.setattr(trainer, "TeacherDescriptorCache", lambda *a: SimpleNamespace(fingerprint="tiny-teacher"))
    return replace(config, vpr=replace(
        config.vpr, train_domains="day_fake", cross_domain_aux="cross_domain_relation",
        relation_teacher="salad", relation_queue_size=6, relation_weight=0.3,
        relation_day_weight=0.2,
    )), models


def test_cdnd_logs_actual_relation_losses_and_preserves_queue_updates(training, monkeypatch):
    config, models = cdnd_config(training, monkeypatch)
    candidate_counts = []
    queue_type = trainer.DatasetNeighborhoodQueue

    class Queue(queue_type):
        def candidates(self, *args):
            result = super().candidates(*args)
            candidate_counts.append(len(result.labels))
            return result

    monkeypatch.setattr(trainer, "DatasetNeighborhoodQueue", Queue)
    trainer.train_vpr(config, smoke_epochs=1)
    logs = events(config)
    tags = (
        "classification_day", "classification_fake", "relation_fake_to_day",
        "relation_day_to_day", "relation", "relation_valid_anchors", "loss",
    )
    values = {}
    for tag in tags:
        logged = logs.Scalars(f"train/{tag}")
        assert [item.step for item in logged] == [1, 2]
        values[tag] = [item.value for item in logged]
        assert len(logs.Scalars(f"epoch_loss/{tag}")) == 1
    for index in range(2):
        expected_relation = (
            values["relation_fake_to_day"][index]
            + config.vpr.relation_day_weight * values["relation_day_to_day"][index]
        )
        expected_loss = 0.5 * (
            values["classification_day"][index] + values["classification_fake"][index]
        ) + config.vpr.relation_weight * expected_relation
        assert values["relation"][index] == pytest.approx(expected_relation, abs=1e-5)
        assert values["loss"][index] == pytest.approx(expected_loss, abs=1e-5)
    assert values["relation_valid_anchors"] == [4., 4.]
    assert candidate_counts == [4, 8, 10]
    assert models[0].calls == 6  # Three day/fake batches, no logging forward.
    state = torch.load(config.vpr.output_dir + "/checkpoints/latest.pt", weights_only=False)
    assert state["optimizer_step"] == 2
    assert len(state["relation_queue_state"]["entries"]["robotcar"]["labels"]) == 6


def test_cdnd_failure_closes_writer_through_public_training_entry(training, monkeypatch):
    from torch.utils import tensorboard
    config, _ = cdnd_config(training, monkeypatch)
    calls = []

    class Writer:
        def __init__(self, **kwargs):
            calls.append("open")
        def add_text(self, *args, **kwargs):
            calls.append("text")
        def flush(self):
            calls.append("flush")
        def close(self):
            calls.append("close")

    def fail(*args, **kwargs):
        raise RuntimeError("CDND relation failed")

    monkeypatch.setattr(tensorboard, "SummaryWriter", Writer)
    monkeypatch.setattr(trainer, "cross_domain_neighborhood_loss", fail)
    with pytest.raises(RuntimeError, match="CDND relation failed"):
        trainer.train_vpr(config, smoke_epochs=1)
    assert calls == ["open", "text", "flush", "close"]


def test_invalid_cdnd_resume_does_not_open_or_purge_writer(training, monkeypatch):
    from pathlib import Path
    from torch.utils import tensorboard
    config, _ = cdnd_config(training, monkeypatch)
    trainer.train_vpr(config, smoke_epochs=1)
    path = Path(config.vpr.output_dir) / "checkpoints/latest.pt"
    state = torch.load(path, weights_only=False)
    state.pop("relation_queue_state")
    torch.save(state, path)
    config = replace(config, vpr=replace(config.vpr, resume=str(path)))

    def unexpected_writer(**kwargs):
        raise AssertionError("Writer opened before resume validation completed")

    monkeypatch.setattr(tensorboard, "SummaryWriter", unexpected_writer)
    with pytest.raises(ValueError, match="missing relation queue state"):
        trainer.train_vpr(config)
    assert [item.step for item in events(config).Scalars("train/loss")] == [1, 2]
