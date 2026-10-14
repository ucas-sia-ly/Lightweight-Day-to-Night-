from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from a2b_lightvpr.config import DataConfig, ExperimentConfig, GeneratorConfig
from a2b_lightvpr.generator import trainer


class TinyGenerator(nn.Module):
    forward_calls = 0

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def encode_features(self, inputs):
        features = inputs * self.scale
        return features, [features]

    def forward(self, inputs, *, return_features=False):
        type(self).forward_calls += 1
        latent, features = self.encode_features(inputs)
        fake = torch.tanh(latent)
        return (fake, features) if return_features else fake


class TinyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, inputs):
        return inputs.mean(dim=1, keepdim=True) * self.scale


class TinyObjective(nn.Module):
    def generator_loss(self, source, fake, source_features, fake_features):
        return (source_features[0] - fake_features[0]).square().mean()

    def identity_loss(self, target_features, identity_features):
        return (target_features[0] - identity_features[0]).square().mean()


class TinyDataset(Dataset):
    def __init__(self, config, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Normalized generator input uses [-1, 1], and image previews undo it.
        source = torch.linspace(-1, 1, 3 * 4 * 4).reshape(3, 4, 4)
        return {"source": source, "target": source.flip(-1)}


class RecordingWriter:
    def __init__(self):
        self.scalars = []
        self.images = []
        self.closed = False
        self.flushed = False

    def add_scalar(self, tag, scalar_value, global_step=None, **kwargs):
        assert not isinstance(scalar_value, torch.Tensor) or not scalar_value.requires_grad
        self.scalars.append((tag, float(scalar_value), global_step))

    def add_images(self, tag, images, global_step=None, **kwargs):
        assert not images.requires_grad and images.grad_fn is None
        assert images.device.type == "cpu"
        assert images.shape[0] <= 3
        assert images.min() >= 0 and images.max() <= 1
        self.images.append((tag, images.clone(), global_step))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


@pytest.fixture
def tiny_training(monkeypatch, tmp_path):
    monkeypatch.setattr(trainer, "_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(trainer, "ResnetGenerator", TinyGenerator)
    monkeypatch.setattr(trainer, "PatchDiscriminator", TinyDiscriminator)
    monkeypatch.setattr(trainer, "build_content_objective", lambda *a, **kw: TinyObjective())
    monkeypatch.setattr(trainer, "BalancedGeneratorDataset", TinyDataset)
    monkeypatch.setattr(trainer, "data_fingerprints", lambda *a, **kw: {})
    TinyGenerator.forward_calls = 0
    writers = []
    boundaries = []

    @contextmanager
    def create_writer(cfg, output_dir, purge_step=None):
        boundaries.append(purge_step)
        if not cfg.tensorboard_enabled:
            yield None
            return
        writer = RecordingWriter()
        writers.append(writer)
        try:
            yield writer
        finally:
            writer.flush()
            writer.close()

    monkeypatch.setattr(trainer, "create_summary_writer", create_writer)
    config = ExperimentConfig(
        data=DataConfig(),
        generator=GeneratorConfig(
            output_dir=str(tmp_path / "generator"),
            max_steps=5,
            batch_size=4,
            workers=0,
            amp=False,
            checkpoint_every=5,
            qs_nce_identity=True,
            tensorboard_log_interval=3,
            tensorboard_image_interval=3,
        ),
    )
    return config, writers, boundaries


def test_generator_logs_loss_lr_amp_and_preview_without_extra_forward(tiny_training):
    config, writers, boundaries = tiny_training
    path = trainer.train_generator(config)
    assert path.is_file()
    assert boundaries == [None]
    writer = writers[0]
    assert writer.closed and writer.flushed
    assert {step for _, _, step in writer.scalars} == {1, 3, 5}
    values = {tag: value for tag, value, step in writer.scalars if step == 5}
    assert set(values) == {
        "train/loss_g", "train/loss_d", "train/loss_gan", "train/loss_content",
        "train/loss_content_source", "train/loss_content_identity", "train/amp_scale",
        "train/amp_step_skipped", "lr/generator", "lr/discriminator",
    }
    assert values["train/loss_g"] == pytest.approx(
        values["train/loss_gan"] + config.generator.content_weight * values["train/loss_content"]
    )
    assert values["train/loss_content"] == pytest.approx(
        0.5 * (values["train/loss_content_source"] + values["train/loss_content_identity"])
    )
    assert values["train/amp_scale"] == 1.0
    assert values["train/amp_step_skipped"] == 0.0
    assert values["lr/generator"] == values["lr/discriminator"] == 0.0
    assert {tag for tag, _, _ in writer.images} == {
        "images/day", "images/fake_night", "images/real_night", "images/identity_night"
    }
    assert {step for _, _, step in writer.images} == {1, 3, 5}
    assert TinyGenerator.forward_calls == 10  # Source and identity, exactly once each per step.


def test_generator_resume_keeps_absolute_steps_and_purges_future_events(tiny_training):
    config, writers, boundaries = tiny_training
    trainer.train_generator(config, smoke_steps=2)
    resume = str(config.generator.output_dir + "/checkpoints/latest.pt")
    resumed_config = replace(config, generator=replace(config.generator, resume=resume))
    trainer.train_generator(resumed_config)
    assert boundaries == [None, 3]
    assert {step for _, _, step in writers[0].scalars} == {1, 2}
    assert {step for _, _, step in writers[1].scalars} == {3, 5}
    assert all(writer.closed and writer.flushed for writer in writers)


def test_generator_tensorboard_disabled_and_without_identity(tiny_training):
    config, writers, boundaries = tiny_training
    disabled = replace(
        config,
        generator=replace(config.generator, tensorboard_enabled=False, qs_nce_identity=False),
    )
    trainer.train_generator(disabled, smoke_steps=1)
    assert writers == [] and boundaries == [None]
    assert TinyGenerator.forward_calls == 1


def test_generator_image_cadence_is_separate_and_can_be_disabled(tiny_training):
    config, writers, _ = tiny_training
    separate = replace(
        config,
        generator=replace(config.generator, tensorboard_image_interval=4, qs_nce_identity=False),
    )
    trainer.train_generator(separate)
    assert {step for _, _, step in writers[0].scalars} == {1, 3, 5}
    assert {step for _, _, step in writers[0].images} == {1, 4, 5}
    assert "train/loss_content_identity" not in {tag for tag, _, _ in writers[0].scalars}
    assert "images/identity_night" not in {tag for tag, _, _ in writers[0].images}
    assert TinyGenerator.forward_calls == 5
    no_images = replace(separate, generator=replace(separate.generator, tensorboard_image_interval=0))
    trainer.train_generator(no_images, smoke_steps=1)
    assert writers[1].scalars and not writers[1].images


def test_generator_training_failure_exits_writer_context(tiny_training, monkeypatch):
    config, writers, _ = tiny_training

    def fail(*args, **kwargs):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(TinyObjective, "generator_loss", fail)
    with pytest.raises(RuntimeError, match="simulated training failure"):
        trainer.train_generator(config)
    assert writers[0].closed and writers[0].flushed


def test_generator_writes_readable_tensorboard_events(tiny_training, monkeypatch):
    event_accumulator = pytest.importorskip("tensorboard.backend.event_processing.event_accumulator")
    from a2b_lightvpr.training_logging import create_summary_writer

    config, _, _ = tiny_training
    monkeypatch.setattr(trainer, "create_summary_writer", create_summary_writer)
    trainer.train_generator(config, smoke_steps=1)
    log_dir = Path(config.generator.output_dir) / "tensorboard"
    assert list(log_dir.glob("events.out.tfevents.*"))
    events = event_accumulator.EventAccumulator(str(log_dir)).Reload()
    assert events.Scalars("train/loss_g")[0].step == 1
    assert events.Scalars("lr/generator")[0].step == 1
    assert events.Images("images/fake_night")[0].step == 1
