import builtins
import hashlib
import json
from dataclasses import replace

import pytest
import torch

from a2b_lightvpr.checkpointing import checkpoint_config_matches
from a2b_lightvpr.config import DataConfig, ExperimentConfig, TENSORBOARD_FIELDS, apply_overrides
from a2b_lightvpr.training_logging import create_summary_writer, log_scalars


def config():
    return ExperimentConfig(DataConfig())


def test_logging_controls_do_not_change_old_experiment_identity_or_resume():
    original = config()
    saved = original.as_dict()
    for section in ("generator", "vpr"):
        for name in TENSORBOARD_FIELDS:
            saved[section].pop(name)
    before_hash = hashlib.sha256(json.dumps(saved, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert original.fingerprint() == before_hash
    changed = replace(original,
        vpr=replace(original.vpr, tensorboard_enabled=False, tensorboard_log_dir="somewhere", tensorboard_log_interval=1),
        generator=replace(original.generator, tensorboard_flush_secs=5, tensorboard_image_interval=0))
    assert changed.fingerprint() == before_hash
    assert checkpoint_config_matches({"config": saved, "config_fingerprint": "older"}, changed)
    assert not checkpoint_config_matches({"config": saved}, replace(changed, vpr=replace(changed.vpr, encoder_lr=2e-5)))


@pytest.mark.parametrize("expression", [
    "vpr.tensorboard_enabled=1", "generator.tensorboard_enabled=sometimes",
    "vpr.tensorboard_log_interval=0", "generator.tensorboard_flush_secs=-1",
    "vpr.tensorboard_image_interval=-1", "generator.tensorboard_image_interval=true",
    "vpr.tensorboard_log_interval=1.5", "generator.tensorboard_log_dir=123",
])
def test_invalid_logging_options_fail_validation(expression):
    with pytest.raises(ValueError, match="tensorboard"):
        apply_overrides(config(), [expression])


def test_disabled_writer_needs_no_tensorboard_import_or_directory(tmp_path, monkeypatch):
    old_import = builtins.__import__

    def importing(name, *args, **kwargs):
        if name.startswith("torch.utils.tensorboard"):
            raise ImportError("simulated missing TensorBoard")
        return old_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    disabled = replace(config().vpr, tensorboard_enabled=False)
    with create_summary_writer(disabled, tmp_path / "disabled") as writer:
        assert writer is None
        log_scalars(writer, {"loss": torch.tensor(1.)}, 1)
    assert not (tmp_path / "disabled").exists()
    with pytest.raises(RuntimeError, match="pip install tensorboard"):
        with create_summary_writer(config().vpr, tmp_path / "enabled"):
            pass


def test_writer_flushes_and_closes_on_exception(tmp_path, monkeypatch):
    from torch.utils import tensorboard
    called = []

    class Writer:
        def __init__(self, **kwargs):
            called.append(("create", kwargs))
        def add_text(self, *args, **kwargs):
            called.append(("text", kwargs))
        def flush(self):
            called.append(("flush", {}))
        def close(self):
            called.append(("close", {}))

    monkeypatch.setattr(tensorboard, "SummaryWriter", Writer)
    with pytest.raises(ValueError, match="training failed"):
        with create_summary_writer(config().vpr, tmp_path, purge_step=5):
            raise ValueError("training failed")
    assert called[0][1]["purge_step"] == 5
    assert called[0][1]["log_dir"] == str(tmp_path / "tensorboard")
    assert [name for name, _ in called][-2:] == ["flush", "close"]


def test_real_events_are_finite_detached_and_resume_hides_stale_future(tmp_path, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.summary.writer import event_file_writer
    # Exercise the filename boundary that otherwise depends on preceding tests.
    monkeypatch.setattr(event_file_writer, "_global_uid", event_file_writer.AtomicCounter(9))
    cfg = replace(config().vpr, tensorboard_log_dir=str(tmp_path / "custom"))
    value = torch.tensor(2., requires_grad=True)
    with create_summary_writer(cfg, tmp_path) as writer:
        log_scalars(writer, {"loss": value, "not_measured": None, "bad": float("nan")}, 1, "train/")
        log_scalars(writer, {"loss": 999.}, 999, "train")
    assert value.grad is None
    with create_summary_writer(cfg, tmp_path, purge_step=2) as writer:
        log_scalars(writer, {"loss": 3.}, 2, "train")
    events = EventAccumulator(str(tmp_path / "custom")).Reload()
    assert "train/not_measured" not in events.Tags()["scalars"]
    assert "train/bad" not in events.Tags()["scalars"]
    assert [(event.step, event.value) for event in events.Scalars("train/loss")] == [(1, 2.), (2, 3.)]
