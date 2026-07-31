from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from a2b_lightvpr.config import DataConfig, ExperimentConfig
from a2b_lightvpr.generation_evaluation import (
    _load_generator,
    bootstrap_summary,
    polynomial_mmd,
)
from a2b_lightvpr.generator.networks import ResnetGenerator


def _config():
    return ExperimentConfig(DataConfig("robot.csv", "protocol.json"))


def test_bootstrap_summary_is_deterministic_and_contains_mean():
    first = bootstrap_summary([1.0, 2.0, 3.0], samples=100, seed=4)
    second = bootstrap_summary([1.0, 2.0, 3.0], samples=100, seed=4)
    assert first == second
    assert first["mean"] == pytest.approx(2.0)
    assert first["count"] == 3


def test_polynomial_mmd_is_near_zero_for_same_features():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(20, 8))
    assert polynomial_mmd(features, features) == pytest.approx(0.0, abs=0.1)


def test_load_project_training_state_with_numpy_rng(tmp_path):
    output_dir = tmp_path / "generator"
    checkpoint = output_dir / "checkpoints" / "state_0050000.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({
        "kind": "generator_training_state",
        "step": 50_000,
        "generator": ResnetGenerator().state_dict(),
        "rng": {"numpy": np.random.get_state()},
    }, checkpoint)
    config = _config()
    config = replace(config, generator=replace(config.generator, output_dir=str(output_dir)))

    _, state = _load_generator(checkpoint, torch.device("cpu"), config)

    assert state["kind"] == "generator_training_state"
    assert state["step"] == 50_000


def test_load_generator_rejects_unsupported_kind(tmp_path):
    checkpoint = tmp_path / "unsupported.pt"
    torch.save({"kind": "vpr_checkpoint", "generator": {}}, checkpoint)

    with pytest.raises(ValueError, match="Expected a generator"):
        _load_generator(checkpoint, torch.device("cpu"), _config())


def test_external_inference_checkpoint_keeps_weights_only_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "generator_final.pt"
    checkpoint.touch()
    payload = {
        "kind": "generator_inference",
        "generator": ResnetGenerator().state_dict(),
    }
    load_calls = []

    def fake_load(path, *, map_location, weights_only):
        load_calls.append((Path(path), map_location, weights_only))
        return payload

    monkeypatch.setattr(torch, "load", fake_load)

    _load_generator(checkpoint, torch.device("cpu"), _config())

    assert load_calls == [(checkpoint.resolve(), "cpu", True)]
