from dataclasses import replace

import torch

from a2b_lightvpr.checkpointing import (
    checkpoint_config_matches,
    restore_rng_state,
    rng_state,
)
from a2b_lightvpr.config import DataConfig, ExperimentConfig


def test_rng_state_round_trip_restores_torch_sequence():
    torch.manual_seed(123)
    state = rng_state()
    expected = torch.rand(4)
    torch.rand(8)
    restore_rng_state(state)
    assert torch.equal(torch.rand(4), expected)


def test_legacy_seed_zero_config_maps_to_new_experiment_controls():
    config = ExperimentConfig(DataConfig("robot.csv", "protocol.json"))
    saved = config.as_dict()
    saved["vpr"].pop("random_seed")
    saved["vpr"].pop("relation_mask_self")
    saved["vpr"].pop("relation_queue_scope")
    saved["vpr"].pop("backbone_name")
    saved["vpr"].pop("backbone_repo")
    saved["vpr"].pop("backbone_checkpoint")
    saved["vpr"].pop("backbone_trainable_blocks")
    saved["vpr"]["seed"] = 0
    state = {"config": saved, "config_fingerprint": "legacy"}
    assert checkpoint_config_matches(state, config, ("data", "vpr", "evaluation"))

    seed_one = replace(config, vpr=replace(config.vpr, random_seed=1))
    assert not checkpoint_config_matches(state, seed_one, ("data", "vpr", "evaluation"))
