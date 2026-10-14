from dataclasses import replace

import torch

from a2b_lightvpr.checkpointing import (
    checkpoint_config_matches,
    data_fingerprints,
    file_sha256,
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
    config = ExperimentConfig(DataConfig())
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


def test_old_custom_validation_checkpoint_cannot_resume_as_public_validation_run():
    config = ExperimentConfig(DataConfig())
    saved = config.as_dict()
    saved["data"].pop("robotcar_seasons_root")
    saved["data"].update(robotcar_manifest="old.csv", robotcar_protocol="old.json")
    saved["evaluation"]["protocol"] = "project"
    state = {"config": saved}
    assert not checkpoint_config_matches(state, config, ("data", "vpr", "evaluation"))
    assert not checkpoint_config_matches(state, config)
    assert checkpoint_config_matches(state, config, ("data", "generator"))
    changed_manifest = replace(config, data=replace(config.data, generator_manifest="different.csv"))
    assert not checkpoint_config_matches(state, changed_manifest, ("data", "generator"))
    changed_crop = replace(config, data=replace(config.data, crop_size=256))
    assert not checkpoint_config_matches(state, changed_crop, ("data", "generator"))
    changed_objective = replace(config, generator=replace(config.generator, objective="different"))
    assert not checkpoint_config_matches(state, changed_objective, ("data", "generator"))


def test_generator_fingerprint_requires_only_its_training_manifest(tmp_path):
    manifest = tmp_path / "generator.csv"
    manifest.write_text("training source bytes")
    config = ExperimentConfig(DataConfig(generator_manifest=str(manifest), vpr_source_manifest="absent.csv"))
    assert data_fingerprints(config, purpose="generator") == {"generator_manifest": file_sha256(manifest)}


def test_source_metric_checkpoint_does_not_depend_on_retired_generator_manifest():
    from a2b_lightvpr.config import load_config
    config = load_config('configs/experiments/metric/salad_small_gsv_day_only_544.yaml')
    saved = config.as_dict()
    saved['data']['generator_manifest'] = 'deleted/qsattn_training.csv'
    state = {'config': saved, 'config_fingerprint': 'historical-config'}
    assert checkpoint_config_matches(state, config, ('data', 'vpr', 'evaluation'))
    saved['data']['vpr_source_manifest'] = 'different_source_labels.csv'
    assert not checkpoint_config_matches(state, config, ('data', 'vpr', 'evaluation'))
