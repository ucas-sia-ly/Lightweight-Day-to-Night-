from dataclasses import replace

import pytest

from a2b_lightvpr.config import DataConfig, ExperimentConfig, apply_overrides, load_config, validate_config


def base_config():
    return ExperimentConfig(DataConfig(generator_manifest="external.csv"))


def test_only_official_qsattn_objective_is_allowed():
    with pytest.raises(ValueError, match="qsattn"):
        validate_config(replace(base_config(), generator=replace(base_config().generator, objective="unsupported")))


@pytest.mark.parametrize("key", ["generator.seed=1", "vpr.seed=1", "tracking.enabled=true"])
def test_public_config_rejects_seed_fields(key):
    with pytest.raises(ValueError, match="Unknown config override"):
        apply_overrides(base_config(), [key])


def test_vpr_random_seed_and_ablation_controls_are_public():
    config = apply_overrides(base_config(), [
        "vpr.random_seed=2",
        "vpr.train_domains=day",
        "vpr.relation_mask_self=false",
        "vpr.relation_queue_scope=shared",
    ])
    assert config.vpr.random_seed == 2
    assert config.vpr.train_domains == "day"
    assert not config.vpr.relation_mask_self
    assert config.vpr.relation_queue_scope == "shared"


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("vpr.random_seed=-1", "non-negative"),
        ("vpr.train_domains=night", "day or day_fake"),
        ("vpr.relation_queue_scope=global", "dataset or shared"),
        ("vpr.relation_mask_self=1", "must be a boolean"),
    ],
)
def test_vpr_experiment_controls_reject_invalid_values(key, message):
    with pytest.raises(ValueError, match=message):
        apply_overrides(base_config(), [key])


def test_resume_paths_do_not_change_experiment_fingerprint():
    config = base_config()
    changed = replace(
        config,
        generator=replace(config.generator, resume="generator.pt"),
        vpr=replace(config.vpr, resume="vpr.pt"),
    )
    assert changed.fingerprint() == config.fingerprint()


def test_metric_configs_have_identical_budget_and_model():
    baseline = load_config("configs/experiments/metric/salad_small_gsv_day_only_544.yaml")
    candidate = load_config("configs/experiments/metric/salad_small_gsv_day_aug_544.yaml")
    assert baseline.data == candidate.data
    assert baseline.vpr.descriptor_dim == candidate.vpr.descriptor_dim == 544
    assert baseline.vpr.model_family == candidate.vpr.model_family == "salad"
    assert baseline.vpr.training_objective == candidate.vpr.training_objective == "multi_similarity"
    assert baseline.vpr.cross_domain_aux == candidate.vpr.cross_domain_aux == "none"
    assert baseline.vpr.relation_teacher == candidate.vpr.relation_teacher == "none"
    ignored = {"output_dir", "train_domains"}
    baseline_values = {
        key: value for key, value in baseline.as_dict()["vpr"].items() if key not in ignored
    }
    candidate_values = {
        key: value for key, value in candidate.as_dict()["vpr"].items() if key not in ignored
    }
    assert baseline_values == candidate_values
    assert baseline.vpr.sampler == "pk_balanced" and baseline.vpr.metric_within_city
    assert (baseline.vpr.pk_places_per_batch, baseline.vpr.pk_images_per_place) == (32, 4)
    assert baseline.vpr.samples_per_epoch == 251392
    assert baseline.vpr.epochs == 2
    assert baseline.vpr.workers == 0 and baseline.vpr.require_cuda


def test_relation_teacher_requires_relation_training():
    config = base_config()
    with pytest.raises(ValueError, match="Only cross_domain_relation"):
        validate_config(replace(config, vpr=replace(config.vpr, relation_teacher="boq")))


def test_reference_config_still_loads():
    assert load_config("configs/references/salad.yaml")


def test_config_uses_public_robotcar_and_day_validation_by_default():
    config = base_config()
    assert config.data.robotcar_seasons_root.endswith("/datasets/RobotCar-Seasons")
    assert not hasattr(config.data, "robotcar_manifest")
    assert not hasattr(config.data, "robotcar_protocol")
    assert config.evaluation.protocol == "standard"
    assert config.evaluation.datasets == ("msls_day_val",)
    assert config.evaluation.positive_radius_m == 25.0
    assert config.evaluation.positive_yaw_deg is None


@pytest.mark.parametrize("key", ["data.robotcar_manifest=old.csv", "data.robotcar_protocol=old.json"])
def test_retired_test_protocol_fields_are_not_active_overrides(key):
    with pytest.raises(ValueError, match="Unknown config override"):
        apply_overrides(base_config(), [key])


@pytest.mark.parametrize("key,message", [
    ("evaluation.protocol=project", "custom RobotCar protocols are retired"),
    ("evaluation.protocol=paper", "custom RobotCar protocols are retired"),
    ("evaluation.datasets=[robotcar_seasons_night]", "test datasets cannot select checkpoints"),
    ("evaluation.datasets=[tokyo_night]", "test datasets cannot select checkpoints"),
])
def test_test_sets_cannot_replace_checkpoint_validation(key, message):
    with pytest.raises(ValueError, match=message):
        apply_overrides(base_config(), [key])
