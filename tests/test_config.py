from dataclasses import replace

import pytest

from a2b_lightvpr.config import DataConfig, ExperimentConfig, apply_overrides, load_config, validate_config


def base_config():
    return ExperimentConfig(DataConfig("robot.csv", "protocol.json", "external.csv"))


def test_only_official_qsattn_objective_is_allowed():
    with pytest.raises(ValueError, match="qsattn"):
        validate_config(replace(base_config(), generator=replace(base_config().generator, objective="unsupported")))


@pytest.mark.parametrize("key", ["generator.seed=1", "vpr.seed=1", "tracking.enabled=true"])
def test_public_config_rejects_seed_fields(key):
    with pytest.raises(ValueError, match="Unknown config override"):
        apply_overrides(base_config(), [key])


def test_resume_paths_do_not_change_experiment_fingerprint():
    config = base_config()
    changed = replace(
        config,
        generator=replace(config.generator, resume="generator.pt"),
        vpr=replace(config.vpr, resume="vpr.pt"),
    )
    assert changed.fingerprint() == config.fingerprint()


@pytest.mark.parametrize("dimension", [128, 512, 4096])
def test_formal_dimension_configs_are_strictly_paired(dimension):
    baseline = load_config(f"configs/vpr_day_fake_pk_{dimension}.yaml")
    candidate = load_config(f"configs/vpr_cdnd_salad_{dimension}.yaml")
    assert baseline.data == candidate.data
    assert baseline.vpr.descriptor_dim == candidate.vpr.descriptor_dim == dimension
    assert baseline.vpr.cross_domain_aux == "none"
    assert baseline.vpr.relation_teacher == "none"
    assert candidate.vpr.cross_domain_aux == "cross_domain_relation"
    assert candidate.vpr.relation_teacher == "salad"
    assert candidate.vpr.relation_weight == pytest.approx(0.5)
    assert candidate.vpr.relation_day_weight == pytest.approx(0.5)
    assert candidate.vpr.relation_temperature == pytest.approx(0.07)
    assert candidate.vpr.relation_queue_size == 2048
    ignored = {
        "output_dir", "cross_domain_aux", "relation_teacher", "relation_teacher_repo",
        "relation_teacher_checkpoint", "relation_teacher_cache", "relation_teacher_input_size",
    }
    baseline_values = {
        key: value for key, value in baseline.as_dict()["vpr"].items() if key not in ignored
    }
    candidate_values = {
        key: value for key, value in candidate.as_dict()["vpr"].items() if key not in ignored
    }
    assert baseline_values == candidate_values
    assert baseline.vpr.sampler == "pk_balanced"
    assert (baseline.vpr.pk_places_per_batch, baseline.vpr.pk_images_per_place) == (4, 2)
    assert baseline.vpr.samples_per_epoch == 60_000
    assert baseline.vpr.epochs == 20


def test_only_salad_can_be_the_training_teacher():
    config = base_config()
    with pytest.raises(ValueError, match="none or salad"):
        validate_config(replace(config, vpr=replace(config.vpr, relation_teacher="boq")))


def test_reference_config_still_loads():
    assert load_config("configs/reference_models_extended.yaml")


def test_official_generator_config_has_no_public_seed():
    config = load_config("configs/generator_qsattn_official_objective_150k.yaml")
    assert config.generator.objective == "qsattn"
    assert config.generator.max_steps == 150_000
    assert config.generator.qs_nce_identity
    assert not hasattr(config.generator, "seed")
