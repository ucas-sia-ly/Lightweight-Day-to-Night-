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


@pytest.mark.parametrize("dimension", [128, 512, 4096])
def test_formal_dimension_configs_are_strictly_paired(dimension):
    baseline = load_config(f"configs/vpr/day_fake_{dimension}.yaml")
    candidate = load_config(f"configs/vpr/cdnd_salad_{dimension}.yaml")
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


def test_relation_teacher_requires_relation_training():
    config = base_config()
    with pytest.raises(ValueError, match="Only cross_domain_relation"):
        validate_config(replace(config, vpr=replace(config.vpr, relation_teacher="boq")))


def test_reference_config_still_loads():
    assert load_config("configs/references/official.yaml")


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


def test_official_generator_config_has_no_public_seed():
    config = load_config("configs/generator/qsattn_official_objective_150k.yaml")
    assert config.generator.objective == "qsattn"
    assert config.generator.max_steps == 150_000
    assert config.generator.qs_nce_identity
    assert not hasattr(config.generator, "seed")


def test_robotcar_only_ablation_configs_share_budget_and_initialization():
    from a2b_lightvpr.config import load_config
    configs = [load_config(f'configs/experiments/single_dataset/robotcar_only_{name}.yaml') for name in ('day_only','day_fake','cdnd')]
    for c in configs:
        assert c.vpr.datasets == ('robotcar',)
        assert c.vpr.dataset_epoch_quotas == {'robotcar': 8944}
        assert c.vpr.samples_per_epoch % c.vpr.batch_size == 0
        assert c.vpr.epochs == 20 and c.generator.max_steps == 30000
        assert not c.vpr.early_stopping and c.vpr.epoch_one_max_recall_drop == 0
    assert len({c.vpr.init_checkpoint for c in configs}) == 1
    assert len({c.data.generated_root for c in configs}) == 1
    assert [c.vpr.relation_teacher for c in configs] == ['none','none','salad']
    assert [c.vpr.train_domains for c in configs] == ['day','day_fake','day_fake']


def test_svox_msls_only_configs_use_fixed_budget_without_robotcar_validation():
    from a2b_lightvpr.config import load_config, validate_config
    from dataclasses import replace
    import pytest
    for ds, quota in [('svox',33520),('msls',75840)]:
        for group in ('day_only','day_fake','cdnd'):
            c=load_config(f'configs/experiments/single_dataset/{ds}_only_{group}.yaml')
            assert c.vpr.datasets==(ds,)
            assert c.vpr.selection_rule=='fixed_budget_last'
            assert c.vpr.samples_per_epoch==quota
            assert not c.vpr.evaluate_epoch_zero
            assert c.vpr.relation_teacher==('salad' if group=='cdnd' else 'none')
            with pytest.raises(ValueError,match='Fixed-budget'):
                validate_config(replace(c,vpr=replace(c.vpr,early_stopping=True)))
