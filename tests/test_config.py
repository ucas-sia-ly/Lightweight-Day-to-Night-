from dataclasses import replace

import pytest

from a2b_lightvpr.config import (
    DataConfig,
    ExperimentConfig,
    apply_overrides,
    load_config,
    validate_config,
)


def base_config():
    return ExperimentConfig(DataConfig("robot.csv", "protocol.json", "external.csv"))


def test_content_objective_is_mutually_exclusive():
    config = base_config()
    updated = apply_overrides(config, ["generator.objective=qsattn"])
    assert updated.generator.objective == "qsattn"


def test_unknown_objective_is_rejected():
    config = base_config()
    with pytest.raises(ValueError, match="objective"):
        validate_config(replace(config, generator=replace(config.generator, objective="all")))


def test_descriptor_contract_is_fixed():
    config = base_config()
    with pytest.raises(ValueError, match="512"):
        validate_config(replace(config, vpr=replace(config.vpr, descriptor_dim=256)))


def test_resume_path_does_not_change_experiment_fingerprint():
    config = base_config()
    resumed = replace(config, generator=replace(config.generator, resume="state.pt"))
    assert resumed.fingerprint() == config.fingerprint()


def test_removed_generator_relation_field_is_rejected():
    with pytest.raises(ValueError, match="Unknown config override"):
        apply_overrides(base_config(), ["generator.vpr_relation_weight=0.1"])


def test_screening_resolution_is_restricted_to_protocol_values():
    config = base_config()
    with pytest.raises(ValueError, match="resolution"):
        validate_config(
            replace(
                config,
                vpr=replace(config.vpr, image_height=384, image_width=512),
            )
        )


def test_config_rejects_512_square_input():
    config = base_config()
    with pytest.raises(ValueError, match="resolution"):
        validate_config(replace(config, vpr=replace(config.vpr, image_height=512, image_width=512)))


def test_tracking_settings_do_not_change_training_fingerprint():
    config = base_config()
    tracked = apply_overrides(
        config,
        [
            "tracking.enabled=true",
            "tracking.project=research",
            "tracking.tags=[pilot, negcut]",
        ],
    )
    assert tracked.tracking.tags == ("pilot", "negcut")
    assert tracked.fingerprint() == config.fingerprint()


def test_unknown_vpr_scheduler_is_rejected():
    with pytest.raises(ValueError, match="scheduler"):
        apply_overrides(base_config(), ["vpr.scheduler=linear"])

def test_v3_requires_relation_teacher():
    with pytest.raises(ValueError, match="V3 requires"):
        validate_config(replace(base_config(), vpr=replace(base_config().vpr, loss_profile="v3")))


def test_non_v3_rejects_relation_teacher():
    with pytest.raises(ValueError, match="Only V3"):
        validate_config(
            replace(base_config(), vpr=replace(base_config().vpr, relation_teacher="self"))
        )


def test_removed_finetune_fields_are_rejected():
    with pytest.raises(ValueError, match="Unknown config override"):
        apply_overrides(base_config(), ["vpr.train_datasets=[robotcar]"])


def test_removed_paired_edge_field_is_rejected():
    with pytest.raises(ValueError, match="Unknown config override"):
        apply_overrides(base_config(), ["generator.paired_edge_weight=0.5"])


@pytest.mark.parametrize(
    "path,backbone",
    [
        ("configs/vpr_screen_efficientnet_b0_three_320x480.yaml", "efficientnet_b0"),
        ("configs/vpr_screen_regnet_y_800mf_three_320x480.yaml", "regnet_y_800mf"),
        ("configs/vpr_screen_regnet_y_1_6gf_three_320x480.yaml", "regnet_y_1_6gf"),
        ("configs/vpr_screen_convnext_tiny_three_320x480.yaml", "convnext_tiny"),
    ],
)
def test_three_dataset_backbone_screen_configs(path, backbone):
    config = load_config(path)
    assert config.vpr.backbone == backbone
    assert config.vpr.samples_per_epoch == 12000
    assert config.vpr.min_class_images_by_dataset["robotcar"] == 2
    assert config.vpr.dataset_epoch_quotas == {"msls": 4000, "robotcar": 4000, "svox": 4000}
    assert config.vpr.sampler == "rolling_balanced"


def test_clean_generator_excludes_muses_and_edge():
    config = load_config("configs/generator_qsattn_clean_night.yaml")
    assert "muses" not in config.data.dataset_weights
    assert config.data.dataset_weights == {
        "darkdriving": 1,
        "robotcar": 6,
        "bdd100k": 6,
        "msls": 3,
        "svox": 1,
    }
    assert config.generator.paired_probability == pytest.approx(0.2)
    assert not hasattr(config.generator, "paired_edge_weight")


@pytest.mark.parametrize(
    "path,family,dimension,batch,accum",
    [
        ("configs/vpr_cosplace_r50_512_three_320x480.yaml", "cosplace", 512, 8, 2),
        ("configs/vpr_eigenplaces_r50_512_three_320x480.yaml", "eigenplaces", 512, 8, 2),
        ("configs/vpr_eigenplaces_r50_2048_three_320x480.yaml", "eigenplaces", 2048, 8, 2),
        ("configs/vpr_mixvpr_r50_512_three_320x320.yaml", "mixvpr", 512, 8, 2),
        ("configs/vpr_mixvpr_r50_4096_three_320x320.yaml", "mixvpr", 4096, 4, 4),
    ],
)
def test_official_adaptation_configs_share_three_dataset_protocol(path, family, dimension, batch, accum):
    config = load_config(path)
    assert config.vpr.model_family == family
    assert config.vpr.descriptor_dim == dimension
    assert config.vpr.samples_per_epoch == 12000
    assert sum(config.vpr.dataset_epoch_quotas.values()) == 12000
    assert config.vpr.sampler == "rolling_balanced"
    assert config.vpr.epochs == 60
    assert config.vpr.batch_size == batch
    assert config.vpr.gradient_accumulation_steps == accum
    assert batch * accum == 16
    assert config.vpr.classifier_cpu_offload == (dimension == 4096)


def test_150k_generator_config_is_fresh_output():
    config = load_config("configs/generator_qsattn_clean_night_150k.yaml")
    assert config.generator.max_steps == 150000
    assert config.generator.resume == ""
    assert not hasattr(config.generator, "vpr_relation_weight")
    assert config.data.generated_root.endswith("_150k")


@pytest.mark.parametrize(
    "path,family,dimension,batch,accum",
    [
        ("configs/vpr_cosplace_r50_512_safe_60k.yaml", "cosplace", 512, 8, 2),
        ("configs/vpr_eigenplaces_r50_512_safe_60k.yaml", "eigenplaces", 512, 8, 2),
        ("configs/vpr_eigenplaces_r50_2048_safe_60k.yaml", "eigenplaces", 2048, 8, 2),
        ("configs/vpr_mixvpr_r50_512_safe_60k.yaml", "mixvpr", 512, 8, 2),
        ("configs/vpr_mixvpr_r50_4096_safe_60k.yaml", "mixvpr", 4096, 4, 4),
    ],
)
def test_safe_official_configs_preserve_initialization(path, family, dimension, batch, accum):
    config = load_config(path)
    cfg = config.vpr
    assert cfg.model_family == family
    assert cfg.descriptor_dim == dimension
    assert cfg.samples_per_epoch == 60000
    assert cfg.dataset_epoch_quotas == {"msls": 20000, "robotcar": 20000, "svox": 20000}
    assert cfg.epochs == 20
    assert cfg.batch_size == batch
    assert cfg.gradient_accumulation_steps == accum
    assert batch * accum == 16
    assert cfg.cross_domain_aux == "preserve_day"
    assert cfg.preservation_teacher == "initialization"
    assert cfg.freeze_batch_norm
    assert cfg.evaluate_epoch_zero
    assert not cfg.early_stopping
    assert cfg.classifier_warmup_epochs == 1
    assert cfg.aggregator_lr == pytest.approx(1e-6)
    assert not cfg.classifier_cpu_offload


@pytest.mark.parametrize(
    "path,backbone",
    [
        ("configs/vpr_screen_efficientnet_b0_safe_60k.yaml", "efficientnet_b0"),
        ("configs/vpr_screen_regnet_y_800mf_safe_60k.yaml", "regnet_y_800mf"),
        ("configs/vpr_screen_regnet_y_1_6gf_safe_60k.yaml", "regnet_y_1_6gf"),
        ("configs/vpr_screen_convnext_tiny_safe_60k.yaml", "convnext_tiny"),
    ],
)
def test_safe_lightweight_configs_use_fixed_no_pair_budget(path, backbone):
    cfg = load_config(path).vpr
    assert cfg.backbone == backbone
    assert cfg.samples_per_epoch == 60000
    assert cfg.epochs == 20
    assert cfg.cross_domain_aux == "none"
    assert cfg.preservation_teacher == "none"
    assert cfg.freeze_batch_norm
    assert cfg.evaluate_epoch_zero
    assert not cfg.early_stopping
    assert cfg.scheduler == "cosine"


def test_nocplace_repro_override_and_fingerprint():
    config = load_config("configs/nocplace_repro_clean5k.yaml")
    updated = apply_overrides(config, ["nocplace_repro.workers=0"])
    assert updated.nocplace_repro.workers == 0
    assert updated.fingerprint() != config.fingerprint()
