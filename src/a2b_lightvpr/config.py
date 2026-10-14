from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


TENSORBOARD_FIELDS = (
    "tensorboard_enabled", "tensorboard_log_dir", "tensorboard_log_interval",
    "tensorboard_image_interval", "tensorboard_flush_secs",
)


@dataclass(frozen=True)
class DataConfig:
    robotcar_seasons_root: str = "/home/admin123/github/datasets/RobotCar-Seasons"
    generator_manifest: str = "data/manifests/generator_train.csv"
    vpr_source_manifest: str = "data/manifests/vpr_train.csv"
    generated_root: str = "data/generated/robotcar_train"
    tokyo_root: str = ""
    svox_root: str = ""
    msls_root: str = ""
    dataset_weights: dict[str, float] = field(default_factory=dict)
    crop_size: int = 320
    load_short_side: int = 352
    output_height: int = 320
    output_width: int = 480


@dataclass(frozen=True)
class GeneratorConfig:
    objective: str = "qsattn"
    output_dir: str = "runs/generator/qsattn_official"
    max_steps: int = 30_000
    batch_size: int = 4
    workers: int = 4
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    checkpoint_every: int = 5_000
    amp: bool = True
    resume: str = ""
    content_weight: float = 1.0
    num_patches: int = 64
    feature_dim: int = 256
    qs_mode: str = "global"
    qs_nce_identity: bool = False
    qs_netf: str = "mlp_sample"
    tensorboard_enabled: bool = True
    tensorboard_log_dir: str = ""
    tensorboard_log_interval: int = 10
    tensorboard_image_interval: int = 100
    tensorboard_flush_secs: int = 30


@dataclass(frozen=True)
class VPRConfig:
    output_dir: str = "runs/vpr/resnet50_gem_320x480"
    model_family: str = "custom"
    aggregator: str = "gem"
    descriptor_dim: int = 512
    num_clusters: int = 64
    image_height: int = 256
    image_width: int = 384
    loss_profile: str = "v2"
    random_seed: int = 0
    epochs: int = 5
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    classifier_cpu_offload: bool = False
    workers: int = 4
    encoder_lr: float = 1e-5
    aggregator_lr: float = 1e-4
    classifier_lr: float = 1e-4
    classifier_warmup_lr: float = 1e-3
    classifier_warmup_epochs: int = 0
    freeze_batch_norm: bool = False
    selection_rule: str = "validation_best"
    evaluate_epoch_zero: bool = False
    early_stopping: bool = True
    epoch_one_max_recall_drop: float = 0.0
    weight_decay: float = 1e-4
    relation_weight: float = 0.5
    amp: bool = True
    patience: int = 4
    init_checkpoint: str = ""
    backbone_name: str = ""
    backbone_repo: str = ""
    backbone_checkpoint: str = ""
    backbone_trainable_blocks: int = 0
    aggregator_repo: str = ""
    resume: str = ""
    min_class_images: int = 4
    min_class_images_by_dataset: dict[str, int] = field(default_factory=dict)
    dataset_sampling_weights: dict[str, float] = field(default_factory=dict)
    datasets: tuple[str, ...] = ("msls", "robotcar", "svox")
    samples_per_epoch: int = 120_000
    sampler: str = "weighted"
    dataset_epoch_quotas: dict[str, int] = field(default_factory=dict)
    train_domains: str = "day_fake"
    cross_domain_aux: str = "none"
    database_encoder: str = "adapted"
    scheduler: str = "constant"
    scheduler_factor: float = 0.2
    scheduler_patience: int = 0
    scheduler_threshold: float = 0.001
    scheduler_min_lr_ratio: float = 0.01
    relation_teacher: str = "none"
    relation_teacher_checkpoint: str = ""
    relation_teacher_repo: str = ""
    relation_teacher_cache: str = ""
    relation_teacher_input_size: int = 322
    relation_queue_size: int = 2048
    relation_temperature: float = 0.07
    relation_day_weight: float = 0.5
    relation_mask_self: bool = True
    relation_queue_scope: str = "dataset"
    pk_places_per_batch: int = 4
    pk_images_per_place: int = 2
    stable_place_weight: float = 1.0
    stable_place_temperature: float = 0.07
    stable_stability_weight: float = 0.0
    stable_reliability_weight: float = 0.01
    stable_reliability_target: str = "none"
    stable_reliability_floor: float = 0.25
    stable_dustbin_enabled: bool = True
    stable_dustbin_gamma_init: float = 1.0
    stable_dustbin_gamma_max: float = 4.0
    stable_sinkhorn_iterations: int = 20
    stable_feature_fusion: str = "none"
    tensorboard_enabled: bool = True
    tensorboard_log_dir: str = ""
    tensorboard_log_interval: int = 10
    tensorboard_image_interval: int = 100
    tensorboard_flush_secs: int = 30


@dataclass(frozen=True)
class EvalConfig:
    positive_radius_m: float = 25.0
    positive_yaw_deg: float | None = None
    recalls: tuple[int, ...] = (1, 5, 10)
    protocol: str = "standard"
    datasets: tuple[str, ...] = ("msls_day_val",)
    database_domain: str = "day"
    fake_database_manifest: str = ""


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    vpr: VPRConfig = field(default_factory=VPRConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        values = self.as_dict()
        # A checkpoint path is runtime state, not part of experiment identity.
        values["generator"]["resume"] = ""
        values["vpr"]["resume"] = ""
        for section in ("generator", "vpr"):
            for field_name in TENSORBOARD_FIELDS:
                values[section].pop(field_name, None)
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _construct(cls, values: dict[str, Any] | None):
    values = dict(values or {})
    if cls is EvalConfig and "recalls" in values:
        values["recalls"] = tuple(int(item) for item in values["recalls"])
    if cls is EvalConfig and "datasets" in values:
        values["datasets"] = tuple(str(item) for item in values["datasets"])
    if cls is VPRConfig and "datasets" in values:
        values["datasets"] = tuple(str(item) for item in values["datasets"])
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = ExperimentConfig(
        data=_construct(DataConfig, raw.get("data")),
        generator=_construct(GeneratorConfig, raw.get("generator")),
        vpr=_construct(VPRConfig, raw.get("vpr")),
        evaluation=_construct(EvalConfig, raw.get("evaluation")),
    )
    validate_config(config)
    return config


def apply_overrides(config: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    """Apply explicit `section.field=value` CLI overrides through YAML scalar parsing."""
    sections = {
        "data": config.data,
        "generator": config.generator,
        "vpr": config.vpr,
        "evaluation": config.evaluation,
    }
    for expression in overrides:
        if "=" not in expression or "." not in expression.split("=", 1)[0]:
            raise ValueError(f"Invalid override {expression!r}; expected section.field=value")
        key, raw_value = expression.split("=", 1)
        section, field_name = key.split(".", 1)
        if section not in sections or not hasattr(sections[section], field_name):
            raise ValueError(f"Unknown config override: {key}")
        value = yaml.safe_load(raw_value)
        if section == "evaluation" and field_name == "recalls":
            value = tuple(int(item) for item in value)
        if section == "evaluation" and field_name == "datasets":
            value = tuple(str(item) for item in value)
        if section == "vpr" and field_name == "datasets":
            value = tuple(str(item) for item in value)
        sections[section] = replace(sections[section], **{field_name: value})
    updated = ExperimentConfig(**sections)
    validate_config(updated)
    return updated


def validate_config(config: ExperimentConfig) -> None:
    for name, cfg in (("generator", config.generator), ("vpr", config.vpr)):
        if not isinstance(cfg.tensorboard_enabled, bool):
            raise ValueError(f"{name}.tensorboard_enabled must be a boolean")
        if not isinstance(cfg.tensorboard_log_dir, str):
            raise ValueError(f"{name}.tensorboard_log_dir must be a string")
        for field_name in ("tensorboard_log_interval", "tensorboard_flush_secs"):
            value = getattr(cfg, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name}.{field_name} must be a positive integer")
        value = cfg.tensorboard_image_interval
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name}.tensorboard_image_interval must be a non-negative integer")
    if config.generator.objective != "qsattn":
        raise ValueError("The only supported generator objective is qsattn")
    if config.generator.qs_mode != "global":
        raise ValueError("The active official QS-Attn objective requires qs_mode=global")
    if config.generator.qs_netf != "mlp_sample":
        raise ValueError("The active official QS-Attn objective requires qs_netf=mlp_sample")
    if config.vpr.model_family not in {
        "custom", "cosplace", "eigenplaces", "mixvpr", "netvlad", "salad", "boq", "dino_mix", "stable_night_salad"
    }:
        raise ValueError(
            "vpr.model_family must be custom, cosplace, eigenplaces, mixvpr, "
            "netvlad, salad, or boq"
        )
    if config.vpr.aggregator not in {"gem", "convap", "mixvpr", "netvlad", "salad", "boq"}:
        raise ValueError("vpr.aggregator must be gem, convap, mixvpr, netvlad, salad, or boq")
    if config.vpr.loss_profile != "v2":
        raise ValueError("The single-run VPR workflow requires loss_profile=v2")
    if config.vpr.scheduler not in {"constant", "cosine", "plateau"}:
        raise ValueError("vpr.scheduler must be constant, cosine, or plateau")
    if config.vpr.relation_teacher not in {"none", "salad", "boq"}:
        raise ValueError("vpr.relation_teacher must be none, salad, or boq")
    cross_domain_relation = config.vpr.cross_domain_aux == "cross_domain_relation"
    if not cross_domain_relation and config.vpr.relation_teacher != "none":
        raise ValueError("Only cross_domain_relation may configure a relation teacher")
    if cross_domain_relation and config.vpr.relation_teacher not in {"salad", "boq"}:
        raise ValueError("cross_domain_relation requires an offline SALAD or BoQ teacher")
    if config.vpr.relation_teacher in {"salad", "boq"}:
        if not config.vpr.relation_teacher_checkpoint:
            raise ValueError("Strong relation teachers require relation_teacher_checkpoint")
        if not config.vpr.relation_teacher_repo:
            raise ValueError("Strong relation teachers require a local relation_teacher_repo")
        if not config.vpr.relation_teacher_cache:
            raise ValueError("Strong relation teachers require relation_teacher_cache")
    if config.vpr.descriptor_dim not in {128, 512, 544, 2048, 4096, 12288}:
        raise ValueError("vpr.descriptor_dim must be 128, 512, 544, 2048, 4096, or 12288")
    if config.vpr.model_family == "stable_night_salad":
        cfg = config.vpr
        for name in ("backbone_trainable_blocks", "pk_places_per_batch", "pk_images_per_place"):
            if isinstance(getattr(cfg, name), bool) or not isinstance(getattr(cfg, name), int):
                raise ValueError(f"StableNight {name} must be an integer")
        if cfg.aggregator != "salad" or cfg.descriptor_dim != 544 or (cfg.image_height, cfg.image_width) != (322, 322):
            raise ValueError("StableNight-SALAD requires SALAD, 544D, and 322x322 input")
        if cfg.backbone_name not in {"dinov2_vits14", "dinov2_vits14_reg"}:
            raise ValueError("StableNight-SALAD requires DINOv2-S/14")
        if not cfg.backbone_repo or not cfg.backbone_checkpoint or not 0 <= cfg.backbone_trainable_blocks <= 12:
            raise ValueError("StableNight-SALAD requires local backbone weights and 0--12 trainable blocks")
        if cfg.cross_domain_aux != "none" or cfg.classifier_warmup_epochs != 0:
            raise ValueError("Stage-one StableNight-SALAD requires no CDND or classifier warmup")
        if cfg.sampler != "pk_balanced" or cfg.pk_places_per_batch < 2 or cfg.pk_images_per_place < 2:
            raise ValueError("StableNight place contrastive training requires PK with P>=2 and K>=2")
        if cfg.train_domains not in {"day_only", "day_aug", "day_fake", "day_aug_fake"}:
            raise ValueError("StableNight train_domains must be day_only, day_aug, day_fake, or day_aug_fake")
        for name in ("stable_place_weight", "stable_stability_weight", "stable_reliability_weight"):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if cfg.stable_place_weight == 0 and cfg.stable_stability_weight == 0 and cfg.stable_reliability_weight == 0:
            raise ValueError("At least one StableNight loss must be enabled")
        if not math.isfinite(cfg.stable_place_temperature) or cfg.stable_place_temperature <= 0:
            raise ValueError("StableNight place temperature must be finite and positive")
        if not 0 <= cfg.stable_reliability_floor <= 1:
            raise ValueError("StableNight reliability floor must be in [0,1]")
        if cfg.stable_reliability_target not in {"none", "augmentation"}:
            raise ValueError("StableNight reliability target must be none or augmentation")
        needs_local = cfg.stable_stability_weight > 0 or (
            cfg.stable_reliability_weight > 0 and cfg.stable_reliability_target == "augmentation"
        )
        if needs_local and cfg.train_domains not in {"day_aug", "day_aug_fake"}:
            raise ValueError("StableNight local losses require geometrically aligned Day/Aug views")
        if cfg.stable_stability_weight > 0 and cfg.backbone_trainable_blocks == 0:
            raise ValueError("StableNight local feature stability requires a trainable backbone block")
        if not isinstance(cfg.stable_dustbin_enabled, bool):
            raise ValueError("stable_dustbin_enabled must be a boolean")
        if not math.isfinite(cfg.stable_dustbin_gamma_max) or not 0 < cfg.stable_dustbin_gamma_init < cfg.stable_dustbin_gamma_max:
            raise ValueError("StableNight dustbin gamma requires 0 < init < finite max")
        if isinstance(cfg.stable_sinkhorn_iterations, bool) or not isinstance(cfg.stable_sinkhorn_iterations, int) or cfg.stable_sinkhorn_iterations < 1:
            raise ValueError("StableNight Sinkhorn iterations must be a positive integer")
        if cfg.stable_feature_fusion != "none":
            raise ValueError("StableNight multi-layer fusion is deferred to stage two; use none")
    if config.vpr.model_family == "dino_mix":
        if config.vpr.backbone_name != "dinov2_vits14" or config.vpr.aggregator != "mixvpr" or config.vpr.descriptor_dim != 512:
            raise ValueError("DINO-Mix requires plain DINOv2-S, MixVPR and 512D")
        if (config.vpr.image_height, config.vpr.image_width) != (322, 322):
            raise ValueError("DINO-Mix input must be 322x322")
        if not config.vpr.backbone_repo or not config.vpr.backbone_checkpoint or not 1 <= config.vpr.backbone_trainable_blocks <= 12:
            raise ValueError("DINO-Mix needs local backbone and 1-12 trainable blocks")
    if config.vpr.model_family == "boq":
        if config.vpr.aggregator != "boq" or config.vpr.descriptor_dim != 12288:
            raise ValueError("BoQ CDND requires boq aggregation and 12,288 dimensions")
        if config.vpr.train_domains != "day_fake":
            raise ValueError("BoQ CDND requires paired day+fake training")
        if not config.vpr.backbone_name or not config.vpr.backbone_repo or not config.vpr.backbone_checkpoint:
            raise ValueError("BoQ CDND requires a DINOv2-S backbone configuration")
        if not config.vpr.aggregator_repo:
            raise ValueError("BoQ CDND requires the local BoQ aggregator repository")
        if config.vpr.cross_domain_aux == "cross_domain_relation" and config.vpr.relation_teacher != "boq":
            raise ValueError("BoQ CDND requires a BoQ teacher")
    if config.vpr.model_family in {"custom", "cosplace"} and config.vpr.descriptor_dim != 512:
        raise ValueError("Custom and CosPlace candidates use 512-dimensional descriptors")
    if config.vpr.model_family == "eigenplaces" and config.vpr.descriptor_dim not in {512, 2048}:
        raise ValueError("EigenPlaces adaptation supports descriptor_dim 512 or 2048")
    if config.vpr.model_family == "mixvpr" and config.vpr.descriptor_dim not in {128, 512, 4096}:
        raise ValueError("MixVPR adaptation supports descriptor_dim 128, 512, or 4096")
    if config.vpr.model_family == "salad":
        if config.vpr.aggregator != "salad" or config.vpr.descriptor_dim != 544:
            raise ValueError("Compact SALAD requires salad aggregation and 544 dimensions")
        if config.vpr.backbone_name != "dinov2_vits14_reg":
            raise ValueError("Compact SALAD student requires dinov2_vits14_reg")
        if not config.vpr.backbone_repo or not config.vpr.backbone_checkpoint:
            raise ValueError("Compact SALAD requires backbone_repo and backbone_checkpoint")
        if not 0 < config.vpr.backbone_trainable_blocks <= 12:
            raise ValueError("Compact SALAD requires 1--12 trainable backbone blocks")
    if config.vpr.model_family == "netvlad":
        if config.vpr.aggregator != "netvlad" or config.vpr.descriptor_dim != 4096:
            raise ValueError("NetVLAD adaptation requires netvlad aggregation and 4096 dimensions")
        if config.vpr.num_clusters != 64:
            raise ValueError("The official NetVLAD checkpoint requires 64 clusters")
    if (config.vpr.image_height, config.vpr.image_width) not in {
        (256, 384),
        (320, 480),
        (320, 320),
        (322, 322),
        (322, 322),
    }:
        raise ValueError("VPR resolution must be 256x384, 320x480, or 320x320")
    if any(weight <= 0 for weight in config.data.dataset_weights.values()):
        raise ValueError("data.dataset_weights must contain only positive values")
    if config.vpr.min_class_images < 1:
        raise ValueError("vpr.min_class_images must be at least one")
    if any(value < 2 for value in config.vpr.min_class_images_by_dataset.values()):
        raise ValueError("Per-dataset minimum class images must be at least two")
    if any(value <= 0 for value in config.vpr.dataset_sampling_weights.values()):
        raise ValueError("VPR dataset sampling weights must be positive")
    if config.vpr.samples_per_epoch < config.vpr.batch_size:
        raise ValueError("vpr.samples_per_epoch must be at least one batch")
    if config.vpr.gradient_accumulation_steps < 1:
        raise ValueError("vpr.gradient_accumulation_steps must be positive")
    if config.vpr.sampler not in {
        "weighted", "coverage_balanced", "rolling_balanced", "pk_balanced"
    }:
        raise ValueError(
            "vpr.sampler must be weighted, coverage_balanced, rolling_balanced, or pk_balanced"
        )
    if config.vpr.sampler in {"coverage_balanced", "rolling_balanced", "pk_balanced"}:
        if not config.vpr.dataset_epoch_quotas:
            raise ValueError("Balanced samplers require vpr.dataset_epoch_quotas")
        if sum(config.vpr.dataset_epoch_quotas.values()) != config.vpr.samples_per_epoch:
            raise ValueError("dataset_epoch_quotas must sum to samples_per_epoch")
        if any(value <= 0 for value in config.vpr.dataset_epoch_quotas.values()):
            raise ValueError("dataset_epoch_quotas must be positive")
    if config.vpr.sampler == "pk_balanced":
        pk_batch = config.vpr.pk_places_per_batch * config.vpr.pk_images_per_place
        if min(config.vpr.pk_places_per_batch, config.vpr.pk_images_per_place) < 1:
            raise ValueError("PK sampler P and K must be positive")
        if config.vpr.batch_size != pk_batch:
            raise ValueError("PK sampler requires batch_size == pk_places_per_batch * pk_images_per_place")
        if any(value % pk_batch for value in config.vpr.dataset_epoch_quotas.values()):
            raise ValueError("Every PK dataset quota must be divisible by the physical batch size")
    if config.vpr.selection_rule not in {"validation_best", "fixed_budget_last"}:
        raise ValueError("Unknown VPR selection_rule")
    if config.vpr.selection_rule == "fixed_budget_last" and (
        config.vpr.evaluate_epoch_zero or config.vpr.early_stopping
        or config.vpr.scheduler == "plateau" or config.vpr.epoch_one_max_recall_drop > 0
    ):
        raise ValueError("Fixed-budget selection must disable validation-based controls")
    expected_datasets = {"msls", "robotcar", "svox"}
    if (
        config.vpr.model_family == "stable_night_salad"
        and set(config.vpr.datasets)
        and set(config.vpr.datasets).issubset(expected_datasets)
    ):
        expected_datasets = set(config.vpr.datasets)
    if set(config.vpr.datasets) in ({"gsv_cities"}, {"robotcar"}, {"svox"}, {"msls"}):
        expected_datasets = set(config.vpr.datasets)
    if set(config.vpr.datasets) != expected_datasets:
        raise ValueError(
            "Formal VPR training must use msls/robotcar/svox, gsv_cities alone, or robotcar alone"
        )
    if config.vpr.dataset_epoch_quotas and set(config.vpr.dataset_epoch_quotas) != expected_datasets:
        raise ValueError("VPR epoch quotas must match the active training datasets")
    if config.vpr.model_family == "mixvpr" and (
        config.vpr.image_height, config.vpr.image_width
    ) != (320, 320):
        raise ValueError("Official MixVPR adaptation requires 320x320 input")
    if config.vpr.model_family == "salad":
        if (config.vpr.image_height, config.vpr.image_width) != (322, 322):
            raise ValueError("Compact SALAD requires 322x322 input")
        if config.vpr.backbone_name != "dinov2_vits14_reg":
            raise ValueError("Compact SALAD requires dinov2_vits14_reg")
        if not config.vpr.backbone_repo or not config.vpr.backbone_checkpoint:
            raise ValueError("Compact SALAD requires a backbone repo and checkpoint")
        if not 0 <= config.vpr.backbone_trainable_blocks <= 12:
            raise ValueError("backbone_trainable_blocks must be between 0 and 12")
    if config.vpr.model_family == "salad" and (
        config.vpr.image_height, config.vpr.image_width
    ) != (322, 322):
        raise ValueError("Compact SALAD requires 322x322 input")
    if config.vpr.model_family != "stable_night_salad" and config.vpr.train_domains not in {"day", "day_fake"}:
        raise ValueError("vpr.train_domains must be day or day_fake")
    if config.vpr.cross_domain_aux not in {"none", "cross_domain_relation"}:
        raise ValueError("cross_domain_aux must be none or cross_domain_relation")
    if config.vpr.cross_domain_aux == "cross_domain_relation":
        if config.vpr.train_domains != "day_fake":
            raise ValueError("cross_domain_relation requires vpr.train_domains=day_fake")
        if config.vpr.sampler != "pk_balanced":
            raise ValueError("cross_domain_relation requires vpr.sampler=pk_balanced")
    if config.vpr.relation_queue_scope not in {"dataset", "shared"}:
        raise ValueError("relation_queue_scope must be dataset or shared")
    if not isinstance(config.vpr.relation_mask_self, bool):
        raise ValueError("relation_mask_self must be a boolean")
    if (
        isinstance(config.vpr.random_seed, bool)
        or not isinstance(config.vpr.random_seed, int)
        or config.vpr.random_seed < 0
    ):
        raise ValueError("vpr.random_seed must be a non-negative integer")
    if config.vpr.relation_queue_size < 0:
        raise ValueError("relation_queue_size cannot be negative")
    if config.vpr.relation_temperature <= 0:
        raise ValueError("Relation temperature must be positive")
    if config.vpr.relation_weight < 0 or config.vpr.relation_day_weight < 0:
        raise ValueError("Relation loss weights cannot be negative")
    if config.vpr.database_encoder not in {"adapted", "original"}:
        raise ValueError("vpr.database_encoder must be adapted or original")
    if not 0.0 < config.vpr.scheduler_factor < 1.0:
        raise ValueError("vpr.scheduler_factor must be between zero and one")
    if config.vpr.scheduler_patience < 0 or config.vpr.scheduler_threshold < 0:
        raise ValueError("VPR scheduler patience and threshold cannot be negative")
    if not 0.0 < config.vpr.scheduler_min_lr_ratio <= 1.0:
        raise ValueError("vpr.scheduler_min_lr_ratio must be in (0, 1]")
    if min(
        config.vpr.encoder_lr,
        config.vpr.aggregator_lr,
        config.vpr.classifier_lr,
        config.vpr.classifier_warmup_lr,
    ) <= 0:
        raise ValueError("VPR optimizer learning rates must be positive")
    if not 0 <= config.vpr.classifier_warmup_epochs < config.vpr.epochs:
        raise ValueError("classifier_warmup_epochs must be in [0, epochs)")
    if config.vpr.epoch_one_max_recall_drop < 0:
        raise ValueError("Epoch-one drop threshold cannot be negative")
    if config.evaluation.protocol != "standard":
        raise ValueError("evaluation.protocol must be standard; the custom RobotCar protocols are retired")
    if not config.evaluation.datasets or len(set(config.evaluation.datasets)) != len(config.evaluation.datasets):
        raise ValueError("evaluation.datasets must contain unique validation datasets")
    if set(config.evaluation.datasets) - {"msls_day_val", "svox_day_val"}:
        raise ValueError("Training validation requires public msls_day_val or svox_day_val; test datasets cannot select checkpoints")
    if config.vpr.model_family == "stable_night_salad":
        if config.evaluation.database_domain not in {"day", "fake_night"}:
            raise ValueError("StableNight database_domain must be day or fake_night")
        if config.evaluation.database_domain == "fake_night" and not config.evaluation.fake_database_manifest:
            raise ValueError("Fake Night Database requires an explicit fake_database_manifest")
    elif config.evaluation.database_domain != "day":
        raise ValueError("The paper workflow evaluates against a daytime database")
    for value in (
        config.data.crop_size,
        config.data.output_height,
        config.data.output_width,
        config.generator.batch_size,
        config.vpr.image_height,
        config.vpr.image_width,
        config.vpr.relation_teacher_input_size,
    ):
        if int(value) <= 0:
            raise ValueError("Image sizes and batch sizes must be positive")
    if config.data.output_height % 4 or config.data.output_width % 4:
        raise ValueError("Generator output dimensions must be divisible by four")
