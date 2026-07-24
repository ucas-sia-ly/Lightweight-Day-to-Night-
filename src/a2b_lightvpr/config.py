from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    robotcar_manifest: str
    robotcar_protocol: str
    generator_manifest: str = "data/manifests/generator_train.csv"
    vpr_source_manifest: str = "data/manifests/vpr_train.csv"
    generated_root: str = "data/generated/robotcar_train"
    tokyo_root: str = ""
    aachen_root: str = ""
    svox_root: str = ""
    msls_root: str = ""
    dataset_weights: dict[str, float] = field(default_factory=dict)
    crop_size: int = 320
    load_short_side: int = 352
    output_height: int = 320
    output_width: int = 480


@dataclass(frozen=True)
class GeneratorConfig:
    objective: str = "negcut"
    output_dir: str = "runs/generator/negcut_pilot"
    max_steps: int = 30_000
    batch_size: int = 4
    workers: int = 4
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    checkpoint_every: int = 5_000
    seed: int = 0
    amp: bool = True
    resume: str = ""
    content_weight: float = 1.0
    num_patches: int = 64
    feature_dim: int = 256
    paired_probability: float = 0.2
    paired_charbonnier_weight: float = 1.0


@dataclass(frozen=True)
class VPRConfig:
    output_dir: str = "runs/vpr/efficientnet_b0_gem_320x480"
    model_family: str = "custom"
    backbone: str = "efficientnet_b0"
    aggregator: str = "gem"
    descriptor_dim: int = 512
    image_height: int = 256
    image_width: int = 384
    loss_profile: str = "v2"
    epochs: int = 5
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    classifier_cpu_offload: bool = False
    workers: int = 4
    backbone_lr: float = 1e-5
    head_lr: float = 1e-4
    aggregator_lr: float = 1e-4
    classifier_lr: float = 1e-4
    classifier_warmup_lr: float = 1e-3
    classifier_warmup_epochs: int = 0
    freeze_batch_norm: bool = False
    evaluate_epoch_zero: bool = False
    preservation_teacher: str = "none"
    preservation_target: float = 0.95
    preservation_weight: float = 1.0
    early_stopping: bool = True
    epoch_one_max_recall_drop: float = 0.0
    weight_decay: float = 1e-4
    pair_weight: float = 0.5
    pair_target: float = 0.85
    relation_weight: float = 0.5
    seed: int = 0
    amp: bool = True
    patience: int = 4
    init_checkpoint: str = ""
    init_checkpoint_scope: str = "full"
    resume: str = ""
    min_class_images: int = 4
    min_class_images_by_dataset: dict[str, int] = field(default_factory=dict)
    dataset_sampling_weights: dict[str, float] = field(default_factory=dict)
    datasets: tuple[str, ...] = ("msls", "robotcar", "svox")
    samples_per_epoch: int = 120_000
    sampler: str = "weighted"
    dataset_epoch_quotas: dict[str, int] = field(default_factory=dict)
    train_domains: str = "legacy"
    cross_domain_aux: str = "legacy"
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


@dataclass(frozen=True)
class EvalConfig:
    positive_radius_m: float = 10.0
    positive_yaw_deg: float = 30.0
    recalls: tuple[int, ...] = (1, 5, 10)
    baseline_recall1: float = 0.8160535117056856
    bootstrap_samples: int = 10_000
    protocol: str = "project"
    datasets: tuple[str, ...] = ("robotcar",)


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = False
    project: str = "a2b-lightvpr"
    entity: str = ""
    group: str = ""
    name: str = ""
    tags: tuple[str, ...] = ()
    mode: str = "online"
    log_every_steps: int = 50
    image_every_steps: int = 1_000
    upload_checkpoints: bool = False


@dataclass(frozen=True)
class NocPlaceReproConfig:
    negcut_repo: str = "/home/admin123/diffusion_vpr_test/reference_code/NEGCUT"
    nocplace_repo: str = "/home/admin123/diffusion_vpr_test/NocPlace"
    cosplace_checkpoint: str = "/home/admin123/diffusion_vpr_test/reference_code/CosPlace/resnet50_512.pth"
    data_root: str = "data/nocplace_repro"
    negcut_name: str = "a2b_nocplace_5k"
    negcut_checkpoints_dir: str = "reference_code/NEGCUT/checkpoints"
    generated_root: str = "data/generated/nocplace_negcut_official_5k"
    output_dir: str = "runs/reference/nocplace_protocol_clean5k"
    samples_per_dataset_domain: int = 500
    seed: int = 0
    load_size: int = 640
    crop_size: int = 512
    negcut_epochs: int = 200
    negcut_decay_epochs: int = 200
    negcut_batch_size: int = 1
    negcut_lr: float = 2e-4
    translation_batch_size: int = 4
    vpr_batch_size: int = 32
    vpr_iterations: int = 10_000
    model_lr: float = 1e-6
    classifier_lr: float = 1e-2
    ikt_temperature: float = 4.0
    ikt_weight: float = 30.0
    workers: int = 4


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    vpr: VPRConfig = field(default_factory=VPRConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    nocplace_repro: NocPlaceReproConfig = field(default_factory=NocPlaceReproConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        values = self.as_dict()
        # A checkpoint path is runtime state, not part of experiment identity.
        values["generator"]["resume"] = ""
        values["vpr"]["resume"] = ""
        values.pop("tracking", None)
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
    if cls is TrackingConfig and "tags" in values:
        values["tags"] = tuple(str(item) for item in values["tags"])
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = ExperimentConfig(
        data=_construct(DataConfig, raw.get("data")),
        generator=_construct(GeneratorConfig, raw.get("generator")),
        vpr=_construct(VPRConfig, raw.get("vpr")),
        evaluation=_construct(EvalConfig, raw.get("evaluation")),
        tracking=_construct(TrackingConfig, raw.get("tracking")),
        nocplace_repro=_construct(NocPlaceReproConfig, raw.get("nocplace_repro")),
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
        "tracking": config.tracking,
        "nocplace_repro": config.nocplace_repro,
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
        if section == "tracking" and field_name == "tags":
            value = tuple(str(item) for item in value)
        sections[section] = replace(sections[section], **{field_name: value})
    updated = ExperimentConfig(**sections)
    validate_config(updated)
    return updated


def validate_config(config: ExperimentConfig) -> None:
    repro = config.nocplace_repro
    if repro.samples_per_dataset_domain != 500:
        raise ValueError("NocPlace clean5k requires 500 images per dataset and domain")
    if (repro.load_size, repro.crop_size) != (640, 512):
        raise ValueError("NocPlace NEG-CUT preprocessing must be 640 -> 512")
    if (repro.negcut_epochs, repro.negcut_decay_epochs) != (200, 200):
        raise ValueError("NocPlace NEG-CUT must use the official 200+200 epoch schedule")
    if repro.vpr_iterations != 10_000 or repro.vpr_batch_size != 32:
        raise ValueError("NocPlace reproduction requires 10,000 iterations and batch 32")
    if repro.translation_batch_size < 1:
        raise ValueError("NocPlace translation_batch_size must be positive")
    if repro.ikt_temperature != 4.0 or repro.ikt_weight != 30.0:
        raise ValueError("NocPlace reproduction requires T=4 and IKT weight=30")
    if config.generator.objective not in {"negcut", "qsattn"}:
        raise ValueError("generator.objective must be one of: negcut, qsattn")
    if config.vpr.model_family not in {"custom", "cosplace", "eigenplaces", "mixvpr"}:
        raise ValueError("vpr.model_family must be custom, cosplace, eigenplaces, or mixvpr")
    if config.vpr.backbone not in {
        "efficientnet_b0", "regnet_y_800mf", "regnet_y_1_6gf",
        "convnext_tiny", "resnet50",
    }:
        raise ValueError("Unsupported lightweight VPR backbone")
    if config.vpr.aggregator not in {"gem", "convap", "mixvpr"}:
        raise ValueError("vpr.aggregator must be gem, convap, or mixvpr")
    if config.vpr.init_checkpoint_scope not in {"full", "backbone"}:
        raise ValueError("vpr.init_checkpoint_scope must be full or backbone")
    if config.vpr.loss_profile not in {"v0", "v1", "v2", "v3"}:
        raise ValueError("vpr.loss_profile must be v0, v1, v2, or v3")
    if config.vpr.scheduler not in {"constant", "cosine", "plateau"}:
        raise ValueError("vpr.scheduler must be constant, cosine, or plateau")
    if config.vpr.relation_teacher not in {"none", "self", "salad", "boq"}:
        raise ValueError("vpr.relation_teacher must be none, self, salad, or boq")
    if config.vpr.loss_profile == "v3":
        if config.vpr.relation_teacher == "none":
            raise ValueError("V3 requires vpr.relation_teacher=self, salad, or boq")
    elif config.vpr.relation_teacher != "none":
        raise ValueError("Only V3 may configure a relation teacher")
    if config.vpr.relation_teacher in {"salad", "boq"}:
        if not config.vpr.relation_teacher_checkpoint:
            raise ValueError("Strong relation teachers require relation_teacher_checkpoint")
        if not config.vpr.relation_teacher_repo:
            raise ValueError("Strong relation teachers require a local relation_teacher_repo")
        if not config.vpr.relation_teacher_cache:
            raise ValueError("Strong relation teachers require relation_teacher_cache")
    if config.vpr.descriptor_dim not in {512, 2048, 4096}:
        raise ValueError("vpr.descriptor_dim must be 512, 2048, or 4096")
    if config.vpr.model_family in {"custom", "cosplace"} and config.vpr.descriptor_dim != 512:
        raise ValueError("Custom and CosPlace candidates use 512-dimensional descriptors")
    if config.vpr.model_family == "eigenplaces" and config.vpr.descriptor_dim not in {512, 2048}:
        raise ValueError("EigenPlaces adaptation supports descriptor_dim 512 or 2048")
    if config.vpr.model_family == "mixvpr" and config.vpr.descriptor_dim not in {512, 4096}:
        raise ValueError("MixVPR adaptation supports descriptor_dim 512 or 4096")
    if config.vpr.model_family in {"cosplace", "eigenplaces", "mixvpr"} and config.vpr.backbone != "resnet50":
        raise ValueError("Official VPR adaptation candidates require backbone=resnet50")
    if (config.vpr.image_height, config.vpr.image_width) not in {
        (256, 384),
        (320, 480),
        (320, 320),
    }:
        raise ValueError("VPR resolution must be 256x384, 320x480, or 320x320")
    if any(weight <= 0 for weight in config.data.dataset_weights.values()):
        raise ValueError("data.dataset_weights must contain only positive values")
    if not 0.0 <= config.generator.paired_probability <= 1.0:
        raise ValueError("generator.paired_probability must be between 0 and 1")
    if config.generator.paired_charbonnier_weight < 0:
        raise ValueError("Paired loss weights cannot be negative")
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
    if config.vpr.sampler not in {"weighted", "coverage_balanced", "rolling_balanced"}:
        raise ValueError("vpr.sampler must be weighted, coverage_balanced, or rolling_balanced")
    if config.vpr.sampler in {"coverage_balanced", "rolling_balanced"}:
        if not config.vpr.dataset_epoch_quotas:
            raise ValueError("coverage_balanced requires vpr.dataset_epoch_quotas")
        if sum(config.vpr.dataset_epoch_quotas.values()) != config.vpr.samples_per_epoch:
            raise ValueError("dataset_epoch_quotas must sum to samples_per_epoch")
        if any(value <= 0 for value in config.vpr.dataset_epoch_quotas.values()):
            raise ValueError("dataset_epoch_quotas must be positive")
    expected_datasets = {"msls", "robotcar", "svox"}
    if set(config.vpr.datasets) != expected_datasets:
        raise ValueError("Active VPR training datasets must be exactly msls, robotcar, and svox")
    if config.vpr.dataset_epoch_quotas and set(config.vpr.dataset_epoch_quotas) != expected_datasets:
        raise ValueError("VPR epoch quotas must name exactly msls, robotcar, and svox")
    if config.vpr.model_family == "mixvpr" and (
        config.vpr.image_height, config.vpr.image_width
    ) != (320, 320):
        raise ValueError("Official MixVPR adaptation requires 320x320 input")
    if config.vpr.train_domains not in {"legacy", "day", "fake", "day_fake"}:
        raise ValueError("vpr.train_domains must be legacy, day, fake, or day_fake")
    if config.vpr.cross_domain_aux not in {
        "legacy", "none", "pair", "ikt", "relation", "preserve_day"
    }:
        raise ValueError("Unsupported vpr.cross_domain_aux")
    if config.vpr.cross_domain_aux == "pair" and config.vpr.train_domains != "day_fake":
        raise ValueError("Pair auxiliary requires vpr.train_domains=day_fake")
    if config.vpr.cross_domain_aux in {"ikt", "relation"} and config.vpr.train_domains not in {"fake", "day_fake"}:
        raise ValueError("IKT/relation auxiliary requires fake training images")
    if config.vpr.cross_domain_aux == "preserve_day":
        if config.vpr.train_domains != "day_fake":
            raise ValueError("Day preservation requires vpr.train_domains=day_fake")
        if config.vpr.preservation_teacher != "initialization":
            raise ValueError("Day preservation requires preservation_teacher=initialization")
    elif config.vpr.preservation_teacher != "none":
        raise ValueError("preservation_teacher is only valid with cross_domain_aux=preserve_day")
    if config.vpr.database_encoder not in {"adapted", "original"}:
        raise ValueError("vpr.database_encoder must be adapted or original")
    if not 0.0 < config.vpr.scheduler_factor < 1.0:
        raise ValueError("vpr.scheduler_factor must be between zero and one")
    if config.vpr.scheduler_patience < 0 or config.vpr.scheduler_threshold < 0:
        raise ValueError("VPR scheduler patience and threshold cannot be negative")
    if not 0.0 < config.vpr.scheduler_min_lr_ratio <= 1.0:
        raise ValueError("vpr.scheduler_min_lr_ratio must be in (0, 1]")
    if min(
        config.vpr.backbone_lr,
        config.vpr.aggregator_lr,
        config.vpr.classifier_lr,
        config.vpr.classifier_warmup_lr,
    ) <= 0:
        raise ValueError("VPR optimizer learning rates must be positive")
    if not 0 <= config.vpr.classifier_warmup_epochs < config.vpr.epochs:
        raise ValueError("classifier_warmup_epochs must be in [0, epochs)")
    if not 0.0 < config.vpr.preservation_target <= 1.0:
        raise ValueError("preservation_target must be in (0, 1]")
    if config.vpr.preservation_weight < 0 or config.vpr.epoch_one_max_recall_drop < 0:
        raise ValueError("Preservation weight and epoch-one drop threshold cannot be negative")
    if config.evaluation.protocol not in {"project", "paper"}:
        raise ValueError("evaluation.protocol must be project or paper")
    if config.tracking.mode not in {"online", "offline"}:
        raise ValueError("tracking.mode must be online or offline")
    if config.tracking.log_every_steps < 1 or config.tracking.image_every_steps < 1:
        raise ValueError("W&B logging intervals must be positive")
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
