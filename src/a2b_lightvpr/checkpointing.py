from __future__ import annotations

import json
import hashlib
import os
import random
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .paths import resolve_resource_path


SCHEMA_VERSION = 1


def checkpoint_config_matches(state: dict, config, sections: tuple[str, ...] | None = None) -> bool:
    """Compare active training settings while tolerating retired metadata fields."""
    saved = deepcopy(state.get("config", {}))
    current = deepcopy(config.as_dict())
    if (sections is None or "vpr" in sections or "evaluation" in sections) and (
        {"robotcar_manifest", "robotcar_protocol"} & set(saved.get("data", {}))
        or saved.get("evaluation", {}).get("protocol", "standard") != "standard"
    ):
        # Old validation-selected optimizer states are different experiments.
        # They can still be loaded as inference weights, but not resumed under
        # the replacement public validation protocol.
        return False
    if state.get("config_fingerprint") == config.fingerprint():
        return True
    if not saved:
        return False
    if sections is None or "generator" not in sections:
        # The GSV metric route reads only source images. A retired generator
        # manifest is metadata, so deleting its unused path must not prevent
        # exporting or resuming otherwise identical metric checkpoints.
        source_metric = all(
            values.get("vpr", {}).get("training_objective") == "multi_similarity"
            and values.get("vpr", {}).get("train_domains") in {"day_only", "day_aug"}
            for values in (saved, current)
        )
        if source_metric:
            for values in (saved, current):
                values.get("data", {}).pop("generator_manifest", None)
    if sections is not None and "generator" in sections and "vpr" not in sections:
        # Only these data controls affect unpaired generator training. Public
        # evaluation roots, VPR labels and inference sizes do not change it.
        generator_data_fields = ("generator_manifest", "dataset_weights", "crop_size", "load_short_side")
        for values in (saved, current):
            values["data"] = {key: values.get("data", {})[key]
                              for key in generator_data_fields if key in values.get("data", {})}
    from .config import TENSORBOARD_FIELDS
    for values in (saved, current):
        values.get("generator", {}).update(resume="", output_dir="")
        values.get("vpr", {}).update(resume="", output_dir="")
        for section in ("generator", "vpr"):
            for name in TENSORBOARD_FIELDS:
                values.get(section, {}).pop(name, None)
    saved_vpr = saved.get("vpr", {})
    saved_vpr.setdefault("random_seed", int(saved_vpr.get("seed", 0)))
    saved_vpr.setdefault("relation_mask_self", True)
    saved_vpr.setdefault("relation_queue_scope", "dataset")
    saved_vpr.setdefault("backbone_name", "")
    saved_vpr.setdefault("backbone_repo", "")
    saved_vpr.setdefault("backbone_checkpoint", "")
    saved_vpr.setdefault("backbone_trainable_blocks", 0)
    # New prototype controls have no effect on legacy model families. Migrate
    # only their defaults so old baseline training states remain resumable.
    from .config import VPRConfig
    prototype_defaults = VPRConfig()
    for name, value in vars(prototype_defaults).items():
        if name.startswith("stable_") or name in {
            "training_objective", "input_transform", "salad_dropout",
            "metric_augmentation_probability", "metric_within_city",
            "require_cuda",
        }:
            saved_vpr.setdefault(name, value)
    saved.setdefault("evaluation", {}).setdefault("fake_database_manifest", "")
    saved["evaluation"].setdefault("batch_size", 0)
    def active_fields(candidate, reference):
        if isinstance(reference, dict):
            return {
                key: active_fields(candidate[key], value)
                for key, value in reference.items()
                if key in candidate
            }
        if isinstance(candidate, str) and isinstance(reference, str) and candidate != reference:
            if candidate.startswith("/") or reference.startswith("/"):
                resolved = resolve_resource_path(candidate)
                if resolved.exists() and resolved.resolve() == resolve_resource_path(reference).resolve():
                    return reference
        return candidate

    if sections is not None:
        saved = {name: saved.get(name, {}) for name in sections}
        current = {name: current[name] for name in sections}
    return active_fields(saved, current) == current


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve_resource_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_resource_paths(config) -> dict[str, Path]:
    """Exact public validation metadata consumed by checkpoint selection."""
    paths = {}
    for dataset in config.evaluation.datasets:
        if dataset == "msls_day_val":
            root = resolve_resource_path(config.data.msls_root)
            for city in ("cph", "sf"):
                for role in ("database", "query"):
                    paths[f"validation_msls_{city}_{role}"] = root / "train_val" / city / role / "postprocessed.csv"
        elif dataset == "svox_day_val":
            root = resolve_resource_path(config.data.svox_root)
            for role in ("gallery", "queries"):
                paths[f"validation_svox_{role}"] = root / "images" / "val" / role
        else:
            raise ValueError(f"Unsupported public training validation dataset: {dataset}")
    return paths


def data_fingerprints(config, *, generated: str | Path | None = None, purpose: str = "vpr") -> dict[str, str]:
    if purpose == "generator":
        return {"generator_manifest": file_sha256(config.data.generator_manifest)}
    if purpose != "vpr":
        raise ValueError("Fingerprint purpose must be vpr or generator")
    paths = {"vpr_source_manifest": config.data.vpr_source_manifest}
    needs_validation = config.vpr.evaluate_epoch_zero or config.vpr.selection_rule == "validation_best"
    if needs_validation:
        paths.update(validation_resource_paths(config))
    if generated is not None:
        paths["generated_manifest"] = generated
    fingerprints = {}
    for name, path in paths.items():
        path = resolve_resource_path(path)
        if path.is_dir():
            # SVOX encodes its poses in image names rather than separate tables.
            names = sorted(item.name for item in path.iterdir() if item.is_file())
            if not names:
                raise ValueError(f"Empty public validation image folder: {path}")
            fingerprints[name] = hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest()
        else:
            fingerprints[name] = file_sha256(path)
    return fingerprints


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
