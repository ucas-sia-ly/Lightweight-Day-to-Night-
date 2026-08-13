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


SCHEMA_VERSION = 1


def checkpoint_config_matches(state: dict, config, sections: tuple[str, ...] | None = None) -> bool:
    """Compare active training settings while tolerating retired metadata fields."""
    if state.get("config_fingerprint") == config.fingerprint():
        return True
    saved = deepcopy(state.get("config", {}))
    current = deepcopy(config.as_dict())
    if not saved:
        return False
    for values in (saved, current):
        values.get("generator", {}).update(resume="", output_dir="")
        values.get("vpr", {}).update(resume="", output_dir="")
    # Older formal checkpoints predate the opt-in RA-GTR fine-tuning fields.
    # Their absence is equivalent to the inactive defaults, not a changed experiment.
    saved_vpr = saved.get("vpr", {})
    current_vpr = current.get("vpr", {})
    inactive_defaults = {
        "finetune_checkpoint": "",
        "route_contrastive_weight": 0.0,
        "route_mining_cache": "",
        "route_positive_radius_m": 10.0,
        "route_positive_yaw_deg": 30.0,
        "route_negative_radius_m": 25.0,
        "route_contrastive_temperature": 0.07,
    }
    for key, value in inactive_defaults.items():
        if key not in saved_vpr and current_vpr.get(key) == value:
            saved_vpr[key] = value

    def active_fields(candidate, reference):
        if isinstance(reference, dict):
            return {
                key: active_fields(candidate[key], value)
                for key, value in reference.items()
                if key in candidate
            }
        return candidate

    if sections is not None:
        saved = {name: saved.get(name, {}) for name in sections}
        current = {name: current[name] for name in sections}
    return active_fields(saved, current) == current


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_fingerprints(config, *, generated: str | Path | None = None) -> dict[str, str]:
    paths = {
        "robotcar_manifest": config.data.robotcar_manifest,
        "robotcar_protocol": config.data.robotcar_protocol,
        "generator_manifest": config.data.generator_manifest,
        "vpr_source_manifest": config.data.vpr_source_manifest,
    }
    if generated is not None:
        paths["generated_manifest"] = generated
    return {name: file_sha256(path) for name, path in paths.items()}


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
