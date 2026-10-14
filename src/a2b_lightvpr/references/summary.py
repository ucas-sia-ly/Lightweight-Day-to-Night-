from __future__ import annotations

import json
from pathlib import Path

from ..checkpointing import write_json
from .models import _registry
from ..reporting.protocols import (DAY_MACRO, NIGHT_MACRO, REPORT_PROTOCOLS,
                                  excluded, macro, protocol_result)



def summarize_small_references(
    registry_path: str | Path,
    reference_root: str | Path,
    resources_paths: list[str | Path],
    output: str | Path,
    *,
    model_names: tuple[str, ...] = ("all",),
    paired_summary: str | Path | None = None,
) -> Path:
    registry = _registry(registry_path)
    selected = tuple(registry) if "all" in model_names else model_names
    unknown = sorted(set(selected) - set(registry))
    if unknown:
        raise ValueError(f"Unknown Small-reference models: {unknown}")
    resources = {}
    for path in resources_paths:
        resources.update(
            json.loads(Path(path).read_text(encoding="utf-8")).get("models", {})
        )
    models = {}
    for name in selected:
        spec = registry[name]
        summary_path = (
            Path(reference_root) / name / "final_suite" / "native" / "adapted" / "summary.json"
        )
        if not summary_path.is_file() or name not in resources:
            raise FileNotFoundError(f"Incomplete Small-reference artifacts for {name}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        models[name] = {
            "backbone": spec.get("backbone", ""),
            "descriptor_dim": int(spec["descriptor_dim"]),
            "official_checkpoint": bool(spec.get("official_checkpoint", True)),
            "training_status": spec.get("training_status", "official"),
            "per_protocol": {
                protocol: protocol_result(summary, protocol)
                for protocol in REPORT_PROTOCOLS
            },
            "night_macro_recall1": macro(summary, NIGHT_MACRO),
            "day_macro_recall1": macro(summary, DAY_MACRO),
            "excluded": excluded(summary),
            "resources": resources[name],
        }
    output = Path(output)
    payload = {
        "schema": "dinov2_small_comparison_v1",
        "macro_definition": "fixed_supervised_protocols_v2",
        "night_macro_protocols": list(NIGHT_MACRO),
        "day_macro_protocols": list(DAY_MACRO),
        "models": models,
    }
    if paired_summary is not None:
        payload["paired_transfer"] = json.loads(
            Path(paired_summary).read_text(encoding="utf-8")
        )
    write_json(payload, output)
    return output
