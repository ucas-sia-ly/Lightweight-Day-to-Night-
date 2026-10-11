from __future__ import annotations

import json
from pathlib import Path

from ..checkpointing import write_json
from .models import _registry, audit_reference_models


def audit_reference_coverage(
    registry_path: str | Path,
    reference_root: str | Path,
    resource_paths: list[str | Path],
    output: str | Path,
) -> Path:
    registry = _registry(registry_path)
    readiness = audit_reference_models(registry_path)["models"]
    resources = {}
    for path in resource_paths:
        path = Path(path)
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            resources.update(payload.get("models", {}))
    reference_root = Path(reference_root)
    models = {}
    for name in registry:
        native = reference_root / name / "final_suite" / "native" / "adapted" / "summary.json"
        unified = reference_root / name / "final_suite" / "unified" / "adapted" / "summary.json"
        accuracy = native if native.is_file() else unified if unified.is_file() else None
        ready = bool(readiness[name]["ready"])
        missing = []
        if not ready:
            missing.append("required model or projection artifact")
        if accuracy is None:
            missing.append("accuracy suite")
        if name not in resources:
            missing.append("resource benchmark")
        models[name] = {
            "ready": ready,
            "accuracy_summary": str(accuracy.resolve()) if accuracy else None,
            "resource_available": name in resources,
            "complete": ready and accuracy is not None and name in resources,
            "missing": missing,
        }
    result = {
        "schema": "reference_coverage_v1",
        "registry": str(Path(registry_path).resolve()),
        "complete": all(item["complete"] for item in models.values()),
        "models": models,
    }
    output = Path(output)
    write_json(result, output)
    return output
