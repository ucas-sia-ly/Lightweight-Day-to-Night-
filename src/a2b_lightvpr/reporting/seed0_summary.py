from __future__ import annotations

import json
from pathlib import Path

from ..checkpointing import write_json
from .protocols import (DAY_MACRO, NIGHT_MACRO, REPORT_PROTOCOLS, difference, excluded,
                        macro, protocol_result, recall, validate_protocol_pair)


DESCRIPTOR_DIMS = (128, 512, 4096)


def _load_suite(path: str | Path, descriptor_dim: int) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("descriptor_dim", 0)) != descriptor_dim:
        raise ValueError(
            f"Expected {descriptor_dim}-D suite at {path}, got {payload.get('descriptor_dim')}"
        )
    return payload


def _resources(payload: dict, descriptor_dim: int) -> dict:
    benchmark = payload.get("benchmark", {})
    checkpoint = Path(payload.get("checkpoint", ""))
    return {
        "parameters": benchmark.get("parameters"),
        "checkpoint_mib": checkpoint.stat().st_size / 1024**2 if checkpoint.is_file() else None,
        "latency_p50_ms": benchmark.get("latency_ms", {}).get("p50"),
        "peak_cuda_mib": benchmark.get("peak_cuda_mib"),
        "descriptor_storage_per_million_images": {
            "fp32_gib": 1_000_000 * descriptor_dim * 4 / 1024**3,
            "fp16_gib": 1_000_000 * descriptor_dim * 2 / 1024**3,
        },
    }


def _validate_pair(baseline: dict, candidate: dict, descriptor_dim: int) -> None:
    paired_fields = (
        "model_family",
        "aggregator",
        "input_size",
        "manifest_sha256",
        "initial_checkpoint_sha256",
    )
    mismatches = {
        field: (baseline.get(field), candidate.get(field))
        for field in paired_fields
        if baseline.get(field) != candidate.get(field)
    }
    if mismatches:
        raise ValueError(f"Non-paired {descriptor_dim}-D baseline/CDND suites: {mismatches}")


def summarize_cdnd_seed0(
    baseline_paths: dict[int, str | Path],
    candidate_paths: dict[int, str | Path],
    output: str | Path,
) -> Path:
    """Create a descriptive report for one fixed run at each descriptor dimension."""
    expected = set(DESCRIPTOR_DIMS)
    if set(baseline_paths) != expected or set(candidate_paths) != expected:
        raise ValueError(f"Single-run summary requires dimensions {DESCRIPTOR_DIMS}")
    report = {
        "status": "single_run_development_only",
        "fixed_random_state": 0,
        "macro_definition": "fixed_supervised_protocols_v2",
        "night_macro_protocols": list(NIGHT_MACRO),
        "day_macro_protocols": list(DAY_MACRO),
        "dimensions": {},
        "capacity_delta_from_512": {},
        "limitations": [
            "One run per method; no cross-run variance estimate is available.",
            "Development protocols are not a sealed external test set.",
            "RobotCar Seasons public test poses are hidden and excluded from the fixed supervised macros.",
            "A fixed macro is unavailable if any required supervised protocol was not measured.",
        ],
    }
    loaded = {}
    for descriptor_dim in DESCRIPTOR_DIMS:
        baseline = _load_suite(baseline_paths[descriptor_dim], descriptor_dim)
        candidate = _load_suite(candidate_paths[descriptor_dim], descriptor_dim)
        _validate_pair(baseline, candidate, descriptor_dim)
        for protocol in REPORT_PROTOCOLS:
            validate_protocol_pair(baseline, candidate, protocol)
        baseline_night = macro(baseline, NIGHT_MACRO)
        candidate_night = macro(candidate, NIGHT_MACRO)
        baseline_day = macro(baseline, DAY_MACRO)
        candidate_day = macro(candidate, DAY_MACRO)
        per_protocol = {}
        for protocol in REPORT_PROTOCOLS:
            baseline_result = protocol_result(baseline, protocol)
            candidate_result = protocol_result(candidate, protocol)
            per_protocol[protocol] = {
                "baseline_recall1": recall(baseline, protocol),
                "cdnd_recall1": recall(candidate, protocol),
                "delta": difference(recall(candidate, protocol), recall(baseline, protocol)),
                "baseline_status": baseline_result["status"],
                "cdnd_status": candidate_result["status"],
            }
        report["dimensions"][str(descriptor_dim)] = {
            "baseline": {
                "night_macro_recall1": baseline_night,
                "day_macro_recall1": baseline_day,
            },
            "cdnd": {
                "night_macro_recall1": candidate_night,
                "day_macro_recall1": candidate_day,
            },
            "cdnd_delta": {
                "night_macro_recall1": difference(candidate_night, baseline_night),
                "day_macro_recall1": difference(candidate_day, baseline_day),
            },
            "per_protocol": per_protocol,
            "excluded": {"baseline": excluded(baseline), "cdnd": excluded(candidate)},
            "resources": _resources(candidate, descriptor_dim),
            "inputs": {
                "baseline": str(Path(baseline_paths[descriptor_dim]).resolve()),
                "cdnd": str(Path(candidate_paths[descriptor_dim]).resolve()),
            },
        }
        loaded[descriptor_dim] = report["dimensions"][str(descriptor_dim)]

    reference = loaded[512]["cdnd"]
    for descriptor_dim in (128, 4096):
        current = loaded[descriptor_dim]["cdnd"]
        report["capacity_delta_from_512"][str(descriptor_dim)] = {
            "night_macro_recall1": (
                difference(current["night_macro_recall1"], reference["night_macro_recall1"])
            ),
            "day_macro_recall1": difference(current["day_macro_recall1"], reference["day_macro_recall1"]),
        }
    write_json(report, output)
    return Path(output)
