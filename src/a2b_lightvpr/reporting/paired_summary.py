from __future__ import annotations

import json
from pathlib import Path

from ..checkpointing import write_json
from .seed0_summary import DAY_MACRO, NIGHT_MACRO, REPORT_PROTOCOLS


def _macro(payload: dict, protocols: tuple[str, ...]) -> float:
    return sum(
        float(payload["datasets"][name]["recalls"]["1"]) for name in protocols
    ) / len(protocols)


def summarize_paired_vpr(
    baseline_path: str | Path,
    candidate_path: str | Path,
    label: str,
    output: str | Path,
) -> Path:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    if baseline.get("descriptor_dim") != candidate.get("descriptor_dim"):
        raise ValueError("Paired suites have different descriptor dimensions")
    if baseline.get("initial_checkpoint_sha256") != candidate.get("initial_checkpoint_sha256"):
        raise ValueError("Paired suites have different initialization checkpoints")
    missing_baseline = sorted(set(REPORT_PROTOCOLS) - set(baseline.get("datasets", {})))
    missing_candidate = sorted(set(REPORT_PROTOCOLS) - set(candidate.get("datasets", {})))
    if missing_baseline or missing_candidate:
        raise ValueError(
            "Paired suites are missing protocols: "
            f"baseline={missing_baseline}, candidate={missing_candidate}"
        )
    for protocol in REPORT_PROTOCOLS:
        first = baseline["datasets"][protocol].get("evaluation_fingerprint")
        second = candidate["datasets"][protocol].get("evaluation_fingerprint")
        if not first or first != second:
            raise ValueError(f"Evaluation fingerprint mismatch for {protocol}")

    baseline_night = _macro(baseline, NIGHT_MACRO)
    candidate_night = _macro(candidate, NIGHT_MACRO)
    baseline_day = _macro(baseline, DAY_MACRO)
    candidate_day = _macro(candidate, DAY_MACRO)
    report = {
        "schema": "paired_vpr_summary_v1",
        "label": label,
        "descriptor_dim": int(candidate["descriptor_dim"]),
        "random_seed": int(candidate.get("random_seed", 0)),
        "night": {
            "baseline_macro_recall1": baseline_night,
            "candidate_macro_recall1": candidate_night,
            "delta": candidate_night - baseline_night,
        },
        "day": {
            "baseline_macro_recall1": baseline_day,
            "candidate_macro_recall1": candidate_day,
            "delta": candidate_day - baseline_day,
        },
        "protocols": {
            protocol: {
                "baseline": baseline["datasets"][protocol]["recalls"],
                "candidate": candidate["datasets"][protocol]["recalls"],
                "recall1_delta": (
                    float(candidate["datasets"][protocol]["recalls"]["1"])
                    - float(baseline["datasets"][protocol]["recalls"]["1"])
                ),
            }
            for protocol in REPORT_PROTOCOLS
        },
        "training": {
            "baseline": baseline.get("training", {}),
            "candidate": candidate.get("training", {}),
        },
        "resources": candidate.get("benchmark", {}),
        "inputs": {
            "baseline": str(Path(baseline_path).resolve()),
            "candidate": str(Path(candidate_path).resolve()),
        },
    }
    output = Path(output)
    write_json(report, output)
    return output
