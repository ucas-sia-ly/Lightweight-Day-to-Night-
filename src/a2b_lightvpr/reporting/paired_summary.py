from __future__ import annotations

import json
from pathlib import Path

from ..checkpointing import write_json
from .protocols import (DAY_MACRO, NIGHT_MACRO, REPORT_PROTOCOLS, difference, excluded,
                        macro, protocol_result, recall, validate_protocol_pair)


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
    for protocol in REPORT_PROTOCOLS:
        validate_protocol_pair(baseline, candidate, protocol)

    baseline_night = macro(baseline, NIGHT_MACRO)
    candidate_night = macro(candidate, NIGHT_MACRO)
    baseline_day = macro(baseline, DAY_MACRO)
    candidate_day = macro(candidate, DAY_MACRO)
    report = {
        "schema": "paired_vpr_summary_v1",
        "label": label,
        "descriptor_dim": int(candidate["descriptor_dim"]),
        "random_seed": int(candidate.get("random_seed", 0)),
        "macro_definition": "fixed_supervised_protocols_v2",
        "night_macro_protocols": list(NIGHT_MACRO),
        "day_macro_protocols": list(DAY_MACRO),
        "excluded": {"baseline": excluded(baseline), "candidate": excluded(candidate)},
        "night": {
            "baseline_macro_recall1": baseline_night,
            "candidate_macro_recall1": candidate_night,
            "delta": difference(candidate_night, baseline_night),
        },
        "day": {
            "baseline_macro_recall1": baseline_day,
            "candidate_macro_recall1": candidate_day,
            "delta": difference(candidate_day, baseline_day),
        },
        "protocols": {
            protocol: {
                "baseline": protocol_result(baseline, protocol)["recalls"],
                "candidate": protocol_result(candidate, protocol)["recalls"],
                "baseline_status": protocol_result(baseline, protocol)["status"],
                "candidate_status": protocol_result(candidate, protocol)["status"],
                "recall1_delta": difference(recall(candidate, protocol), recall(baseline, protocol)),
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
