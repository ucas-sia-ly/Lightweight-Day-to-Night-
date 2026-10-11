from __future__ import annotations

import json
import statistics
from pathlib import Path

from ..checkpointing import write_json
from .seed0_summary import DAY_MACRO, NIGHT_MACRO, REPORT_PROTOCOLS


RECALLS = (1, 5, 10)
REQUIRED_ABLATIONS = {
    "day_only",
    "fake_to_day_only",
    "no_fifo",
    "no_self_mask",
    "shared_fifo",
}


def _load_suite(path: str | Path, expected_seed: int | None = None) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("descriptor_dim", 0)) != 512:
        raise ValueError(f"Expected a 512-D suite at {path}")
    missing = sorted(set(REPORT_PROTOCOLS) - set(payload.get("datasets", {})))
    if missing:
        raise ValueError(f"Suite is missing protocols at {path}: {missing}")
    seed = int(payload.get("random_seed", 0))
    if expected_seed is not None and seed != expected_seed:
        raise ValueError(f"Expected random seed {expected_seed} at {path}, got {seed}")
    return payload


def _fingerprint(payload: dict, protocol: str) -> str:
    metrics = payload["datasets"][protocol]
    return metrics.get("evaluation_fingerprint", metrics.get("protocol_fingerprint", ""))


def _recall(payload: dict, protocol: str, recall: int) -> float:
    return float(payload["datasets"][protocol]["recalls"][str(recall)])


def _macro(payload: dict, protocols: tuple[str, ...]) -> float:
    return statistics.fmean(_recall(payload, protocol, 1) for protocol in protocols)


def _summary(values: list[float]) -> dict:
    if len(values) < 2:
        raise ValueError("Multi-seed statistics require at least two values")
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "count": len(values),
    }


def _pair_identity(payload: dict) -> dict:
    return {
        "model_family": payload.get("model_family"),
        "aggregator": payload.get("aggregator"),
        "input_size": payload.get("input_size"),
        "manifest_sha256": payload.get("manifest_sha256"),
        "initial_checkpoint_sha256": payload.get("initial_checkpoint_sha256"),
    }


def _validate_runs(payloads: list[dict]) -> None:
    identity = _pair_identity(payloads[0])
    for payload in payloads[1:]:
        if _pair_identity(payload) != identity:
            raise ValueError("Paper experiment suites do not share the same model/data identity")
    for protocol in REPORT_PROTOCOLS:
        fingerprints = {_fingerprint(payload, protocol) for payload in payloads}
        if "" in fingerprints or len(fingerprints) != 1:
            raise ValueError(f"Evaluation fingerprint mismatch for {protocol}")


def _aggregate_group(payloads: list[dict]) -> dict:
    return {
        "night_macro_recall1": _summary([_macro(item, NIGHT_MACRO) for item in payloads]),
        "day_macro_recall1": _summary([_macro(item, DAY_MACRO) for item in payloads]),
        "protocols": {
            protocol: {
                f"recall{recall}": _summary(
                    [_recall(item, protocol, recall) for item in payloads]
                )
                for recall in RECALLS
            }
            for protocol in REPORT_PROTOCOLS
        },
    }


def _aggregate_deltas(baselines: list[dict], candidates: list[dict]) -> dict:
    def values(metric):
        return [
            metric(candidate) - metric(baseline)
            for baseline, candidate in zip(baselines, candidates)
        ]

    return {
        "night_macro_recall1": _summary(values(lambda item: _macro(item, NIGHT_MACRO))),
        "day_macro_recall1": _summary(values(lambda item: _macro(item, DAY_MACRO))),
        "protocols": {
            protocol: {
                f"recall{recall}": _summary(
                    values(lambda item, p=protocol, k=recall: _recall(item, p, k))
                )
                for recall in RECALLS
            }
            for protocol in REPORT_PROTOCOLS
        },
    }


def _single_run(payload: dict) -> dict:
    return {
        "random_seed": int(payload.get("random_seed", 0)),
        "night_macro_recall1": _macro(payload, NIGHT_MACRO),
        "day_macro_recall1": _macro(payload, DAY_MACRO),
        "training": payload.get("training", {}),
        "protocols": {
            protocol: {
                f"recall{recall}": _recall(payload, protocol, recall)
                for recall in RECALLS
            }
            for protocol in REPORT_PROTOCOLS
        },
    }


def summarize_cdnd_multiseed(
    baseline_paths: dict[int, str | Path],
    candidate_paths: dict[int, str | Path],
    ablation_paths: dict[str, str | Path],
    output: str | Path,
) -> Path:
    seeds = sorted(baseline_paths)
    if seeds != sorted(candidate_paths) or len(seeds) < 3:
        raise ValueError("Baseline and CDND must provide the same set of at least three seeds")
    if set(ablation_paths) != REQUIRED_ABLATIONS:
        raise ValueError(f"Ablations must be exactly {sorted(REQUIRED_ABLATIONS)}")

    baselines = [_load_suite(baseline_paths[seed], seed) for seed in seeds]
    candidates = [_load_suite(candidate_paths[seed], seed) for seed in seeds]
    ablations = {
        name: _load_suite(path, 0) for name, path in sorted(ablation_paths.items())
    }
    _validate_runs([*baselines, *candidates, *ablations.values()])

    report = {
        "schema": "cdnd_paper_experiments_v1",
        "status": "multi_seed_descriptive",
        "random_seeds": seeds,
        "statistics": "mean_and_sample_standard_deviation; no significance claim",
        "night_macro_protocols": list(NIGHT_MACRO),
        "day_macro_protocols": list(DAY_MACRO),
        "main": {
            "baseline": _aggregate_group(baselines),
            "cdnd": _aggregate_group(candidates),
            "paired_delta": _aggregate_deltas(baselines, candidates),
        },
        "ablations_seed0": {
            name: _single_run(payload) for name, payload in ablations.items()
        },
        "inputs": {
            "baseline": {str(seed): str(Path(baseline_paths[seed]).resolve()) for seed in seeds},
            "cdnd": {str(seed): str(Path(candidate_paths[seed]).resolve()) for seed in seeds},
            "ablations": {
                name: str(Path(path).resolve()) for name, path in sorted(ablation_paths.items())
            },
        },
    }
    output = Path(output)
    write_json(report, output)
    return output
