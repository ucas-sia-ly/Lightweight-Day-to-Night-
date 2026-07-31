from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .checkpointing import write_json


def _valid_success(metrics: dict) -> np.ndarray:
    success = np.asarray(metrics["success_by_k"]["1"], dtype=np.float64)
    valid = np.asarray(metrics["valid_by_query"], dtype=bool)
    if success.shape != valid.shape:
        raise ValueError("R@1 success and valid-query masks must have the same shape")
    selected = success[valid]
    if selected.size == 0:
        raise ValueError("Validation protocol has no valid queries")
    return selected


def stratified_macro_bootstrap(
    candidate: dict[str, list[int] | np.ndarray],
    baseline: dict[str, list[int] | np.ndarray],
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> dict:
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    if set(candidate) != set(baseline) or not candidate:
        raise ValueError("Candidate and baseline must contain the same non-empty strata")
    differences = {}
    for name in sorted(candidate):
        first = np.asarray(candidate[name], dtype=np.float64)
        second = np.asarray(baseline[name], dtype=np.float64)
        if first.shape != second.shape:
            raise ValueError(f"Paired bootstrap query mismatch for {name}")
        if first.ndim != 1 or first.size == 0:
            raise ValueError(f"Bootstrap stratum {name} must be a non-empty vector")
        differences[name] = first - second

    point_deltas = {name: float(values.mean()) for name, values in differences.items()}
    macro_delta = float(np.mean(list(point_deltas.values())))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        count = min(256, samples - start)
        stratum_draws = []
        for values in differences.values():
            indices = rng.integers(0, len(values), size=(count, len(values)))
            stratum_draws.append(values[indices].mean(axis=1))
        draws[start : start + count] = np.mean(stratum_draws, axis=0)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    positive_fraction = float(np.mean(draws > 0.0))
    non_positive_fraction = float(np.mean(draws <= 0.0))
    return {
        "delta": macro_delta,
        "ci95": [float(lower), float(upper)],
        "significant": bool(lower > 0.0),
        "positive_draw_fraction": positive_fraction,
        "non_positive_draw_fraction": non_positive_fraction,
        "one_sided_p_non_positive": non_positive_fraction,
        "samples": int(samples),
        "stratum_deltas": point_deltas,
    }


def _read(path: str | Path) -> tuple[Path, dict]:
    resolved = Path(path).expanduser().resolve()
    return resolved, json.loads(resolved.read_text(encoding="utf-8"))


def _validation_signature(metrics: dict) -> dict:
    keys = (
        "split",
        "database",
        "queries",
        "total_queries",
        "valid_queries",
        "positive_radius_m",
        "positive_yaw_deg",
        "descriptor_dim",
    )
    return {key: metrics.get(key) for key in keys}


def _require_best_epoch(validation_path: Path, suite: dict, label: str) -> None:
    match = re.fullmatch(r"epoch_(\d+)\.json", validation_path.name)
    if match is None:
        raise ValueError(f"{label} validation filename must be epoch_NNN.json")
    validation_epoch = int(match.group(1))
    checkpoint_epoch = int(suite.get("checkpoint_epoch", -1))
    if validation_epoch != checkpoint_epoch:
        raise ValueError(
            f"{label} validation epoch {validation_epoch} does not match "
            f"suite checkpoint epoch {checkpoint_epoch}"
        )


def compare_loss_selection(
    candidate_validation: str | Path,
    baseline_validation: str | Path,
    candidate_suite: str | Path,
    baseline_suite: str | Path,
    output: str | Path,
    *,
    candidate_name: str = "preserve",
    baseline_name: str = "no-preserve",
    samples: int = 10_000,
    seed: int = 0,
) -> Path:
    candidate_validation_path, candidate_robotcar = _read(candidate_validation)
    baseline_validation_path, baseline_robotcar = _read(baseline_validation)
    candidate_suite_path, candidate_payload = _read(candidate_suite)
    baseline_suite_path, baseline_payload = _read(baseline_suite)

    _require_best_epoch(candidate_validation_path, candidate_payload, "Candidate")
    _require_best_epoch(baseline_validation_path, baseline_payload, "Baseline")
    if _validation_signature(candidate_robotcar) != _validation_signature(baseline_robotcar):
        raise ValueError("RobotCar validation protocols do not match")
    if candidate_robotcar["valid_by_query"] != baseline_robotcar["valid_by_query"]:
        raise ValueError("RobotCar validation valid-query masks do not match")

    identity_keys = (
        "manifest_sha256",
        "initial_checkpoint_sha256",
        "model_family",
        "aggregator",
        "descriptor_dim",
        "input_size",
    )
    mismatches = {
        key: (candidate_payload.get(key), baseline_payload.get(key))
        for key in identity_keys
        if candidate_payload.get(key) != baseline_payload.get(key)
    }
    if mismatches:
        raise ValueError(f"Loss-ablation suite identities do not match: {mismatches}")

    try:
        candidate_msls = candidate_payload["datasets"]["msls_night"]
        baseline_msls = baseline_payload["datasets"]["msls_night"]
    except KeyError as error:
        raise ValueError("Both suites must contain msls_night") from error
    candidate_msls_fingerprint = candidate_msls.get(
        "query_protocol_fingerprint", candidate_msls.get("protocol_fingerprint")
    )
    baseline_msls_fingerprint = baseline_msls.get(
        "query_protocol_fingerprint", baseline_msls.get("protocol_fingerprint")
    )
    if candidate_msls_fingerprint != baseline_msls_fingerprint:
        raise ValueError("MSLS validation protocol fingerprints do not match")
    if candidate_msls["valid_by_query"] != baseline_msls["valid_by_query"]:
        raise ValueError("MSLS validation valid-query masks do not match")

    candidate_success = {
        "robotcar_val": _valid_success(candidate_robotcar),
        "msls_val": _valid_success(candidate_msls),
    }
    baseline_success = {
        "robotcar_val": _valid_success(baseline_robotcar),
        "msls_val": _valid_success(baseline_msls),
    }
    bootstrap = stratified_macro_bootstrap(
        candidate_success, baseline_success, samples=samples, seed=seed
    )
    validation = {}
    for name in candidate_success:
        candidate_recall = float(candidate_success[name].mean())
        baseline_recall = float(baseline_success[name].mean())
        validation[name] = {
            "candidate_recall_at_1": candidate_recall,
            "baseline_recall_at_1": baseline_recall,
            "delta": candidate_recall - baseline_recall,
            "valid_queries": int(len(candidate_success[name])),
        }
    candidate_macro = float(np.mean([
        values["candidate_recall_at_1"] for values in validation.values()
    ]))
    baseline_macro = float(np.mean([
        values["baseline_recall_at_1"] for values in validation.values()
    ]))
    macro_delta = candidate_macro - baseline_macro
    candidate_selected = bool(
        macro_delta > 0.0 and bootstrap["one_sided_p_non_positive"] < 0.05
    )
    payload = {
        "candidate": candidate_name,
        "baseline": baseline_name,
        "candidate_validation": str(candidate_validation_path),
        "baseline_validation": str(baseline_validation_path),
        "candidate_suite": str(candidate_suite_path),
        "baseline_suite": str(baseline_suite_path),
        "selection_datasets": ["robotcar_val", "msls_val"],
        "validation": validation,
        "macro_recall_at_1": {
            "candidate": candidate_macro,
            "baseline": baseline_macro,
            "delta": macro_delta,
        },
        "stratified_paired_bootstrap_r1": bootstrap,
        "candidate_selected": candidate_selected,
        "selected": candidate_name if candidate_selected else baseline_name,
        "rule": (
            "Select preserve when the equally weighted RobotCar-night/MSLS-night "
            "macro R@1 delta is positive and fewer than 5% of stratified paired-"
            "bootstrap draws are non-positive. Day and test results do not select loss."
        ),
    }
    output_path = Path(output)
    write_json(payload, output_path)
    return output_path


def compare_universal_loss_selection(
    candidate_validation: str | Path,
    baseline_validation: str | Path,
    candidate_suite: str | Path,
    baseline_suite: str | Path,
    output: str | Path,
    *,
    candidate_name: str = "preserve",
    baseline_name: str = "no-preserve",
    samples: int = 10_000,
    seed: int = 0,
) -> Path:
    """Compatibility alias; day metrics must never affect preserve selection."""
    return compare_loss_selection(
        candidate_validation,
        baseline_validation,
        candidate_suite,
        baseline_suite,
        output,
        candidate_name=candidate_name,
        baseline_name=baseline_name,
        samples=samples,
        seed=seed,
    )
