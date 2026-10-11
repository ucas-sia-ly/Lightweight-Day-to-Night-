import json

import pytest

from a2b_lightvpr.reporting.multiseed_summary import (
    REQUIRED_ABLATIONS,
    summarize_cdnd_multiseed,
)
from a2b_lightvpr.reporting.figures import _student_result
from a2b_lightvpr.reporting.seed0_summary import REPORT_PROTOCOLS


def _suite(path, seed, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "descriptor_dim": 512,
        "random_seed": seed,
        "model_family": "mixvpr",
        "aggregator": "mixvpr",
        "input_size": [320, 320],
        "manifest_sha256": "manifest",
        "initial_checkpoint_sha256": "init-512",
        "training": {"variant": path.stem},
        "datasets": {
            name: {
                "recalls": {"1": value, "5": value + 0.1, "10": value + 0.2},
                "evaluation_fingerprint": name,
            }
            for name in REPORT_PROTOCOLS
        },
    }), encoding="utf-8")


def _inputs(tmp_path):
    baseline, candidate = {}, {}
    for seed, baseline_value in enumerate((0.50, 0.52, 0.54)):
        baseline[seed] = tmp_path / f"baseline-{seed}.json"
        candidate[seed] = tmp_path / f"candidate-{seed}.json"
        _suite(baseline[seed], seed, baseline_value)
        _suite(candidate[seed], seed, baseline_value + 0.03)
    ablations = {}
    for offset, name in enumerate(sorted(REQUIRED_ABLATIONS)):
        ablations[name] = tmp_path / f"{name}.json"
        _suite(ablations[name], 0, 0.51 + offset / 100)
    return baseline, candidate, ablations


def test_multiseed_summary_reports_sample_std_and_paired_deltas(tmp_path):
    baseline, candidate, ablations = _inputs(tmp_path)
    output = tmp_path / "summary.json"
    summarize_cdnd_multiseed(baseline, candidate, ablations, output)
    report = json.loads(output.read_text(encoding="utf-8"))
    night = report["main"]["baseline"]["night_macro_recall1"]
    delta = report["main"]["paired_delta"]["night_macro_recall1"]
    assert night["count"] == 3
    assert night["sample_std"] == pytest.approx(0.02)
    assert delta["mean"] == pytest.approx(0.03)
    assert "significant" not in json.dumps(report)
    assert set(report["ablations_seed0"]) == REQUIRED_ABLATIONS


def test_multiseed_summary_rejects_seed_and_ablation_mismatches(tmp_path):
    baseline, candidate, ablations = _inputs(tmp_path)
    _suite(candidate[2], 7, 0.57)
    with pytest.raises(ValueError, match="Expected random seed 2"):
        summarize_cdnd_multiseed(baseline, candidate, ablations, tmp_path / "bad-seed.json")

    _suite(candidate[2], 2, 0.57)
    ablations.pop("shared_fifo")
    with pytest.raises(ValueError, match="Ablations must be exactly"):
        summarize_cdnd_multiseed(baseline, candidate, ablations, tmp_path / "bad-ablation.json")


def test_paper_plot_prefers_multiseed_mean_and_uses_seed_zero_resources(tmp_path):
    seed_zero = tmp_path / "seed-zero.json"
    _suite(seed_zero, 0, 0.5)
    accuracy, resources, statistic = _student_result({
        "schema": "cdnd_paper_experiments_v1",
        "main": {"cdnd": {"night_macro_recall1": {"mean": 0.625}}},
        "inputs": {"cdnd": {"0": str(seed_zero)}},
    })
    assert accuracy == pytest.approx(0.625)
    assert resources["random_seed"] == 0
    assert statistic == "three-seed mean"
