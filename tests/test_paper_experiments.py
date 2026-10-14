import json

import pytest

from a2b_lightvpr.reporting.multiseed_summary import (
    REQUIRED_ABLATIONS,
    summarize_cdnd_multiseed,
)
from a2b_lightvpr.reporting.figures import _student_result
from a2b_lightvpr.reporting.seed0_summary import NIGHT_MACRO, REPORT_PROTOCOLS


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
                "recalls": None if name.startswith("robotcar_seasons_") else {"1": value, "5": value + 0.1, "10": value + 0.2},
                "ground_truth_available": not name.startswith("robotcar_seasons_"),
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
    unlabelled = report["main"]["cdnd"]["protocols"]["robotcar_seasons_night"]["recall1"]
    assert unlabelled["mean"] is None
    assert unlabelled["values"] == [None, None, None]


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
        "night_macro_protocols": list(NIGHT_MACRO),
        "main": {"cdnd": {"night_macro_recall1": {"mean": 0.625}}},
        "inputs": {"cdnd": {"0": str(seed_zero)}},
    })
    assert accuracy == pytest.approx(0.625)
    assert resources["random_seed"] == 0
    assert statistic == "three-seed mean"


def test_multiseed_missing_measurement_is_not_replaced_by_available_seeds(tmp_path):
    baseline, candidate, ablations = _inputs(tmp_path)
    payload = json.loads(candidate[1].read_text())
    payload["datasets"]["svox_night"] = {"status": "skipped_missing_images", "recalls": None}
    candidate[1].write_text(json.dumps(payload))
    output = tmp_path / "summary.json"
    summarize_cdnd_multiseed(baseline, candidate, ablations, output)
    report = json.loads(output.read_text())
    summary = report["main"]["cdnd"]["night_macro_recall1"]
    assert summary["values"] == [pytest.approx(0.53), None, pytest.approx(0.57)]
    assert summary["mean"] is None
    assert summary["sample_std"] is None
    assert report["main"]["paired_delta"]["night_macro_recall1"]["mean"] is None
    assert report["main"]["cdnd"]["excluded"][1]["svox_night"]["status"] == "skipped_missing_images"


def test_paper_plot_rejects_summaries_without_current_fixed_macro():
    with pytest.raises(ValueError, match="fixed night macro"):
        _student_result({
            "schema": "cdnd_paper_experiments_v1",
            "main": {"cdnd": {"night_macro_recall1": {"mean": 0.6}}},
        })


def test_paper_plot_records_missing_model_accuracy_without_plotting_zero(tmp_path):
    from a2b_lightvpr.reporting.figures import plot_accuracy_efficiency

    pytest.importorskip("matplotlib")
    student = tmp_path / "student.json"
    student.write_text(json.dumps({
        "descriptor_dim": 512, "benchmark": {"parameters": 10},
        "datasets": {"svox_night": {"recalls": {"1": 0.8}}},
    }))
    output = tmp_path / "figure.pdf"
    plot_accuracy_efficiency(student, tmp_path / "references", [], output)
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["points"] == []
    assert "CDND-512" in metadata["excluded"]
    assert metadata["night_macro_protocols"] == list(NIGHT_MACRO)
