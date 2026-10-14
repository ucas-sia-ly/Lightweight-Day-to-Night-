import json

import pytest

from a2b_lightvpr.reporting.seed0_summary import REPORT_PROTOCOLS, summarize_cdnd_seed0


def _suite(path, dimension, value, fingerprint_suffix=""):
    datasets = {
        name: {
            "recalls": None if name.startswith("robotcar_seasons_") else {"1": value},
            "ground_truth_available": not name.startswith("robotcar_seasons_"),
            "evaluation_fingerprint": name + fingerprint_suffix,
        }
        for name in REPORT_PROTOCOLS
    }
    path.write_text(json.dumps({
        "descriptor_dim": dimension,
        "model_family": "mixvpr",
        "aggregator": "mixvpr",
        "input_size": [320, 320],
        "manifest_sha256": "manifest",
        "initial_checkpoint_sha256": f"init-{dimension}",
        "checkpoint": "missing-fixture.pt",
        "benchmark": {
            "parameters": dimension * 10,
            "latency_ms": {"p50": dimension / 100},
            "peak_cuda_mib": 100,
        },
        "datasets": datasets,
    }), encoding="utf-8")


def test_single_run_summary_reports_deltas_without_statistics(tmp_path):
    baseline, candidate = {}, {}
    for dimension in (128, 512, 4096):
        baseline[dimension] = tmp_path / f"b-{dimension}.json"
        candidate[dimension] = tmp_path / f"c-{dimension}.json"
        _suite(baseline[dimension], dimension, 0.5)
        _suite(candidate[dimension], dimension, 0.55 + dimension / 100_000)
    output = tmp_path / "summary.json"
    summarize_cdnd_seed0(baseline, candidate, output)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["fixed_random_state"] == 0
    assert report["dimensions"]["512"]["cdnd_delta"]["night_macro_recall1"] > 0
    assert not {"mean", "std", "ci95", "significant"} & set(report)
    assert set(report["capacity_delta_from_512"]) == {"128", "4096"}
    assert report["night_macro_protocols"] == ["svox_night", "tokyo_night"]
    assert report["dimensions"]["512"]["per_protocol"]["robotcar_seasons_night"]["delta"] is None
    assert "robotcar_seasons_night" in report["dimensions"]["512"]["excluded"]["cdnd"]


def test_single_run_summary_rejects_dimension_and_protocol_mismatches(tmp_path):
    baseline, candidate = {}, {}
    for dimension in (128, 512, 4096):
        baseline[dimension] = tmp_path / f"b-{dimension}.json"
        candidate[dimension] = tmp_path / f"c-{dimension}.json"
        _suite(baseline[dimension], dimension, 0.5)
        _suite(candidate[dimension], dimension, 0.6)
    _suite(candidate[128], 512, 0.6)
    with pytest.raises(ValueError, match="Expected 128-D"):
        summarize_cdnd_seed0(baseline, candidate, tmp_path / "bad-dim.json")
    _suite(candidate[128], 128, 0.6, fingerprint_suffix="-changed")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        summarize_cdnd_seed0(baseline, candidate, tmp_path / "bad-protocol.json")


def test_single_run_summary_rejects_non_paired_inputs(tmp_path):
    baseline, candidate = {}, {}
    for dimension in (128, 512, 4096):
        baseline[dimension] = tmp_path / f"b-{dimension}.json"
        candidate[dimension] = tmp_path / f"c-{dimension}.json"
        _suite(baseline[dimension], dimension, 0.5)
        _suite(candidate[dimension], dimension, 0.6)
    payload = json.loads(candidate[512].read_text(encoding="utf-8"))
    payload["manifest_sha256"] = "different"
    candidate[512].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Non-paired 512-D"):
        summarize_cdnd_seed0(baseline, candidate, tmp_path / "bad-pair.json")


def test_single_run_missing_supervised_protocol_does_not_shrink_macro(tmp_path):
    baseline, candidate = {}, {}
    for dimension in (128, 512, 4096):
        baseline[dimension], candidate[dimension] = tmp_path / f"b{dimension}.json", tmp_path / f"c{dimension}.json"
        _suite(baseline[dimension], dimension, 0.5)
        _suite(candidate[dimension], dimension, 0.6)
    payload = json.loads(candidate[512].read_text())
    del payload["datasets"]["tokyo_night"]
    candidate[512].write_text(json.dumps(payload))
    output = tmp_path / "summary.json"
    summarize_cdnd_seed0(baseline, candidate, output)
    result = json.loads(output.read_text())["dimensions"]["512"]
    assert result["cdnd"]["night_macro_recall1"] is None
    assert result["cdnd_delta"]["night_macro_recall1"] is None
    assert result["excluded"]["cdnd"]["tokyo_night"]["status"] == "not_evaluated"
