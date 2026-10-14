import json

import pytest

from a2b_lightvpr.reporting.paired_summary import summarize_paired_vpr
from a2b_lightvpr.reporting.seed0_summary import REPORT_PROTOCOLS


def _suite(path, value):
    path.write_text(json.dumps({
        "descriptor_dim": 544,
        "random_seed": 0,
        "initial_checkpoint_sha256": "dino-small",
        "training": {},
        "benchmark": {"parameters": 10},
        "datasets": {
            name: {
                "recalls": None if name.startswith("robotcar_seasons_") else {"1": value, "5": value + 0.1, "10": value + 0.2},
                "ground_truth_available": not name.startswith("robotcar_seasons_"),
                "evaluation_fingerprint": name,
            }
            for name in REPORT_PROTOCOLS
        },
    }), encoding="utf-8")


def test_paired_summary_reports_protocol_and_macro_deltas(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _suite(baseline, 0.5)
    _suite(candidate, 0.55)
    output = tmp_path / "summary.json"
    summarize_paired_vpr(baseline, candidate, "salad", output)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["descriptor_dim"] == 544
    assert report["night"]["delta"] == pytest.approx(0.05)
    assert report["protocols"]["tokyo_night"]["recall1_delta"] == pytest.approx(0.05)
    assert report["protocols"]["robotcar_seasons_night"]["candidate"] is None
    assert report["protocols"]["robotcar_seasons_night"]["recall1_delta"] is None


def test_paired_summary_rejects_unpaired_initialization(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _suite(baseline, 0.5)
    _suite(candidate, 0.55)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["initial_checkpoint_sha256"] = "different"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="initialization"):
        summarize_paired_vpr(baseline, candidate, "salad", tmp_path / "out.json")
