import json

import pytest

from a2b_lightvpr.loss_selection import (
    compare_loss_selection,
    compare_universal_loss_selection,
    stratified_macro_bootstrap,
)


def test_stratified_macro_bootstrap_weights_datasets_equally():
    result = stratified_macro_bootstrap(
        {
            "large": [1] * 100,
            "small": [0] * 10,
        },
        {
            "large": [0] * 100,
            "small": [0] * 10,
        },
        samples=200,
        seed=4,
    )
    assert result["delta"] == pytest.approx(0.5)
    assert result["stratum_deltas"] == {"large": 1.0, "small": 0.0}


def _metrics(success, *, split="val", fingerprint=None):
    payload = {
        "split": split,
        "database": 2,
        "queries": len(success),
        "total_queries": len(success),
        "valid_queries": len(success),
        "positive_radius_m": 10.0,
        "positive_yaw_deg": 30.0,
        "descriptor_dim": 512,
        "recalls": {"1": sum(success) / len(success)},
        "success_by_k": {"1": success},
        "valid_by_query": [1] * len(success),
    }
    if fingerprint is not None:
        payload["protocol_fingerprint"] = fingerprint
    return payload


def _suite(path, epoch, msls_success, fingerprint="msls-protocol"):
    payload = {
        "checkpoint_epoch": epoch,
        "manifest_sha256": "manifest",
        "initial_checkpoint_sha256": "initial",
        "model_family": "mixvpr",
        "aggregator": "mixvpr",
        "descriptor_dim": 512,
        "input_size": [320, 320],
        "datasets": {
            "msls_night": _metrics(
                msls_success, split="val", fingerprint=fingerprint
            )
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _universal_suite(path, epoch, values):
    payload = {
        "checkpoint_epoch": epoch,
        "manifest_sha256": "manifest",
        "initial_checkpoint_sha256": "initial",
        "model_family": "mixvpr",
        "aggregator": "mixvpr",
        "descriptor_dim": 512,
        "input_size": [320, 320],
        "datasets": {
            name: _metrics(success, fingerprint=f"{name}-protocol")
            for name, success in values.items()
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_loss_selection_requires_both_datasets_and_significance(tmp_path):
    candidate_validation = tmp_path / "candidate" / "epoch_003.json"
    baseline_validation = tmp_path / "baseline" / "epoch_004.json"
    candidate_validation.parent.mkdir()
    baseline_validation.parent.mkdir()
    candidate_validation.write_text(
        json.dumps(_metrics([1] * 20)), encoding="utf-8"
    )
    baseline_validation.write_text(
        json.dumps(_metrics([0] * 20)), encoding="utf-8"
    )
    candidate_suite = tmp_path / "candidate-suite.json"
    baseline_suite = tmp_path / "baseline-suite.json"
    _suite(candidate_suite, 3, [1] * 10)
    _suite(baseline_suite, 4, [0] * 10)

    output = compare_loss_selection(
        candidate_validation,
        baseline_validation,
        candidate_suite,
        baseline_suite,
        tmp_path / "selection.json",
        samples=200,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["candidate_selected"] is True
    assert payload["selected"] == "preserve"
    assert payload["macro_recall_at_1"]["delta"] == pytest.approx(1.0)
    assert payload["stratified_paired_bootstrap_r1"]["one_sided_p_non_positive"] == 0.0


def test_night_selection_allows_one_stratum_to_drop_when_macro_is_supported(tmp_path):
    candidate_validation = tmp_path / "candidate" / "epoch_003.json"
    baseline_validation = tmp_path / "baseline" / "epoch_003.json"
    candidate_validation.parent.mkdir()
    baseline_validation.parent.mkdir()
    candidate_validation.write_text(json.dumps(_metrics([1] * 18 + [0] * 2)), encoding="utf-8")
    baseline_validation.write_text(json.dumps(_metrics([1] * 19 + [0])), encoding="utf-8")
    candidate_suite = tmp_path / "candidate-suite.json"
    baseline_suite = tmp_path / "baseline-suite.json"
    _suite(candidate_suite, 3, [1] * 20)
    _suite(baseline_suite, 3, [0] * 20)

    output = compare_loss_selection(
        candidate_validation, baseline_validation, candidate_suite, baseline_suite,
        tmp_path / "selection.json", samples=1000, seed=3,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["validation"]["robotcar_val"]["delta"] < 0
    assert payload["macro_recall_at_1"]["delta"] > 0
    assert payload["candidate_selected"] is True


def test_compare_loss_selection_rejects_suite_identity_mismatch(tmp_path):
    candidate_validation = tmp_path / "candidate" / "epoch_003.json"
    baseline_validation = tmp_path / "baseline" / "epoch_003.json"
    candidate_validation.parent.mkdir()
    baseline_validation.parent.mkdir()
    metrics = _metrics([1, 0])
    candidate_validation.write_text(json.dumps(metrics), encoding="utf-8")
    baseline_validation.write_text(json.dumps(metrics), encoding="utf-8")
    candidate_suite = tmp_path / "candidate-suite.json"
    baseline_suite = tmp_path / "baseline-suite.json"
    _suite(candidate_suite, 3, [1, 0])
    _suite(baseline_suite, 3, [1, 0])
    payload = json.loads(baseline_suite.read_text(encoding="utf-8"))
    payload["manifest_sha256"] = "different"
    baseline_suite.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identities"):
        compare_loss_selection(
            candidate_validation,
            baseline_validation,
            candidate_suite,
            baseline_suite,
            tmp_path / "selection.json",
            samples=10,
        )


def test_legacy_universal_command_ignores_day_metrics(tmp_path):
    candidate_validation = tmp_path / "candidate" / "epoch_003.json"
    baseline_validation = tmp_path / "baseline" / "epoch_003.json"
    candidate_validation.parent.mkdir()
    baseline_validation.parent.mkdir()
    candidate_validation.write_text(json.dumps(_metrics([1] * 20)), encoding="utf-8")
    baseline_validation.write_text(json.dumps(_metrics([0] * 20)), encoding="utf-8")
    candidate_suite = tmp_path / "candidate-suite.json"
    baseline_suite = tmp_path / "baseline-suite.json"
    _universal_suite(candidate_suite, 3, {
        "msls_night": [1] * 10, "msls_day_val": [0] * 10, "svox_day_val": [0] * 10,
    })
    _universal_suite(baseline_suite, 3, {
        "msls_night": [0] * 10, "msls_day_val": [1] * 10, "svox_day_val": [1] * 10,
    })
    output = compare_universal_loss_selection(
        candidate_validation, baseline_validation, candidate_suite, baseline_suite,
        tmp_path / "selection.json", samples=200,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["candidate_selected"] is True
    assert payload["selection_datasets"] == ["robotcar_val", "msls_val"]
    assert "msls_day_val" not in payload["validation"]
    assert "svox_day_val" not in payload["validation"]
