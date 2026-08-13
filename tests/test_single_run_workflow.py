from pathlib import Path

import pytest

from a2b_lightvpr.cli import _parser


def test_cli_has_no_public_random_state_option():
    with pytest.raises(SystemExit):
        _parser().parse_args(["train-vpr", "--config", "config.yaml", "--" + "seed", "1"])


def test_single_run_script_has_resume_skip_budget_and_all_dimensions():
    script = Path("scripts/run_seed0_all.fish").read_text(encoding="utf-8")
    assert "1198720" in script
    assert "checkpoints/latest.pt" in script
    assert "summarize-cdnd-seed0" in script
    assert "python -m pytest" not in script
    for dimension in (128, 512, 4096):
        assert str(dimension) in script


def test_ragtr_workflow_is_val_gated_and_runs_test_once_after_freeze():
    script = Path("scripts/run_robotcar_ragtr_4096.fish").read_text(encoding="utf-8")
    assert "prepare_val" in script
    assert "salad_threshold" in script
    assert "mine-robotcar-route" in script
    assert "route_contrastive_seed0" in script
    assert "route_training_complete" in script
    assert 'int(x.get(\"samples_seen\",-1))==300000' in script
    assert script.count("--condition night") == 1
    assert script.index("Val-only fallback decision") < script.index("Frozen night test")
    assert "--method ragtr" in script
    assert "-m pytest -q -p no:cacheprovider" in script
    assert "tests/test_ragtr.py" not in script
    assert 'schema==\"robotcar_ragtr_evaluation_v1\"' in script
    assert 'required=(\"query_protocol_fingerprint\",)' in script
