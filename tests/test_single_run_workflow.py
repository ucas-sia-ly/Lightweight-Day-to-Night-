from pathlib import Path

from a2b_lightvpr.cli import _parser


def test_clustered_workflow_contract():
    script = Path("scripts/evaluate/run_robotcar_temporal_512.fish").read_text(encoding="utf-8")
    assert "select-robotcar-temporal" in script
    assert "--condition night" in script
    assert "configs/vpr/cdnd_salad_512.yaml" in script
    assert "cdnd_512/salad_no_gate_seed0/model_best.pt" in script
    assert "--dimension" not in script
    assert '--night "$night"' in script
    assert '--day "$day"' in script
    assert "Temporal Consensus Reranking" in script
    assert 'set selection "$output_dir/val_selection.json"' not in script
    assert 'set ablations "$output_dir/ablations.json"' not in script
    assert not Path("scripts/run_robotcar_temporal_4096.fish").exists()


def test_cli_rejects_removed_method():
    with __import__("pytest").raises(SystemExit):
        _parser().parse_args([
            "select-robotcar-temporal", "--config", "c.yaml", "--checkpoint", "m.pt",
            "--cache", "c.npz", "--output", "o.json", "--method", "mgtr-v1",
        ])


def test_cli_rejects_removed_gardens_commands():
    for command in (
        "evaluate-gardens-temporal",
        "evaluate-gardens-reference",
        "summarize-gardens-models",
    ):
        with __import__("pytest").raises(SystemExit):
            _parser().parse_args([command])
