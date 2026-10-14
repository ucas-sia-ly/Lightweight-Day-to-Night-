import argparse

import pytest

from a2b_lightvpr import cli


def test_cli_exposes_current_training_and_public_evaluation_workflow():
    parser = cli._parser()
    commands = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    assert set(commands.choices) == {
        "audit-data", "audit-resources", "audit-robotcar-seasons", "prepare-gsv-metric",
        "train-vpr", "export-vpr-inference", "export-vpr-initial", "evaluate-trained-suite",
        "benchmark", "audit-reference-models", "fetch-reference-model",
        "benchmark-reference-models", "validate-reference-resources", "audit-reference-coverage",
        "evaluate-reference-suite",
    }
    args = parser.parse_args([
        "evaluate-trained-suite", "--config", "model.yaml", "--checkpoint", "model.pt",
        "--datasets", "tokyo247_night", "robotcar_seasons_night",
    ])
    assert args.datasets == ["tokyo247_night", "robotcar_seasons_night"]


def test_training_cli_passes_explicit_overrides_and_smoke_budget(monkeypatch, capsys):
    config = object()
    monkeypatch.setattr(cli, "load_config", lambda path: config if path == "metric.yaml" else None)
    observed = {}

    def overrides(value, values):
        assert value is config
        observed["overrides"] = values
        return value

    def train(value, *, smoke_epochs):
        assert value is config
        observed["epochs"] = smoke_epochs
        return "runs/current/model_best.pt"

    monkeypatch.setattr(cli, "apply_overrides", overrides)
    monkeypatch.setattr(cli, "train_vpr", train)
    cli.main(["train-vpr", "--config", "metric.yaml", "--set", "vpr.workers=0", "--smoke-epochs", "1"])
    assert observed == {"overrides": ["vpr.workers=0"], "epochs": 1}
    assert capsys.readouterr().out.strip() == "runs/current/model_best.pt"


@pytest.mark.parametrize("command", [
    "prepare-data", "train-generator", "translate-train", "cache-vpr-teacher",
    "summarize-cdnd-seed0", "summarize-cdnd-multiseed", "summarize-paired-vpr",
    "plot-paper-efficiency", "export-qualitative-top1", "train-boq-small",
    "fit-anyloc-small-vocabulary", "fit-reference-pca", "summarize-small-references",
])
def test_retired_workflows_have_no_cli_entry(command):
    with pytest.raises(SystemExit) as exc:
        cli._parser().parse_args([command])
    assert exc.value.code == 2


@pytest.mark.parametrize('omp,mkl,expected', [(None, None, 2), ('4', None, 4), (None, '3', 3)])
def test_cli_limits_cpu_threads_and_respects_explicit_settings(monkeypatch, omp, mkl, expected):
    from a2b_lightvpr import cli
    for name, value in [('OMP_NUM_THREADS', omp), ('MKL_NUM_THREADS', mkl)]:
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    configured = []
    monkeypatch.setattr(cli.torch, 'set_num_threads', configured.append)
    monkeypatch.setattr(cli.torch, 'get_num_threads', lambda: expected)
    monkeypatch.setattr(cli, 'train_vpr', lambda *args, **kwargs: 'mock-model.pt')
    cli.main(['train-vpr', '--config', 'configs/experiments/metric/salad_small_gsv_day_aug_544.yaml'])
    assert configured == [expected]
