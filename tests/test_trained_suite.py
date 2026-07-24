import json
from dataclasses import replace

import torch

from a2b_lightvpr.checkpointing import file_sha256
from a2b_lightvpr.config import DataConfig, ExperimentConfig
from a2b_lightvpr import trained_suite


def test_trained_suite_json_and_tracking_use_same_metrics(tmp_path, monkeypatch):
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "manifest.csv").write_text("image_id\nfixture\n", encoding="utf-8")
    checkpoint = tmp_path / "model_best.pt"
    checkpoint.write_bytes(b"fixture-checkpoint")
    config = ExperimentConfig(
        DataConfig("robot.csv", "protocol.json", generated_root=str(generated))
    )
    config = replace(
        config,
        vpr=replace(config.vpr, output_dir=str(tmp_path / "run"), workers=0),
    )

    class Model(torch.nn.Module):
        def encode(self, images):
            return torch.nn.functional.normalize(torch.ones(images.shape[0], 512), dim=1)

    captured = {}

    class Tracker:
        def __init__(self, *args, **kwargs):
            pass

        def summary(self, values):
            captured.update(values)

        def finish(self):
            pass

    monkeypatch.setattr(
        trained_suite, "load_vpr_model",
        lambda config, path: (Model(), {"epoch": 3}, checkpoint),
    )
    monkeypatch.setattr(trained_suite, "suite_rows", lambda *args: ([object()], [object()]))
    monkeypatch.setattr(
        trained_suite, "evaluate_dual_encoder",
        lambda *args, **kwargs: {
            "recalls": {"1": 0.75, "5": 1.0, "10": 1.0},
            "mrr": 0.8,
            "valid_queries": 1,
        },
    )
    def benchmark(*args, **kwargs):
        assert kwargs["track"] is False
        return {
            "parameters": 10,
            "peak_cuda_mib": None,
            "latency_ms": {"p50": 1.0, "p95": 2.0},
        }

    monkeypatch.setattr(trained_suite, "benchmark_model", benchmark)
    monkeypatch.setattr(trained_suite, "TrainingTracker", Tracker)

    output = trained_suite.evaluate_trained_suite(
        config, checkpoint, ("robotcar_project",)
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    metrics = payload["datasets"]["robotcar_project"]
    assert metrics["recalls"]["1"] == 0.75
    assert metrics["checkpoint_sha256"] == payload["checkpoint_sha256"]
    assert metrics["descriptor_dim"] == 512
    assert captured["test/robotcar_project/recall_at_1"] == 0.75
    assert captured["suite/checkpoint_sha256"] == payload["checkpoint_sha256"]


def test_trained_suite_reuses_matching_summary_for_tracking(tmp_path, monkeypatch):
    generated = tmp_path / "generated"
    generated.mkdir()
    manifest = generated / "manifest.csv"
    manifest.write_text("image_id\nfixture\n", encoding="utf-8")
    checkpoint = tmp_path / "model_best.pt"
    checkpoint.write_bytes(b"fixture-checkpoint")
    output_dir = tmp_path / "run"
    suite_dir = output_dir / "final_suite"
    suite_dir.mkdir(parents=True)
    config = ExperimentConfig(
        DataConfig("robot.csv", "protocol.json", generated_root=str(generated))
    )
    config = replace(config, vpr=replace(config.vpr, output_dir=str(output_dir)))
    payload = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest_sha256": file_sha256(manifest),
        "model_family": config.vpr.model_family,
        "backbone": config.vpr.backbone,
        "aggregator": config.vpr.aggregator,
        "descriptor_dim": 512,
        "initialization": "fixture",
        "input_size": [256, 384],
        "benchmark": {
            "parameters": 10,
            "peak_cuda_mib": 20.0,
            "latency_ms": {"p50": 1.0, "p95": 2.0},
        },
        "datasets": {
            "robotcar_project": {
                "recalls": {"1": 0.8, "5": 0.9, "10": 1.0},
                "mrr": 0.85,
                "valid_queries": 10,
                "protocol_fingerprint": "fixture-protocol",
            }
        },
    }
    (suite_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    captured = {}

    class Tracker:
        def __init__(self, *args, **kwargs):
            pass

        def summary(self, values):
            captured.update(values)

        def finish(self):
            pass

    monkeypatch.setattr(trained_suite, "TrainingTracker", Tracker)
    monkeypatch.setattr(trained_suite, "suite_rows", lambda *args: ([], []))
    monkeypatch.setattr(
        trained_suite, "protocol_fingerprint", lambda *args: "fixture-protocol"
    )
    monkeypatch.setattr(
        trained_suite,
        "load_vpr_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must use cache")),
    )
    result = trained_suite.evaluate_trained_suite(
        config, checkpoint, ("robotcar_project",)
    )
    assert result == suite_dir / "summary.json"
    assert captured["test/robotcar_project/recall_at_1"] == 0.8
    assert captured["suite/latency_p95_ms"] == 2.0
