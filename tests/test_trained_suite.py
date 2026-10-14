import json
from dataclasses import replace

import torch

from a2b_lightvpr import trained_suite
from a2b_lightvpr.checkpointing import file_sha256
from a2b_lightvpr.config import DataConfig, ExperimentConfig


def test_suite_aliases_cover_active_conditions():
    assert trained_suite.expand_suite_datasets(("all",)) == trained_suite.NIGHT_PROTOCOLS
    assert trained_suite.expand_suite_datasets(("all-day",)) == trained_suite.DAY_PROTOCOLS
    assert trained_suite.expand_suite_datasets(("all-conditions",)) == (
        trained_suite.NIGHT_PROTOCOLS + trained_suite.DAY_PROTOCOLS
    )


def _fixture_config(tmp_path):
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "manifest.csv").write_text("image_id\nfixture\n", encoding="utf-8")
    config = ExperimentConfig(
        DataConfig(generated_root=str(generated))
    )
    return replace(
        config, vpr=replace(config.vpr, output_dir=str(tmp_path / "run"), workers=0)
    )


def test_trained_suite_writes_protocol_bound_metrics(tmp_path, monkeypatch):
    config = _fixture_config(tmp_path)
    checkpoint = tmp_path / "model_best.pt"
    checkpoint.write_bytes(b"fixture-checkpoint")

    class Model(torch.nn.Module):
        descriptor_dim = 512

        def encode(self, images):
            return torch.ones(images.shape[0], 512)

    monkeypatch.setattr(
        trained_suite, "load_vpr_model",
        lambda *args: (Model(), {"epoch": 3}, checkpoint),
    )
    monkeypatch.setattr(trained_suite, "suite_rows", lambda *args: ([object()], [object()]))
    monkeypatch.setattr(
        trained_suite, "evaluate_dual_encoder",
        lambda *args, **kwargs: {
            "recalls": {"1": 0.75, "5": 1.0, "10": 1.0},
            "mrr": 0.8, "valid_queries": 1,
        },
    )
    monkeypatch.setattr(
        trained_suite, "benchmark_model",
        lambda *args, **kwargs: {
            "parameters": 10, "peak_cuda_mib": None,
            "latency_ms": {"p50": 1.0, "p95": 2.0},
        },
    )
    output = trained_suite.evaluate_trained_suite(
        config, checkpoint, ("svox_night",)
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    metrics = payload["datasets"]["svox_night"]
    assert metrics["recalls"]["1"] == 0.75
    assert metrics["checkpoint_sha256"] == file_sha256(checkpoint)
    assert len(metrics["evaluation_fingerprint"]) == 64


def test_trained_suite_reuses_matching_summary(tmp_path, monkeypatch):
    config = _fixture_config(tmp_path)
    checkpoint = tmp_path / "model_best.pt"
    checkpoint.write_bytes(b"fixture-checkpoint")
    suite_dir = tmp_path / "run" / "final_suite"
    suite_dir.mkdir(parents=True)
    payload = {
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest_sha256": file_sha256(tmp_path / "generated" / "manifest.csv"),
        "database_domain": "day", "database_manifest_sha256": {},
        "model_family": config.vpr.model_family, "encoder": config.vpr.model_family,
        "evaluation_recalls": list(config.evaluation.recalls),
        "aggregator": config.vpr.aggregator, "descriptor_dim": 512,
        "initialization": "fixture", "input_size": [256, 384],
        "benchmark": {"parameters": 10, "peak_cuda_mib": 20.0,
                      "latency_ms": {"p50": 1.0, "p95": 2.0}},
        "datasets": {"svox_night": {
            "recalls": {"1": 0.8, "5": 0.9, "10": 1.0},
            "mrr": 0.85, "valid_queries": 10,
            "query_protocol_fingerprint": "fixture",
            "evaluation_fingerprint": trained_suite.evaluation_fingerprint(
                "fixture", transform="fit-and-pad", height=256, width=384,
                padding="imagenet-mean-centered",
            ),
        }},
    }
    summary = suite_dir / "summary.json"
    summary.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(trained_suite, "suite_rows", lambda *args: ([], []))
    monkeypatch.setattr(
        trained_suite, "query_protocol_fingerprint", lambda *args: "fixture"
    )
    monkeypatch.setattr(
        trained_suite, "load_vpr_model",
        lambda *args: (_ for _ in ()).throw(AssertionError("must use cache")),
    )
    assert trained_suite.evaluate_trained_suite(
        config, checkpoint, ("svox_night",)
    ) == summary
