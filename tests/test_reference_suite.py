import json
from dataclasses import replace

import torch
from PIL import Image

from a2b_lightvpr.config import DataConfig, ExperimentConfig
from a2b_lightvpr.reference_models import SuiteRow
from a2b_lightvpr import reference_suite


def test_reference_suite_uses_shared_protocol_and_cache(tmp_path, monkeypatch):
    checkpoint = tmp_path / "query.pt"
    database_checkpoint = tmp_path / "database.pt"
    checkpoint.write_bytes(b"query")
    database_checkpoint.write_bytes(b"database")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        f"""reference_models:
  nocplace:
    loader: fixture
    descriptor_dim: 2
    native_size: [8, 8]
    checkpoint: {checkpoint}
    database_checkpoint: {database_checkpoint}
""",
        encoding="utf-8",
    )
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "manifest.csv").write_text("image_id\nfixture\n", encoding="utf-8")
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "white").save(image)
    config = ExperimentConfig(
        DataConfig("robot.csv", "protocol.json", generated_root=str(generated))
    )
    config = replace(
        config,
        vpr=replace(config.vpr, image_height=8, image_width=8, workers=0),
    )

    class Model(torch.nn.Module):
        descriptor_dim = 2

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    built = []
    monkeypatch.setattr(
        reference_suite,
        "audit_reference_models",
        lambda *args, **kwargs: {"models": {"nocplace": {"ready": True}}},
    )
    monkeypatch.setattr(
        reference_suite,
        "build_reference_model",
        lambda spec: built.append(spec["checkpoint"]) or Model(),
    )
    row = SuiteRow(str(image), 0.0, 0.0, 0.0)
    monkeypatch.setattr(reference_suite, "suite_rows", lambda *args: ([row], [row]))
    calls = []

    def evaluate(*args):
        calls.append((args[5], args[6]))
        return {
            "recalls": {"1": 1.0, "5": 1.0, "10": 1.0},
            "mrr": 1.0,
            "valid_queries": 1,
        }

    monkeypatch.setattr(reference_suite, "evaluate_dual_encoder", evaluate)

    output_dir = tmp_path / "output"
    output = reference_suite.evaluate_reference_suite(
        config,
        registry,
        model_name="nocplace",
        database_encoder="original",
        transform_name="unified",
        datasets=("robotcar_project",),
        output_dir=output_dir,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert built == [str(checkpoint), str(database_checkpoint)]
    assert calls == [(10.0, 30.0)]
    metrics = payload["datasets"]["robotcar_project"]
    assert len(metrics["query_protocol_fingerprint"]) == 64
    assert len(metrics["evaluation_fingerprint"]) == 64
    assert metrics["protocol_fingerprint"] == metrics["query_protocol_fingerprint"]

    monkeypatch.setattr(
        reference_suite,
        "build_reference_model",
        lambda *args: (_ for _ in ()).throw(AssertionError("must use cache")),
    )
    assert reference_suite.evaluate_reference_suite(
        config,
        registry,
        model_name="nocplace",
        database_encoder="original",
        transform_name="unified",
        datasets=("robotcar_project",),
        output_dir=output_dir,
    ) == output
