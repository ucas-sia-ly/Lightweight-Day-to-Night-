from pathlib import Path

import pytest
import torch
from PIL import Image

from a2b_lightvpr.reference_models import (
    FixedResizeTransform,
    SuiteRow,
    _msls_rows,
    _svox_rows,
    _tokyo_rows,
    audit_reference_models,
    evaluate_dual_encoder,
)
from a2b_lightvpr.transforms import VPRTransform


DATA_ROOT = Path("/home/admin123/diffusion_vpr_test/CycleGAN-turbo_NetVLAD/dataset")


def test_reference_registry_reports_ready_and_missing_models(tmp_path):
    registry = tmp_path / "models.yaml"
    registry.write_text(
        """reference_models:
  ready:
    loader: fixture
    descriptor_dim: 2
  missing:
    loader: fixture
    descriptor_dim: 2
    checkpoint: /definitely/missing.pt
    download_url: https://example.invalid/model.pt
  protocol_only:
    supported: false
    loader: fixture
    descriptor_dim: 2
""",
        encoding="utf-8",
    )
    result = audit_reference_models(registry)
    assert result["models"]["ready"]["ready"]
    assert not result["models"]["missing"]["ready"]
    assert result["models"]["missing"]["download_url"]
    assert not result["models"]["protocol_only"]["ready"]


@pytest.mark.skipif(not (DATA_ROOT / "svox").is_dir(), reason="local SVOX unavailable")
def test_local_svox_protocol_golden_counts():
    database, queries = _svox_rows(DATA_ROOT / "svox")
    assert (len(database), len(queries)) == (17_166, 823)


@pytest.mark.skipif(not (DATA_ROOT / "MSLS").is_dir(), reason="local MSLS unavailable")
def test_local_msls_night_protocol_golden_counts():
    database, queries = _msls_rows(DATA_ROOT / "MSLS")
    assert (len(database), len(queries)) == (18_886, 55)


@pytest.mark.skipif(not (DATA_ROOT / "tokyo247").is_dir(), reason="local Tokyo unavailable")
def test_local_tokyo_night_and_sunset_are_not_empty():
    night_database, night_queries = _tokyo_rows(DATA_ROOT / "tokyo247", "tokyo_night")
    sunset_database, sunset_queries = _tokyo_rows(DATA_ROOT / "tokyo247", "tokyo_sunset")
    assert len(night_database) == len(sunset_database) == 75_984
    assert len(night_queries) == len(sunset_queries) == 375


def test_native_and_unified_transforms_have_declared_shapes():
    image = Image.new("RGB", (91, 53), "white")
    assert FixedResizeTransform(224, 224)(image).shape == (3, 224, 224)
    assert VPRTransform(320, 480)(image).shape == (3, 320, 480)


def test_dual_encoder_uses_distinct_database_and_query_models(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (8, 8), (index * 255, 0, 0)).save(path)
        paths.append(path)

    class Encoder(torch.nn.Module):
        descriptor_dim = 2

        def __init__(self, output):
            super().__init__()
            self.register_buffer("output", torch.tensor(output, dtype=torch.float32))
            self.calls = 0

        def encode(self, images):
            self.calls += 1
            return self.output.expand(images.shape[0], -1)

    database_model = Encoder([1.0, 0.0])
    query_model = Encoder([1.0, 0.0])
    database = [SuiteRow(str(paths[0]), 0.0, 0.0)]
    queries = [SuiteRow(str(paths[1]), 0.0, 0.0)]
    metrics = evaluate_dual_encoder(
        database_model, query_model, database, queries,
        FixedResizeTransform(8, 8), 25.0, None, (1,),
        torch.device("cpu"), 1, 0,
    )
    assert database_model.calls == query_model.calls == 1
    assert metrics["recalls"]["1"] == 1.0
