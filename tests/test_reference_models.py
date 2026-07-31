import csv
import io
from pathlib import Path

import pytest
import torch
from PIL import Image

from a2b_lightvpr.reference_models import (
    FixedResizeTransform,
    SuiteRow,
    _msls_rows,
    _robotcar_day_rows,
    _svox_rows,
    _tokyo_rows,
    audit_reference_models,
    benchmark_reference_models,
    evaluate_dual_encoder,
    fetch_reference_model,
)
from a2b_lightvpr.config import DataConfig, ExperimentConfig
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


@pytest.mark.skipif(not (DATA_ROOT / "svox").is_dir(), reason="local SVOX unavailable")
def test_local_svox_day_protocol_golden_counts():
    assert tuple(map(len, _svox_rows(DATA_ROOT / "svox", "val", "day"))) == (17226, 14698)
    assert tuple(map(len, _svox_rows(DATA_ROOT / "svox", "test", "day"))) == (17166, 14278)
    assert tuple(map(len, _svox_rows(DATA_ROOT / "svox", "test", "sun"))) == (17166, 854)


@pytest.mark.skipif(not (DATA_ROOT / "MSLS").is_dir(), reason="local MSLS unavailable")
def test_local_msls_day_protocol_golden_counts():
    assert tuple(map(len, _msls_rows(DATA_ROOT / "MSLS", "day"))) == (18886, 11065)


@pytest.mark.skipif(not (DATA_ROOT / "tokyo247").is_dir(), reason="local Tokyo unavailable")
def test_local_tokyo_day_protocol_golden_counts():
    assert tuple(map(len, _tokyo_rows(DATA_ROOT / "tokyo247", "tokyo_day"))) == (75984, 375)


def test_robotcar_day_uses_fixed_untrained_traversals():
    manifest = DATA_ROOT / "dvg_benchmark_paper/robotcar_nightday/manifest.csv"
    if not manifest.is_file():
        pytest.skip("local RobotCar unavailable")
    database, queries = _robotcar_day_rows(manifest)
    assert {row.traverse for row in database} == {"2014-07-14-14-49-50"}
    assert {row.traverse for row in queries} == {"2015-08-12-15-04-18"}
    assert all(row.split != "train" for row in database + queries)


def test_day_validation_queries_do_not_overlap_vpr_training_manifest():
    manifest = Path("data/manifests_clean_night/vpr_train.csv")
    if not manifest.is_file():
        pytest.skip("VPR training manifest unavailable")
    with manifest.open(newline="", encoding="utf-8") as handle:
        training = {str(Path(row["source_path"]).resolve()) for row in csv.DictReader(handle)}
    day_queries = []
    if (DATA_ROOT / "MSLS").is_dir():
        day_queries.extend(_msls_rows(DATA_ROOT / "MSLS", "day")[1])
    if (DATA_ROOT / "svox").is_dir():
        day_queries.extend(_svox_rows(DATA_ROOT / "svox", "val", "day")[1])
    assert day_queries
    assert not training.intersection(row.source_path for row in day_queries)


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


def test_fetch_reference_model_is_explicit_and_records_hash(tmp_path, monkeypatch):
    checkpoint = tmp_path / "pitts.pth.tar"
    registry = tmp_path / "models.yaml"
    registry.write_text(f"""reference_models:
  netvlad:
    loader: netvlad_pytorch
    descriptor_dim: 4096
    checkpoint: {checkpoint}
    download_url: https://huggingface.co/TobiasRobotics/Patch-NetVLAD/resolve/main/pitts_WPCA4096.pth.tar
""", encoding="utf-8")

    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": str(len(b"strict-checkpoint"))}

        def getcode(self):
            return self.status

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request: Response(b"strict-checkpoint")
    )
    result = fetch_reference_model(registry, "netvlad")
    assert result["size_bytes"] == len(b"strict-checkpoint")
    assert len(result["sha256"]) == 64
    assert checkpoint.with_suffix(checkpoint.suffix + ".json").is_file()


def test_fetch_reference_model_resumes_partial_download(tmp_path, monkeypatch):
    checkpoint = tmp_path / "pitts.pth.tar"
    partial = checkpoint.with_suffix(checkpoint.suffix + ".part")
    partial.write_bytes(b"strict-")
    registry = tmp_path / "models.yaml"
    registry.write_text(f"""reference_models:
  netvlad:
    loader: netvlad_pytorch
    descriptor_dim: 4096
    checkpoint: {checkpoint}
    download_url: https://huggingface.co/TobiasRobotics/Patch-NetVLAD/resolve/main/pitts_WPCA4096.pth.tar
""", encoding="utf-8")

    class Response(io.BytesIO):
        status = 206
        headers = {
            "Content-Length": str(len(b"checkpoint")),
            "Content-Range": "bytes 7-16/17",
        }

        def getcode(self):
            return self.status

    def open_request(request):
        assert request.get_header("Range") == "bytes=7-"
        return Response(b"checkpoint")

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    result = fetch_reference_model(registry, "netvlad")
    assert checkpoint.read_bytes() == b"strict-checkpoint"
    assert result["size_bytes"] == 17


def test_reference_benchmark_reports_storage_and_batch_dependence(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    registry = tmp_path / "models.yaml"
    registry.write_text(f"""reference_models:
  fixture:
    loader: fixture
    descriptor_dim: 8
    native_size: [8, 8]
    batch_dependent: true
    checkpoint: {checkpoint}
""", encoding="utf-8")

    class Model(torch.nn.Module):
        descriptor_dim = 8

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(8))

        def encode(self, images):
            value = images.mean((1, 2, 3)).unsqueeze(1) * self.weight
            return torch.nn.functional.normalize(value + 1.0, dim=1)

    monkeypatch.setattr("a2b_lightvpr.reference_models.build_reference_model", lambda spec: Model())
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    config = ExperimentConfig(DataConfig("robot.csv", "protocol.json"))
    output = benchmark_reference_models(
        config, registry, ("fixture",), warmup=1, iterations=2,
        output=tmp_path / "resources.json",
    )
    payload = __import__("json").loads(output.read_text(encoding="utf-8"))
    metrics = payload["models"]["fixture"]
    assert metrics["database_descriptor_memory"]["1000000"]["fp32_gib"] == pytest.approx(
        32_000_000 / 1024**3
    )
    assert metrics["batch_dependence"] is not None
