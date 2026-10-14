import io
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from a2b_lightvpr.references.models import (
    FixedResizeTransform,
    SuiteRow,
    _strict_state,
    _msls_rows,
    _svox_rows,
    _tokyo_rows,
    audit_reference_models,
    benchmark_reference_models,
    evaluate_dual_encoder,
    fetch_reference_model,
    validate_reference_resources,
)
from a2b_lightvpr.config import DataConfig, ExperimentConfig
from a2b_lightvpr.data import read_generated_manifest
from a2b_lightvpr.paths import DATASETS_ROOT, PROJECT_ROOT
from a2b_lightvpr.transforms import VPRTransform


DATA_ROOT = DATASETS_ROOT


@pytest.mark.parametrize("incomplete_core_namespace", [False, True])
def test_strict_state_loads_training_checkpoint_with_numpy_rng(tmp_path, monkeypatch, incomplete_core_namespace):
    import numpy as np
    from types import SimpleNamespace

    source = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "training.pt"
    torch.save(
        {"model": source.state_dict(), "rng": {"numpy": np.random.get_state()}},
        checkpoint,
    )
    target = torch.nn.Linear(3, 2)
    if incomplete_core_namespace:
        # Reproduce NumPy 1.26's partially imported numpy._core package.
        monkeypatch.setattr(np, "_core", SimpleNamespace(), raising=False)

    _strict_state(target, checkpoint)

    for expected, actual in zip(source.parameters(), target.parameters()):
        assert torch.equal(expected, actual)


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


@pytest.mark.skipif(not (DATA_ROOT / "RobotCar-Seasons").is_dir(), reason="local public Seasons unavailable")
def test_public_seasons_protocol_golden_counts():
    from a2b_lightvpr.references.models import suite_rows
    config = ExperimentConfig(DataConfig(robotcar_seasons_root=str(DATA_ROOT / "RobotCar-Seasons")))
    assert tuple(map(len, suite_rows(config, "robotcar_seasons_night"))) == (6954, 429)
    assert tuple(map(len, suite_rows(config, "robotcar_seasons_day"))) == (6954, 1443)


def test_day_validation_queries_do_not_overlap_vpr_training_manifest():
    manifest = Path("data/manifests_clean_night/vpr_train.csv")
    if not manifest.is_file():
        pytest.skip("VPR training manifest unavailable")
    if not any((DATA_ROOT / name).is_dir() for name in ("MSLS", "svox")):
        pytest.skip("Local MSLS/SVOX validation datasets unavailable")
    training = {str(Path(row["source_path"]).resolve()) for row in read_generated_manifest(manifest)}
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


def test_reference_resource_validation_checks_all_artifact_hashes(tmp_path):
    checkpoint = tmp_path / "model.pt"
    vocabulary = tmp_path / "vocabulary.pt"
    checkpoint.write_bytes(b"model")
    vocabulary.write_bytes(b"vocabulary")
    registry = tmp_path / "models.yaml"
    registry.write_text(
        "reference_models:\n"
        "  small:\n"
        "    loader: fixture\n"
        "    descriptor_dim: 4\n"
        f"    checkpoint: {checkpoint}\n"
        f"    required_files: [{vocabulary}]\n",
        encoding="utf-8",
    )
    from a2b_lightvpr.checkpointing import file_sha256
    resources = tmp_path / "resources.json"
    resources.write_text(json.dumps({"models": {"small": {
        "checkpoint_sha256": file_sha256(checkpoint),
        "required_artifacts": {
            str(vocabulary.resolve()): {"sha256": file_sha256(vocabulary)}
        },
    }}}), encoding="utf-8")
    assert validate_reference_resources(registry, resources, ("small",))
    vocabulary.write_bytes(b"changed")
    assert not validate_reference_resources(registry, resources, ("small",))


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

    monkeypatch.setattr("a2b_lightvpr.references.models.build_reference_model", lambda spec: Model())
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    config = ExperimentConfig(DataConfig())
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
