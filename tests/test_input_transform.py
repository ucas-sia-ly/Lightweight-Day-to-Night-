import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from a2b_lightvpr import runtime, trained_suite
from a2b_lightvpr.checkpointing import file_sha256
from a2b_lightvpr.config import DataConfig, ExperimentConfig, VPRConfig
from a2b_lightvpr.references import suite as reference_suite
from a2b_lightvpr.references.models import FixedResizeTransform, SuiteRow
from a2b_lightvpr.transforms import VPRTransform


def test_direct_resize_matches_reference_and_marks_all_patches_valid():
    image = Image.fromarray(np.arange(90 * 160 * 3, dtype=np.uint8).reshape(90, 160, 3))
    transform = VPRTransform(224, 224, resize_mode="resize")
    tensor, patches = transform.transform_with_mask(image)
    _, pixels = transform.transform_with_mask(image, patch_size=None)
    assert torch.equal(tensor, FixedResizeTransform(224, 224)(image))
    assert torch.equal(tensor, transform(image))
    assert patches.shape == (16, 16) and patches.all()
    assert pixels.shape == (224, 224) and pixels.all()
    assert transform.transform_kind == "direct-resize" and transform.padding == "none"
    default, default_mask = VPRTransform(224, 224).transform_with_mask(image)
    explicit = VPRTransform(224, 224, resize_mode="fit_pad")
    assert torch.equal(default, explicit(image))
    assert not default_mask.all()
    assert explicit.transform_kind == "fit-and-pad"
    with pytest.raises(ValueError, match="resize_mode"):
        VPRTransform(224, 224, resize_mode="crop")


def test_runtime_forwards_salad_dropout_without_loading_initial_checkpoint(tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1)
    checkpoint = tmp_path / "inference.pt"
    torch.save({"kind": "vpr_inference", "descriptor_dim": 544, "random_seed": 0,
                "model": model.state_dict()}, checkpoint)
    config = ExperimentConfig(DataConfig(), vpr=VPRConfig(
        model_family="salad", aggregator="salad", descriptor_dim=544,
        image_height=224, image_width=224, input_transform="resize", salad_dropout=0.3,
    ))
    arguments = []

    def builder(*args, **kwargs):
        arguments.append(kwargs)
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(runtime, "build_descriptor_model", builder)
    loaded, _, _ = runtime.load_vpr_model(config, checkpoint)
    assert arguments[0]["salad_dropout"] == 0.3
    assert arguments[0]["load_initial_checkpoint"] is False
    assert torch.equal(loaded.weight, model.weight)


def suite_fixture(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (28, 14), "white").save(image)
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "manifest.csv").write_text("image_id\nfixture\n")
    checkpoint = tmp_path / "trained.pt"
    checkpoint.write_bytes(b"trained fixture")
    full_init = tmp_path / "full_salad.ckpt"
    full_init.write_bytes(b"full SALAD initialization")
    backbone = tmp_path / "backbone.pth"
    backbone.write_bytes(b"foundation backbone initialization")
    config = ExperimentConfig(DataConfig(generated_root=str(generated)), vpr=VPRConfig(
        model_family="salad", aggregator="salad", descriptor_dim=544,
        image_height=224, image_width=224, input_transform="fit_pad",
        backbone_checkpoint=str(backbone), init_checkpoint=str(full_init),
        output_dir=str(tmp_path / "output"), workers=0,
    ))
    return config, checkpoint, SuiteRow(str(image), 0.0, 0.0)


class FixtureModel(torch.nn.Module):
    descriptor_dim = 544

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))


def test_trained_suite_transform_change_remeasures_and_reports_full_salad_init(tmp_path, monkeypatch):
    config, checkpoint, row = suite_fixture(tmp_path)
    calls = []
    monkeypatch.setattr(trained_suite, "load_vpr_model", lambda *args: (FixtureModel(), {"epoch": 2}, checkpoint))
    monkeypatch.setattr(trained_suite, "suite_rows", lambda *args: ([row], [row]))
    monkeypatch.setattr(trained_suite, "benchmark_model", lambda *args, **kwargs: {
        "parameters": 1, "peak_cuda_mib": None, "latency_ms": {"p50": 1.0, "p95": 1.0},
    })

    def evaluate(*args):
        calls.append(args[4])
        return {"recalls": {"1": 1.0, "5": 1.0, "10": 1.0}, "mrr": 1.0, "valid_queries": 1}

    monkeypatch.setattr(trained_suite, "evaluate_dual_encoder", evaluate)
    output = trained_suite.evaluate_trained_suite(config, checkpoint, ("svox_night",))
    first = json.loads(output.read_text())
    assert first["initialization"] == config.vpr.init_checkpoint
    assert first["initial_checkpoint_sha256"] == file_sha256(config.vpr.init_checkpoint)
    assert first["datasets"]["svox_night"]["padding"] == "imagenet-mean-centered"
    trained_suite.evaluate_trained_suite(config, checkpoint, ("svox_night",))
    assert len(calls) == 1
    resized = replace(config, vpr=replace(config.vpr, input_transform="resize"))
    trained_suite.evaluate_trained_suite(resized, checkpoint, ("svox_night",))
    second = json.loads(output.read_text())
    assert len(calls) == 2 and calls[-1].resize_mode == "resize"
    assert second["input_transform"] == "resize"
    assert second["datasets"]["svox_night"]["padding"] == "none"
    assert first["datasets"]["svox_night"]["evaluation_fingerprint"] != second["datasets"]["svox_night"]["evaluation_fingerprint"]


def test_metric_source_only_suite_needs_no_generated_manifest(tmp_path, monkeypatch):
    config, checkpoint, row = suite_fixture(tmp_path)
    source = tmp_path / "source.csv"
    source.write_text("image_id\nsource fixture\n")
    config = replace(config, data=replace(config.data, vpr_source_manifest=str(source),
                                         generated_root=str(tmp_path / "missing_generated")),
                     vpr=replace(config.vpr, training_objective="multi_similarity",
                                 train_domains="day_aug", salad_dropout=0.3,
                                 input_transform="resize"))
    monkeypatch.setattr(trained_suite, "load_vpr_model", lambda *args: (FixtureModel(), {"epoch": 2}, checkpoint))
    monkeypatch.setattr(trained_suite, "suite_rows", lambda *args: ([row], [row]))
    monkeypatch.setattr(trained_suite, "benchmark_model", lambda *args, **kwargs: {
        "parameters": 1, "peak_cuda_mib": None, "latency_ms": {"p50": 1.0, "p95": 1.0},
    })
    monkeypatch.setattr(trained_suite, "evaluate_dual_encoder", lambda *args: {
        "recalls": {"1": 1.0, "5": 1.0, "10": 1.0}, "mrr": 1.0, "valid_queries": 1,
    })
    output = trained_suite.evaluate_trained_suite(config, checkpoint, ("svox_night",))
    payload = json.loads(output.read_text())
    assert payload["manifest_sha256"] == file_sha256(source)
    assert not (tmp_path / "missing_generated").exists()
    assert payload["training"]["training_objective"] == "multi_similarity"
    assert payload["training"]["salad_dropout"] == 0.3
    assert payload["training"]["metric_augmentation_probability"] == config.vpr.metric_augmentation_probability
    assert payload["training"]["metric_within_city"] is True


def test_reference_unified_uses_config_transform_and_native_stays_resize(tmp_path, monkeypatch):
    config, checkpoint, row = suite_fixture(tmp_path)
    registry = tmp_path / "registry.yaml"
    registry.write_text(f"reference_models:\n  fixture:\n    loader: fixture\n    descriptor_dim: 544\n    native_size: [14, 14]\n    checkpoint: {checkpoint}\n")
    monkeypatch.setattr(reference_suite, "audit_reference_models", lambda *args: {"models": {"fixture": {"ready": True}}})
    monkeypatch.setattr(reference_suite, "build_reference_model", lambda *args: FixtureModel())
    monkeypatch.setattr(reference_suite, "suite_rows", lambda *args: ([row], [row]))
    calls = []

    def evaluate(*args):
        calls.append(args[4])
        return {"recalls": {"1": 1.0, "5": 1.0, "10": 1.0}, "mrr": 1.0, "valid_queries": 1}

    monkeypatch.setattr(reference_suite, "evaluate_dual_encoder", evaluate)
    arguments = dict(model_name="fixture", database_encoder="adapted", datasets=("svox_night",))
    output = reference_suite.evaluate_reference_suite(config, registry, output_dir=tmp_path / "unified", **arguments)
    first = json.loads(output.read_text())
    resized = replace(config, vpr=replace(config.vpr, input_transform="resize"))
    reference_suite.evaluate_reference_suite(resized, registry, output_dir=tmp_path / "unified", **arguments)
    second = json.loads(output.read_text())
    assert len(calls) == 2 and calls[-1].resize_mode == "resize"
    assert second["input_transform"] == "resize"
    assert second["datasets"]["svox_night"]["transform_kind"] == "direct-resize"
    assert first["datasets"]["svox_night"]["evaluation_fingerprint"] != second["datasets"]["svox_night"]["evaluation_fingerprint"]
    native = reference_suite.evaluate_reference_suite(config, registry, transform_name="native", output_dir=tmp_path / "native", **arguments)
    assert isinstance(calls[-1], FixedResizeTransform)
    reference_suite.evaluate_reference_suite(resized, registry, transform_name="native", output_dir=tmp_path / "native", **arguments)
    assert len(calls) == 3
    assert json.loads(native.read_text())["input_transform"] == "resize"
