import json
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from PIL import Image
from torch import nn

from a2b_lightvpr.references.coverage import audit_reference_coverage
from a2b_lightvpr.references import pca as reference_pca
from a2b_lightvpr.references.pca import (
    _assert_no_evaluation_overlap,
    _cache_descriptors,
    _stratified_indices,
)
from a2b_lightvpr.references.models import FixedResizeTransform
from a2b_lightvpr.vpr.boq import PCAProjectedReference


class FakeReference(nn.Module):
    descriptor_dim = 4

    def encode(self, images):
        return torch.nn.functional.normalize(images[:, :4], dim=1)


def test_stratified_pca_selection_is_balanced_and_reproducible():
    rows = [
        {"dataset": dataset, "source_path": f"{dataset}-{index}.jpg"}
        for dataset in ("a", "b", "c")
        for index in range(20)
    ]
    first = _stratified_indices(rows, 30, 7)
    second = _stratified_indices(rows, 30, 7)
    assert first == second
    assert {rows[index]["dataset"] for index in first} == {"a", "b", "c"}
    assert {
        dataset: sum(rows[index]["dataset"] == dataset for index in first)
        for dataset in ("a", "b", "c")
    } == {"a": 10, "b": 10, "c": 10}


def test_pca_projection_centers_projects_and_normalizes(tmp_path):
    path = tmp_path / "pca.pt"
    torch.save({
        "kind": "reference_pca",
        "input_dim": 4,
        "output_dim": 2,
        "mean": torch.zeros(4),
        "components": torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]),
        "metadata": {"whiten": False},
    }, path)
    model = PCAProjectedReference(FakeReference(), path, 2)
    output = model.encode(torch.tensor([[3.0, 4.0, 0.0, 0.0]]))
    assert output.shape == (1, 2)
    assert torch.allclose(output, torch.tensor([[0.6, 0.8]]), atol=1e-6)


def test_reference_coverage_reports_missing_and_complete_models(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "reference_models:\n"
        "  ready-model:\n"
        "    loader: unsupported\n"
        "    descriptor_dim: 4\n"
        f"    checkpoint: {checkpoint}\n",
        encoding="utf-8",
    )
    reference_root = tmp_path / "reference"
    summary = reference_root / "ready-model/final_suite/native/adapted/summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}", encoding="utf-8")
    resources = tmp_path / "resources.json"
    resources.write_text(json.dumps({"models": {"ready-model": {}}}), encoding="utf-8")
    output = tmp_path / "coverage.json"
    audit_reference_coverage(registry, reference_root, [resources], output)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["complete"]
    assert report["models"]["ready-model"]["complete"]


def test_pca_fitting_rejects_any_evaluation_overlap(tmp_path, monkeypatch):
    shared = tmp_path / "shared.jpg"
    shared.write_bytes(b"image")
    monkeypatch.setattr(
        reference_pca,
        "suite_rows",
        lambda *args: ([SimpleNamespace(source_path=str(shared))], []),
    )
    with pytest.raises(ValueError, match="overlaps evaluation data"):
        _assert_no_evaluation_overlap(
            object(), [{"source_path": str(shared), "dataset": "gsv"}]
        )


def test_pca_descriptor_cache_resumes_from_completed_rows(tmp_path):
    rows = []
    for index, color in enumerate((32, 96, 192)):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (8, 8), (color, 0, 0)).save(path)
        rows.append(SimpleNamespace(source_path=str(path)))

    class MeanReference(torch.nn.Module):
        descriptor_dim = 2

        def encode(self, images):
            value = images.mean(dim=(2, 3))[:, :2]
            return torch.nn.functional.normalize(value + 1, dim=1)

    output = tmp_path / "cache.npy"
    descriptors = np.lib.format.open_memmap(output, mode="w+", dtype=np.float16, shape=(3, 2))
    descriptors[0] = np.asarray([7.0, 8.0], dtype=np.float16)
    descriptors.flush()
    identity = {"source": "fixture"}
    output.with_suffix(".meta.json").write_text(
        json.dumps({"identity": identity, "completed_rows": 1, "complete": False}),
        encoding="utf-8",
    )
    resumed = _cache_descriptors(
        MeanReference(), rows, FixedResizeTransform(8, 8), torch.device("cpu"),
        1, 0, output, identity,
    )
    assert resumed.shape == (3, 2)
    assert resumed[0].tolist() == [7.0, 8.0]
    metadata = json.loads(output.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert metadata["complete"]
    assert metadata["completed_rows"] == 3
