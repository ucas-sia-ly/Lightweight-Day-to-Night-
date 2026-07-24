import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from a2b_lightvpr.config import DataConfig, ExperimentConfig
from a2b_lightvpr.vpr import teacher_cache as cache_module
from a2b_lightvpr.vpr.teacher_cache import TeacherDescriptorCache, cache_relation_teacher


class FakeTeacher(torch.nn.Module):
    output_dim = 3

    def forward(self, images):
        return torch.nn.functional.normalize(images.mean(dim=(2, 3)), dim=1)


def make_config(tmp_path):
    generated = tmp_path / "generated"
    generated.mkdir()
    image = tmp_path / "image.png"
    Image.new("RGB", (16, 12), color=(10, 20, 30)).save(image)
    manifest = generated / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "dataset", "place_id", "source_path", "generated_path", "image_id"],
        )
        writer.writeheader()
        for index in range(2):
            writer.writerow(
                {
                    "split": "train",
                    "dataset": "test",
                    "place_id": "place",
                    "source_path": image,
                    "generated_path": image,
                    "image_id": f"image-{index}",
                }
            )
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    base = ExperimentConfig(DataConfig("robot.csv", "protocol.json", generated_root=str(generated)))
    vpr = replace(
        base.vpr,
        loss_profile="v3",
        relation_teacher="salad",
        relation_teacher_checkpoint=str(checkpoint),
        relation_teacher_repo=str(tmp_path),
        relation_teacher_cache=str(tmp_path / "cache.npy"),
        relation_teacher_input_size=14,
        batch_size=2,
        workers=0,
        amp=False,
    )
    return replace(base, vpr=vpr), manifest


def test_teacher_cache_builds_and_resumes(monkeypatch, tmp_path):
    config, _ = make_config(tmp_path)
    monkeypatch.setattr(cache_module, "build_relation_teacher", lambda *args: FakeTeacher())
    path = cache_relation_teacher(config)
    metadata_path = path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["complete"] is True
    assert np.load(path, mmap_mode="r").shape == (2, 3)

    metadata.update(completed_rows=1, complete=False)
    metadata_path.write_text(json.dumps(metadata))
    cache_relation_teacher(config)
    assert json.loads(metadata_path.read_text())["completed_rows"] == 2


def test_training_cache_checks_ids_and_completion(monkeypatch, tmp_path):
    config, manifest = make_config(tmp_path)
    monkeypatch.setattr(cache_module, "build_relation_teacher", lambda *args: FakeTeacher())
    path = cache_relation_teacher(config)
    cache = TeacherDescriptorCache(config, manifest)
    assert cache.descriptor(0, "image-0").shape == (3,)
    with pytest.raises(RuntimeError, match="image ID mismatch"):
        cache.descriptor(0, "wrong")

    metadata_path = path.with_suffix(".meta.json")
    metadata = json.loads(metadata_path.read_text())
    metadata["complete"] = False
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="incomplete"):
        TeacherDescriptorCache(config, manifest)
