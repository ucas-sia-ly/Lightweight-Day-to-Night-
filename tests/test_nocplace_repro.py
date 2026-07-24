from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from a2b_lightvpr.config import DataConfig, ExperimentConfig, load_config
from a2b_lightvpr.nocplace_repro import (
    _NocPlaceDataset,
    _route_balanced,
    build_negcut_train_command,
    load_official_negcut_generator,
    nocplace_reproduction_loss,
    PlaceBalancedSampler,
)
from a2b_lightvpr.vpr.losses import MarginCosineClassifier, ikt_loss


def test_nocplace_repro_config_is_paper_protocol_with_substitute_data():
    config = load_config("configs/nocplace_repro_clean5k.yaml")
    repro = config.nocplace_repro
    assert repro.samples_per_dataset_domain == 500
    assert (repro.load_size, repro.crop_size) == (640, 512)
    assert (repro.negcut_epochs, repro.negcut_decay_epochs) == (200, 200)
    assert (repro.vpr_iterations, repro.vpr_batch_size) == (10_000, 32)
    assert repro.translation_batch_size == 4
    assert (repro.ikt_temperature, repro.ikt_weight) == (4.0, 30.0)
    assert repro.model_lr == pytest.approx(1e-6)
    assert repro.classifier_lr == pytest.approx(1e-2)


def test_route_balanced_selection_is_unique_and_deterministic(tmp_path):
    rows = []
    for route in ("a", "b", "c"):
        for index in range(4):
            path = tmp_path / f"{route}_{index}.jpg"
            path.write_bytes(b"x")
            rows.append({"source_path": str(path), "route": route})
    first = _route_balanced(rows, 9, 7)
    second = _route_balanced(rows, 9, 7)
    assert [row["source_path"] for row in first] == [row["source_path"] for row in second]
    assert len({row["source_path"] for row in first}) == 9
    assert {row["route"] for row in first[:3]} == {"a", "b", "c"}


def test_official_negcut_command_is_exact(tmp_path):
    repo = tmp_path / "NEGCUT"
    repo.mkdir()
    (repo / "train.py").write_text("", encoding="utf-8")
    root = tmp_path / "repro"
    (root / "negcut_5k" / "trainA").mkdir(parents=True)
    (root / "negcut_5k" / "trainB").mkdir()
    config = ExperimentConfig(DataConfig("r.csv", "p.json"))
    config = replace(config, nocplace_repro=replace(
        config.nocplace_repro, negcut_repo=str(repo), data_root=str(root),
        negcut_checkpoints_dir=str(tmp_path / "checkpoints"),
    ))
    command = build_negcut_train_command(config)
    joined = " ".join(command)
    for expected in (
        "--model negcut", "--NEGCUT_mode negcut", "--direction AtoB",
        "--netG resnet_9blocks", "--gan_mode lsgan", "--load_size 640",
        "--crop_size 512", "--batch_size 1", "--n_epochs 200",
        "--n_epochs_decay 200", "--save_epoch_freq 100",
    ):
        assert expected in joined
    assert "--serial_batches" not in command


def test_project_generator_checkpoint_is_rejected(tmp_path):
    checkpoint = tmp_path / "generator_final.pt"
    checkpoint.write_bytes(b"not official")
    config = ExperimentConfig(DataConfig("r.csv", "p.json"))
    with pytest.raises(ValueError, match="official NEG-CUT"):
        load_official_negcut_generator(config, checkpoint, torch.device("cpu"))


def test_nocplace_transform_is_deterministic_512_without_augmentation():
    image = Image.new("RGB", (731, 413), (20, 100, 200))
    first = _NocPlaceDataset.transform(image)
    second = _NocPlaceDataset.transform(image)
    assert first.shape == (3, 512, 512)
    assert torch.equal(first, second)


def test_place_balanced_sampler_round_robins_classes():
    dataset = object.__new__(_NocPlaceDataset)
    dataset.place_indices = {0: [0, 1], 1: [2], 2: [3, 4, 5]}
    samples = list(PlaceBalancedSampler(dataset, 9, seed=3))
    reverse = {index: place for place, indices in dataset.place_indices.items() for index in indices}
    assert sorted(reverse[index] for index in samples) == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_nocplace_loss_is_night_lmc_plus_30_ikt():
    torch.manual_seed(2)
    classifier = MarginCosineClassifier(8, 5)
    fake = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    teacher = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    labels = torch.tensor([0, 1, 2, 3])
    total, components = nocplace_reproduction_loss(fake, labels, teacher, classifier, 4.0, 30.0)
    expected_lmc = torch.nn.functional.cross_entropy(classifier(fake, labels), labels)
    expected_ikt = ikt_loss(teacher, fake, classifier, 4.0)
    assert torch.allclose(components["lmc"], expected_lmc)
    assert torch.allclose(components["ikt"], expected_ikt)
    assert torch.allclose(total, expected_lmc + 30.0 * expected_ikt)
    assert np.isfinite(float(total.detach()))
