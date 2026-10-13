from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from a2b_lightvpr.config import DataConfig, ExperimentConfig
from a2b_lightvpr.checkpointing import file_sha256
from a2b_lightvpr.vpr.losses import (
    frozen_patch_reliability_target,
    multi_positive_place_loss,
    patch_reliability_loss,
    patch_validity_mask,
    stable_local_consistency_loss,
)
from a2b_lightvpr.vpr.stable_night import stable_night_batch_loss


def test_place_loss_uses_every_distinct_source_positive_and_backpropagates():
    values = torch.tensor([
        [1.0, 0.1], [0.7, 0.3], [0.1, 1.0], [0.3, 0.7],
        [0.8, 0.2], [0.6, 0.4], [0.2, 0.8], [0.4, 0.6],
    ], requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1] * 2)
    keys = torch.tensor([0, 1, 2, 3] * 2)
    loss, stats = multi_positive_place_loss(values, labels, keys, temperature=0.2)
    normalized = F.normalize(values.float(), dim=1)
    expected_terms = []
    for index in range(len(values)):
        candidates = keys.ne(keys[index])
        positives = candidates & labels.eq(labels[index])
        logits = normalized[index] @ normalized.T / 0.2
        expected_terms.append(-(logits[positives] - torch.logsumexp(logits[candidates], dim=0)).mean())
    assert torch.allclose(loss, torch.stack(expected_terms).mean())
    assert stats["place_valid_anchors"].item() == 8
    assert stats["place_positive_pairs"].item() == 16
    loss.backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert values.grad.abs().sum() > 0


@pytest.mark.parametrize("labels,keys", [
    ([0, 0, 1, 1], [7, 7, 8, 8]),  # Only same-source positive views.
    ([0, 0, 0, 0], [0, 1, 2, 3]),  # No different-place negatives.
    ([0, 1, 2, 3], [0, 1, 2, 3]),  # No same-place positives.
    ([0, 0, 0, 0], [7, 7, 7, 7]),  # Every candidate masked.
])
def test_place_loss_invalid_batches_return_differentiable_zero(labels, keys):
    values = torch.randn(4, 5, requires_grad=True)
    loss, stats = multi_positive_place_loss(values, torch.tensor(labels), torch.tensor(keys))
    assert loss.item() == 0.0 and stats["place_valid_anchors"].item() == 0
    loss.backward()
    assert values.grad is not None and values.grad.eq(0).all()


def _informative_reference():
    features = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]])
    texture = (torch.arange(8)[:, None] + torch.arange(8)[None, :]).remainder(2).float()
    rgb = texture.expand(1, 3, 8, 8)
    mean = rgb.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = rgb.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return features, (rgb - mean) / std


def test_frozen_target_detaches_teacher_and_masks_padding_and_low_information():
    features, images = _informative_reference()
    teacher = features.clone().requires_grad_()
    other_teacher = features.clone().requires_grad_()
    valid = torch.tensor([[[True, True], [True, False]]])
    target = frozen_patch_reliability_target(teacher, other_teacher, images, valid)
    assert not target.requires_grad
    assert target.shape == (1, 4)
    assert target[0, :3].gt(0.1).all() and target[0, 3].item() == 0.0
    uniform_features = torch.ones_like(features)
    featureless = frozen_patch_reliability_target(uniform_features, uniform_features, images, valid)
    assert featureless.eq(0).all()
    uniform_rgb = torch.zeros_like(images)
    textureless = frozen_patch_reliability_target(teacher, other_teacher, uniform_rgb, valid)
    assert textureless.eq(0).all()
    invalid = frozen_patch_reliability_target(teacher, other_teacher, images, torch.zeros_like(valid))
    assert invalid.eq(0).all()


def test_patch_mask_rejects_partial_padding():
    features = torch.randn(1, 3, 2, 2)
    pixels = torch.ones(1, 8, 8, dtype=torch.bool)
    pixels[:, 0, 0] = False
    valid = patch_validity_mask(pixels, features)
    assert valid.tolist() == [[False, True, True, True]]
    assert torch.equal(patch_validity_mask(valid.reshape(1, 2, 2), features), valid)


def test_stability_loss_uses_detached_target_and_excludes_invalid_patches():
    day = torch.randn(1, 3, 2, 2, requires_grad=True)
    aug = torch.randn(1, 3, 2, 2, requires_grad=True)
    target = torch.ones(1, 4, requires_grad=True)
    valid = torch.tensor([[True, False, False, False]])
    loss, _ = stable_local_consistency_loss(day, aug, target, valid)
    loss.backward()
    assert day.grad[:, :, 0, 0].abs().sum() > 0
    assert day.grad[:, :, 0, 1].eq(0).all()
    assert aug.grad[:, :, 1].eq(0).all()
    assert target.grad is None


def test_reliability_floor_is_per_image_and_excludes_padding():
    reliability = torch.tensor([[0.1, 0.1, 0.9], [0.8, 0.8, 0.0]], requires_grad=True)
    valid = torch.tensor([[True, True, False], [True, True, False]])
    loss, stats = patch_reliability_loss(reliability, valid, floor=0.25)
    assert loss.item() == pytest.approx(0.075)
    assert stats["reliability_valid_mean"].item() == pytest.approx(0.45)
    loss.backward()
    assert reliability.grad[0, :2].lt(0).all()
    assert reliability.grad[:, 2].eq(0).all()
    assert reliability.grad[1].eq(0).all()


def test_reliability_bce_detaches_target_and_empty_losses_keep_gradient_graph():
    reliability = torch.tensor([[0.5, 0.5]], requires_grad=True)
    target = torch.tensor([[0.9, 0.1]], requires_grad=True)
    loss, _ = patch_reliability_loss(reliability, torch.ones_like(target, dtype=torch.bool), target, floor=0.0)
    loss.backward()
    assert reliability.grad[0, 0] < 0 and reliability.grad[0, 1] > 0
    assert target.grad is None
    empty = torch.randn(1, 2, 2, 2, requires_grad=True)
    invalid = torch.zeros(1, 4, dtype=torch.bool)
    local_zero, _ = stable_local_consistency_loss(empty, empty + 0.1, torch.ones(1, 4), invalid)
    reliability_zero, _ = patch_reliability_loss(reliability, torch.zeros_like(target, dtype=torch.bool), target)
    assert (local_zero + reliability_zero).item() == 0.0
    (local_zero + reliability_zero).backward()
    assert empty.grad is not None and empty.grad.eq(0).all()


class TinyAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.descriptor = nn.Linear(4, 544)
        self.reliable = nn.Conv2d(4, 1, 1)

    def forward(self, features):
        descriptor = F.normalize(self.descriptor(features.mean(dim=(2, 3))), dim=1)
        reliability = self.reliable(features).flatten(1).sigmoid()
        return descriptor, {"reliability": reliability, "dustbin_assignment": 1.0 - reliability, "local_features": features}


class TinyStableModel(nn.Module):
    descriptor_dim = 544

    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 4, 1)
        self.aggregator = TinyAggregator()

    def encode_with_aux(self, images):
        return self.aggregator(F.adaptive_avg_pool2d(self.backbone(images), (2, 2)))

    def encode(self, images):
        return self.encode_with_aux(images)[0]

    def set_backbone_trainable(self, enabled):
        self.backbone.requires_grad_(enabled)


def _loss_config(**overrides):
    values = dict(
        stable_place_weight=1.0, stable_place_temperature=0.07,
        stable_stability_weight=0.0, stable_reliability_weight=0.01,
        stable_reliability_target="none", stable_reliability_floor=0.25,
    )
    return SimpleNamespace(**(values | overrides))


def test_batch_helper_never_uses_fake_for_local_reference_and_keeps_teacher_frozen():
    model = TinyStableModel()

    class Reference(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1)
            self.inputs = []

        def forward(self, images):
            self.inputs.append(images.detach().clone())
            features = F.adaptive_avg_pool2d(self.conv(images), (2, 2))
            return features, features.mean(dim=(2, 3))

    reference = Reference()
    day = torch.randn(4, 3, 8, 8)
    images = {"day": day, "aug": day * 0.7, "fake": day + 9.0}
    masks = {name: torch.ones(4, 2, 2, dtype=torch.bool) for name in images}
    loss, stats, count = stable_night_batch_loss(
        model, images, masks, torch.tensor([0, 0, 1, 1]), torch.arange(4),
        _loss_config(stable_stability_weight=0.1, stable_reliability_target="augmentation"), reference,
    )
    loss.backward()
    assert count == 8 and len(reference.inputs) == 2
    assert torch.equal(reference.inputs[0], day) and torch.equal(reference.inputs[1], images["aug"])
    assert all(parameter.grad is None for parameter in reference.parameters())
    assert not reference.training
    assert stats["fake_reliability_bce"].item() == 0.0
    assert model.aggregator.reliable.weight.grad is not None


@pytest.mark.parametrize("mode", ["day_only", "day_aug"])
def test_cpu_stable_training_epoch_preserves_checkpoint_and_optimizer_contract(tmp_path, monkeypatch, mode):
    from a2b_lightvpr.vpr import trainer

    source_manifest = tmp_path / "source.csv"
    source_manifest.write_text("source manifest", encoding="utf-8")
    robot_manifest = tmp_path / "robot.csv"
    robot_manifest.write_text("robot manifest", encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    backbone_checkpoint = tmp_path / "backbone.pt"
    backbone_checkpoint.write_bytes(b"frozen reference checkpoint")
    config = ExperimentConfig(DataConfig(
        str(robot_manifest), str(protocol), vpr_source_manifest=str(source_manifest),
        generated_root=str(tmp_path / "no_generated_data"),
    ))
    config = replace(config, vpr=replace(
        config.vpr, output_dir=str(tmp_path / "run"), model_family="stable_night_salad",
        aggregator="salad", descriptor_dim=544, train_domains=mode, epochs=1,
        batch_size=4, workers=0, samples_per_epoch=4, sampler="pk_balanced",
        pk_places_per_batch=2, pk_images_per_place=2, datasets=("robotcar",),
        dataset_epoch_quotas={"robotcar": 4}, selection_rule="fixed_budget_last",
        early_stopping=False, amp=False,
        backbone_checkpoint=str(backbone_checkpoint),
        stable_reliability_target="augmentation" if mode == "day_aug" else "none",
    ))

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, manifest, transform, **kwargs):
            assert manifest == source_manifest
            assert kwargs["train_domains"] == mode
            self.rows = [{"dataset": "robotcar", "global_place_id": f"p{index // 2}", "image_id": f"i{index}"} for index in range(4)]
            self.place_to_label = {"p0": 0, "p1": 1}

        def __len__(self):
            return 4

        def __getitem__(self, index):
            sample = {
                "day": torch.randn(3, 8, 8, generator=torch.Generator().manual_seed(index)),
                "day_valid_mask": torch.ones(2, 2, dtype=torch.bool),
                "label": torch.tensor(index // 2), "image_key": torch.tensor(index),
                "dataset": "robotcar",
            }
            if mode == "day_aug":
                sample["aug"] = sample["day"] * 0.7
            return sample

    reference_models = []

    class Reference(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            assert kwargs["trainable_blocks"] == 0
            self.conv = nn.Conv2d(3, 4, 1)
            reference_models.append(self)

        def forward(self, images):
            features = F.adaptive_avg_pool2d(self.conv(images), (2, 2))
            return features, features.mean(dim=(2, 3))

    def build_model(*args, **kwargs):
        assert kwargs["stable_sinkhorn_iterations"] == config.vpr.stable_sinkhorn_iterations
        return TinyStableModel()

    monkeypatch.setattr(trainer, "StableNightVPRDataset", Dataset)
    monkeypatch.setattr(trainer, "DINOv2SALADBackbone", Reference)
    monkeypatch.setattr(trainer, "build_descriptor_model", build_model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = trainer.train_vpr(config)
    state = torch.load(path, weights_only=True)
    assert state["kind"] == "vpr_inference" and state["descriptor_dim"] == 544
    training_path = tmp_path / "run" / "checkpoints" / "latest.pt"
    training_state = torch.load(training_path, weights_only=False)
    assert training_state["classifier"] == {}
    assert len(training_state["optimizer"]["param_groups"]) == 3
    assert training_state["optimizer"]["param_groups"][2]["params"] == []
    assert not any("reference" in name for name in training_state["model"])
    assert not any("reference" in name for name in state["model"])
    assert "generated_manifest" not in training_state["data_fingerprints"]
    metrics = json.loads((tmp_path / "run" / "training_metrics.jsonl").read_text())
    assert metrics["optimizer_steps"] == 1 and metrics["student_images"] == (8 if mode == "day_aug" else 4)
    assert metrics["reference_images"] == (8 if mode == "day_aug" else 0)
    assert metrics["components_mean"]["place_valid_anchors"] == (8 if mode == "day_aug" else 4)
    assert metrics["validation_recall1"] is None
    if mode == "day_aug":
        assert training_state["teacher_cache_fingerprint"] == "stable_reference:" + file_sha256(backbone_checkpoint)
        assert len(reference_models) == 1
        assert not reference_models[0].training
        assert not any(parameter.requires_grad for parameter in reference_models[0].parameters())
        assert all(parameter.grad is None for parameter in reference_models[0].parameters())
        optimizer_parameters = sum(len(group["params"]) for group in training_state["optimizer"]["param_groups"])
        assert optimizer_parameters == len(list(TinyStableModel().parameters()))
        resumed = replace(config, vpr=replace(config.vpr, resume=str(training_path)))
        backbone_checkpoint.write_bytes(b"changed frozen reference checkpoint")
        with pytest.raises(ValueError, match="frozen reference"):
            trainer.train_vpr(resumed)
    else:
        assert training_state["teacher_cache_fingerprint"] == ""
        assert reference_models == []
