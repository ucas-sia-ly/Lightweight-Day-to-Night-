from dataclasses import replace

import pytest
import torch

from a2b_lightvpr.config import DataConfig, ExperimentConfig
from a2b_lightvpr.vpr.trainer import (
    accumulation_divisor,
    build_vpr_optimizer,
    build_vpr_scheduler,
    configure_training_phase,
    freeze_batch_norm_statistics,
    learning_rates_at_minimum,
)
from a2b_lightvpr.vpr.losses import MarginCosineClassifier


def test_plateau_scheduler_reduces_and_respects_minimum():
    config = ExperimentConfig(DataConfig("robot.csv", "protocol.json"))
    cfg = replace(
        config.vpr,
        scheduler="plateau",
        scheduler_factor=0.2,
        scheduler_patience=0,
        scheduler_threshold=0.001,
        scheduler_min_lr_ratio=0.01,
    )
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1e-3)
    scheduler = build_vpr_scheduler(optimizer, cfg, 10)
    scheduler.step(0.5)
    scheduler.step(0.5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)
    for _ in range(10):
        scheduler.step(0.5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


def test_plateau_scheduler_state_resumes():
    config = ExperimentConfig(DataConfig("robot.csv", "protocol.json"))
    cfg = replace(config.vpr, scheduler="plateau", scheduler_patience=0)
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1e-3)
    scheduler = build_vpr_scheduler(optimizer, cfg, 10)
    scheduler.step(0.7)
    state = scheduler.state_dict()
    resumed = build_vpr_scheduler(optimizer, cfg, 10)
    resumed.load_state_dict(state)
    assert resumed.state_dict()["best"] == pytest.approx(0.7)


def test_convergence_requires_all_learning_rates_at_minimum():
    config = ExperimentConfig(DataConfig("robot.csv", "protocol.json"))
    cfg = replace(
        config.vpr,
        encoder_lr=1e-4,
        aggregator_lr=1e-3,
        classifier_lr=1e-3,
        scheduler_min_lr_ratio=0.01,
    )
    parameters = [torch.nn.Parameter(torch.ones(())) for _ in range(3)]
    optimizer = torch.optim.SGD([
        {"params": [parameters[0]], "lr": 1e-6},
        {"params": [parameters[1]], "lr": 1e-5},
        {"params": [parameters[2]], "lr": 2e-5},
    ])
    assert not learning_rates_at_minimum(optimizer, cfg)
    optimizer.param_groups[2]["lr"] = 1e-5
    assert learning_rates_at_minimum(optimizer, cfg)


def test_partial_gradient_accumulation_uses_actual_group_size():
    assert [accumulation_divisor(index, 5, 2) for index in range(5)] == [2, 2, 2, 2, 1]


def test_optimizer_uses_independent_learning_rates():
    config = ExperimentConfig(DataConfig("robot.csv", "protocol.json"))
    cfg = replace(
        config.vpr,
        encoder_lr=1e-6,
        aggregator_lr=2e-6,
        classifier_lr=3e-4,
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.aggregator = torch.nn.Linear(2, 2)

    model = Model()
    classifier = MarginCosineClassifier(2, 3)
    optimizer = build_vpr_optimizer(model, classifier, cfg)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1e-6, 2e-6, 3e-4]
    )


def test_official_warmup_only_trains_classifier_then_unfreezes():
    config = ExperimentConfig(DataConfig("robot.csv", "protocol.json"))
    cfg = replace(
        config.vpr,
        model_family="cosplace",
        classifier_warmup_epochs=1,
        classifier_warmup_lr=1e-3,
        encoder_lr=1e-6,
        aggregator_lr=1e-6,
        classifier_lr=1e-4,
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.aggregator = torch.nn.Linear(2, 2)

    model = Model()
    classifier = MarginCosineClassifier(2, 3)
    optimizer = build_vpr_optimizer(model, classifier, cfg)
    assert configure_training_phase(model, optimizer, cfg, 0)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0, 0, 1e-3])
    assert not configure_training_phase(model, optimizer, cfg, 1)
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1e-6, 1e-6, 1e-4]
    )


def test_frozen_batch_norm_keeps_running_statistics():
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(3), torch.nn.Conv2d(3, 3, 1))
    model.train()
    freeze_batch_norm_statistics(model)
    before_mean = model[0].running_mean.clone()
    before_var = model[0].running_var.clone()
    model(torch.randn(4, 3, 8, 8)).mean().backward()
    assert torch.equal(model[0].running_mean, before_mean)
    assert torch.equal(model[0].running_var, before_var)
    assert model[0].weight.grad is not None


def test_cosine_scheduler_reaches_configured_ratio_after_main_budget():
    config = ExperimentConfig(DataConfig("robot.csv", "protocol.json"))
    cfg = replace(
        config.vpr,
        epochs=3,
        classifier_warmup_epochs=1,
        scheduler="cosine",
        scheduler_min_lr_ratio=0.01,
    )
    parameters = [torch.nn.Parameter(torch.ones(())) for _ in range(3)]
    optimizer = torch.optim.SGD([
        {"params": [parameters[0]], "lr": cfg.encoder_lr},
        {"params": [parameters[1]], "lr": cfg.aggregator_lr},
        {"params": [parameters[2]], "lr": cfg.classifier_lr},
    ])
    scheduler = build_vpr_scheduler(optimizer, cfg, steps_per_epoch=2)
    for _ in range(4):
        optimizer.step()
        scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([
        cfg.encoder_lr * 0.01,
        cfg.aggregator_lr * 0.01,
        cfg.classifier_lr * 0.01,
    ])
