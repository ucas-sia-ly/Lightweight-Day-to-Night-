import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from a2b_lightvpr.checkpointing import checkpoint_config_matches
from a2b_lightvpr.config import DataConfig, ExperimentConfig, VPRConfig, validate_config
from a2b_lightvpr.runtime import load_vpr_model
from a2b_lightvpr.vpr.model import build_descriptor_model
from a2b_lightvpr.vpr.salad import CompactSALADHead, ReliableSALADHead, _optimal_transport


def stable_config(**kwargs):
    cfg = VPRConfig(
        model_family="stable_night_salad", aggregator="salad", descriptor_dim=544,
        image_height=322, image_width=322, backbone_name="dinov2_vits14_reg",
        backbone_repo="dino", backbone_checkpoint="dino.pt", backbone_trainable_blocks=0,
        sampler="pk_balanced", batch_size=8, pk_places_per_batch=4, pk_images_per_place=2,
        dataset_epoch_quotas={"msls": 8}, samples_per_epoch=8, datasets=("msls",),
        train_domains="day_aug",
    )
    return ExperimentConfig(DataConfig(), vpr=replace(cfg, **kwargs))


def test_scalar_transport_matches_legacy_and_constant_dynamic_bins():
    torch.manual_seed(0)
    scores = torch.randn(2, 4, 20)
    alpha = torch.tensor(1.0)
    # Independently express the original SALAD transport, including its three
    # iterations, to protect the numerical baseline path.
    z = torch.cat((scores, alpha.expand(2, 1, 20)), dim=1)
    norm = -torch.tensor(24.0).log()
    mu = torch.cat((norm.expand(4), torch.tensor(16.0).log()[None] + norm))[None].expand(2, -1)
    nu = norm.expand(2, 20)
    u, v = torch.zeros_like(mu), torch.zeros_like(nu)
    for _ in range(3):
        u = mu - torch.logsumexp(z + v[:, None], dim=2)
        v = nu - torch.logsumexp(z + u[:, :, None], dim=1)
    expected = z + u[:, :, None] + v[:, None] - norm
    assert torch.equal(_optimal_transport(scores, alpha), expected)
    assert torch.equal(_optimal_transport(scores, alpha.expand(2, 20)), expected)


def test_dynamic_dustbin_increases_selected_patch_discard_without_changing_budget():
    scores = torch.zeros(1, 4, 20)
    original = _optimal_transport(scores, torch.tensor(1.), iterations=100).exp()
    bins = torch.ones(1, 20)
    bins[:, 0] += 4.0
    dynamic = _optimal_transport(scores, bins, iterations=100).exp()
    assert dynamic[0, -1, 0] > original[0, -1, 0]
    assert dynamic[0, :-1, 0].sum() < original[0, :-1, 0].sum()
    assert torch.allclose(dynamic[:, :-1].sum(-1), torch.ones(1, 4), atol=1e-5)
    assert torch.allclose(dynamic.sum(1), torch.ones(1, 20), atol=1e-5)
    assert torch.allclose(dynamic[:, -1].sum(-1), torch.tensor([16.]), atol=1e-5)


def test_reliable_head_contract_gradients_and_amp():
    torch.manual_seed(7)
    head = ReliableSALADHead(384)
    features = torch.randn(2, 384, 6, 6, requires_grad=True)
    token = torch.randn(2, 384, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        descriptor, aux = head((features, token), return_aux=True)
    assert descriptor.shape == (2, 544)
    assert descriptor.dtype == torch.float32
    assert torch.allclose(descriptor.norm(dim=1), torch.ones(2), atol=1e-5)
    assert aux["reliability"].shape == aux["dustbin_assignment"].shape == (2, 36)
    assert aux["reliability"].min() >= 0 and aux["reliability"].max() <= 1
    assert aux["dustbin_assignment"].min() >= 0 and aux["dustbin_assignment"].max() <= 1 + 1e-5
    assert torch.allclose(aux["cluster_mass"], torch.ones(2, 16), atol=1e-4)
    # Some CPU builds support BF16 forward but not DNNL BF16 backward. Verify
    # gradients in FP32 here and CUDA AMP separately below.
    descriptor, aux = head((features, token), return_aux=True)
    (descriptor * torch.randn_like(descriptor)).sum().backward()
    for gradient in (features.grad, token.grad, head.gamma_logit.grad,
                     head.reliability_predictor[-1].weight.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    assert 0 < head.gamma < head.gamma_max
    assert isinstance(head((features.detach(), token.detach())), torch.Tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AMP requires a GPU")
def test_reliable_head_cuda_amp_backward():
    head = ReliableSALADHead(384).cuda()
    features = torch.randn(2, 384, 6, 6, device="cuda", requires_grad=True)
    token = torch.randn(2, 384, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.float16):
        descriptor = head((features, token))
        loss = (descriptor * torch.randn_like(descriptor)).sum()
    loss.backward()
    assert descriptor.dtype == torch.float32 and torch.isfinite(descriptor).all()
    for parameter in head.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_extreme_transport_logits_are_finite_and_reject_invalid_shapes():
    scores = (torch.randn(2, 4, 20) * 10000).requires_grad_()
    bins = torch.randn(2, 20, requires_grad=True)
    result = _optimal_transport(scores, bins, iterations=20)
    assert torch.isfinite(result).all()
    result.exp().square().sum().backward()
    assert torch.isfinite(scores.grad).all() and torch.isfinite(bins.grad).all()
    with pytest.raises(ValueError, match="more patches"):
        _optimal_transport(torch.zeros(1, 4, 4), torch.tensor(1.))
    with pytest.raises(ValueError, match="Dustbin logits"):
        _optimal_transport(torch.zeros(1, 4, 20), torch.zeros(3, 2))


@pytest.mark.parametrize("changes", [
    {"stable_stability_weight": 1., "train_domains": "day_fake"},
    {"stable_reliability_target": "augmentation", "train_domains": "day_only"},
    {"stable_dustbin_gamma_init": 0.}, {"stable_sinkhorn_iterations": 0},
    {"stable_feature_fusion": "6_12"}, {"pk_images_per_place": 1},
    {"backbone_name": "dinov2_vitb14"}, {"classifier_warmup_epochs": 1},
    {"stable_place_temperature": float("nan")},
    {"backbone_trainable_blocks": 1.5}, {"pk_images_per_place": 2.5},
    {"stable_stability_weight": 1., "backbone_trainable_blocks": 0},
])
def test_invalid_prototype_controls_fail(changes):
    with pytest.raises(ValueError):
        validate_config(stable_config(**changes))


def test_frozen_backbone_config_and_legacy_checkpoint_defaults():
    validate_config(stable_config(backbone_trainable_blocks=0))
    legacy = ExperimentConfig(DataConfig())
    saved = legacy.as_dict()
    for name in list(saved["vpr"]):
        if name.startswith("stable_"):
            del saved["vpr"][name]
    del saved["evaluation"]["fake_database_manifest"]
    assert checkpoint_config_matches({"config": saved}, legacy)


class TinyDINO(nn.Module):
    embed_dim = 384

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(1, 1) for _ in range(12))
        self.norm = nn.LayerNorm(384)

    def forward_features(self, images):
        grid = torch.nn.functional.avg_pool2d(images.mean(1, keepdim=True), 14, 14)
        patches = grid.flatten(2).transpose(1, 2).expand(-1, -1, 384)
        return {"x_norm_patchtokens": patches, "x_norm_clstoken": patches.mean(1)}


def test_compact_initialization_and_strict_runtime_round_trip(tmp_path, monkeypatch):
    repo = tmp_path / "dino"
    repo.mkdir()
    (repo / "hubconf.py").touch()
    backbone = tmp_path / "backbone.pt"
    torch.save(TinyDINO().state_dict(), backbone)
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: TinyDINO())
    config = stable_config(backbone_repo=str(repo), backbone_checkpoint=str(backbone),
                           stable_sinkhorn_iterations=12, stable_dustbin_enabled=False)
    arguments = dict(image_height=322, image_width=322, descriptor_dim=544,
                     backbone_repo=str(repo), backbone_name="dinov2_vits14_reg",
                     backbone_checkpoint=str(backbone), backbone_trainable_blocks=0)
    compact = build_descriptor_model("salad", model_family="salad", **arguments)
    initial = tmp_path / "compact.pt"
    torch.save({"model": compact.state_dict()}, initial)
    stable = build_descriptor_model("salad", model_family="stable_night_salad",
                                    init_checkpoint=str(initial), stable_sinkhorn_iterations=12,
                                    stable_dustbin_enabled=False, **arguments)
    for key, value in compact.state_dict().items():
        assert torch.equal(value, stable.state_dict()[key])
    assert not any(p.requires_grad for p in stable.backbone.parameters())
    checkpoint = tmp_path / "inference.pt"
    torch.save({"kind": "vpr_inference", "descriptor_dim": 544, "random_seed": 0,
                "model": stable.state_dict()}, checkpoint)
    loaded, _, _ = load_vpr_model(config, checkpoint)
    assert not loaded.aggregator.dustbin_enabled
    assert loaded.aggregator.sinkhorn_iterations == 12
    images = torch.randn(1, 3, 322, 322)
    with torch.no_grad():
        assert torch.equal(stable.encode(images), loaded.encode(images))
    with pytest.raises(ValueError, match="head settings"):
        load_vpr_model(replace(config, vpr=replace(config.vpr, stable_dustbin_enabled=True)), checkpoint)


def test_mixed_training_with_real_dataset_head_checkpoint_and_runtime(tmp_path, monkeypatch):
    """Only replace external DINO loading, exercising the actual stage-one path."""
    from a2b_lightvpr.vpr import trainer

    class ImageDINO(TinyDINO):
        def forward_features(self, images):
            grid = torch.nn.functional.avg_pool2d(images, 14, 14)
            patches = grid.flatten(2).transpose(1, 2).repeat(1, 1, 128)
            return {"x_norm_patchtokens": patches, "x_norm_clstoken": patches.mean(1)}

    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: ImageDINO())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    repo = tmp_path / "dino"
    repo.mkdir()
    (repo / "hubconf.py").touch()
    checkpoint = tmp_path / "backbone.pt"
    torch.save(ImageDINO().state_dict(), checkpoint)
    generated = tmp_path / "generated"
    generated.mkdir()
    rows = []
    for index in range(4):
        source, fake = tmp_path / f"day{index}.png", tmp_path / f"fake{index}.png"
        pixels = np.random.default_rng(index).integers(0, 255, (28, 70, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(source)
        # Different aspect ratio deliberately rules out a Day/Fake local loss.
        Image.fromarray(pixels[:24] // 2).save(fake)
        rows.append(dict(dataset="msls", split="train", place_id=f"p{index // 2}",
                         image_id=f"i{index}", source_path=str(source), generated_path=str(fake)))
    for manifest in (tmp_path / "source.csv", generated / "manifest.csv"):
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    robot, protocol = tmp_path / "robot.csv", tmp_path / "protocol.json"
    robot.write_text("fixture", encoding="utf-8")
    protocol.write_text("{}", encoding="utf-8")
    config = stable_config(
        output_dir=str(tmp_path / "run"), backbone_repo=str(repo),
        backbone_checkpoint=str(checkpoint), train_domains="day_aug_fake", workers=0,
        epochs=1, batch_size=4, samples_per_epoch=4, dataset_epoch_quotas={"msls": 4},
        pk_places_per_batch=2, pk_images_per_place=2, min_class_images=2,
        selection_rule="fixed_budget_last", early_stopping=False, amp=False,
    )
    config = replace(config, data=DataConfig(
        vpr_source_manifest=str(tmp_path / "source.csv"),
        generated_root=str(generated),
    ))
    validate_config(config)
    inference_path = trainer.train_vpr(config)
    model, state, _ = load_vpr_model(config, inference_path)
    assert state["kind"] == "vpr_inference"
    assert "aggregator._extra_state" in state["model"]
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    metrics = json.loads((tmp_path / "run" / "training_metrics.jsonl").read_text())
    assert metrics["student_images"] == 12 and metrics["optimizer_steps"] == 1
    assert metrics["reference_images"] == 0
    assert metrics["components_mean"]["place_valid_anchors"] == 12
    with torch.no_grad():
        output = model.encode(torch.randn(1, 3, 322, 322))
    assert output.shape == (1, 544) and torch.isfinite(output).all()
