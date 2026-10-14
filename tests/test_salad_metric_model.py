import copy
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from a2b_lightvpr.vpr.model import build_descriptor_model
from a2b_lightvpr.vpr.salad import CompactSALADHead, _optimal_transport


class TinyDINO(nn.Module):
    embed_dim = 384

    def __init__(self, registers=False):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(1, 1) for _ in range(12))
        self.norm = nn.LayerNorm(384)
        if registers:
            self.register_tokens = nn.Parameter(torch.zeros(1, 4, 384))

    def forward_features(self, images):
        patches = F.avg_pool2d(images.mean(1, keepdim=True), 14, 14).flatten(2).transpose(1, 2)
        patches = patches.expand(-1, -1, 384)
        return {"x_norm_patchtokens": patches, "x_norm_clstoken": patches.mean(1)}


@pytest.fixture
def model_arguments(tmp_path, monkeypatch):
    repo = tmp_path / "dino"
    repo.mkdir()
    (repo / "hubconf.py").touch()
    plain, reg = tmp_path / "plain.pt", tmp_path / "reg.pt"
    torch.save(TinyDINO().state_dict(), plain)
    torch.save(TinyDINO(registers=True).state_dict(), reg)
    monkeypatch.setattr(torch.hub, "load", lambda repo, name, **kwargs: TinyDINO(name.endswith("_reg")))
    return dict(
        image_height=224, image_width=224, descriptor_dim=544,
        model_family="salad", backbone_name="dinov2_vits14",
        backbone_repo=str(repo), backbone_checkpoint=str(plain), backbone_trainable_blocks=2,
    ), reg


def test_salad_builder_strictly_initializes_backbone_and_head(model_arguments, tmp_path):
    arguments, _ = model_arguments
    torch.manual_seed(1)
    original = build_descriptor_model("salad", **arguments)
    checkpoint = tmp_path / "gsv.ckpt"
    torch.save({"state_dict": original.state_dict()}, checkpoint)
    torch.manual_seed(2)
    initialized = build_descriptor_model("salad", init_checkpoint=str(checkpoint), salad_dropout=0.3, **arguments)
    assert isinstance(initialized.aggregator.cluster_features[1], nn.Dropout)
    assert initialized.aggregator.cluster_features[1].p == 0.3
    for key, value in original.state_dict().items():
        assert torch.equal(value, initialized.state_dict()[key])
    original.eval()
    initialized.eval()
    images = torch.randn(2, 3, 224, 224)
    torch.testing.assert_close(original.encode(images), initialized.encode(images))


def test_salad_inference_builder_skips_missing_initial_weights(model_arguments, tmp_path):
    arguments, _ = model_arguments
    missing = str(tmp_path / "missing-training-init.ckpt")
    model = build_descriptor_model("salad", init_checkpoint=missing, load_initial_checkpoint=False, **arguments)
    assert model.descriptor_dim == 544
    with pytest.raises(FileNotFoundError):
        build_descriptor_model("salad", init_checkpoint=missing, **arguments)


def test_salad_rejects_wrong_checkpoint_backbone_or_incomplete_state(model_arguments, tmp_path):
    arguments, reg_checkpoint = model_arguments
    original = build_descriptor_model("salad", **arguments)
    checkpoint = tmp_path / "plain-gsv.ckpt"
    torch.save({"state_dict": original.state_dict()}, checkpoint)
    with pytest.raises(RuntimeError, match="register_tokens"):
        build_descriptor_model("salad", init_checkpoint=str(checkpoint), **{
            **arguments, "backbone_name": "dinov2_vits14_reg", "backbone_checkpoint": str(reg_checkpoint),
        })
    wrong_channels = original.state_dict()
    wrong_channels["aggregator.token_features.0.weight"] = torch.zeros(512, 768)
    torch.save(wrong_channels, checkpoint)
    with pytest.raises(RuntimeError, match="size mismatch"):
        build_descriptor_model("salad", init_checkpoint=str(checkpoint), **arguments)
    incomplete = original.state_dict()
    del incomplete["aggregator.score.3.weight"]
    torch.save(incomplete, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        build_descriptor_model("salad", init_checkpoint=str(checkpoint), **arguments)


@pytest.mark.parametrize("size", [(224, 224), (322, 322), (140, 42), (14, 238)])
def test_salad_supports_valid_patch_grids(model_arguments, size):
    arguments, _ = model_arguments
    model = build_descriptor_model("salad", **{**arguments, "image_height": size[0], "image_width": size[1]})
    result = model.encode(torch.randn(1, 3, *size))
    assert result.shape == (1, 544)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.norm(dim=1), torch.ones(1))


@pytest.mark.parametrize("size", [(0, 224), (-224, 224), (225, 224), (14, 224), (56, 56), (224.0, 224), (True, 224)])
def test_salad_rejects_invalid_patch_grids(model_arguments, size):
    arguments, _ = model_arguments
    with pytest.raises(ValueError, match="positive multiples"):
        build_descriptor_model("salad", **{**arguments, "image_height": size[0], "image_width": size[1]})


@pytest.mark.parametrize("shape", [(1, 3, 225, 224), (1, 3, 56, 56), (1, 1, 224, 224)])
def test_salad_rejects_invalid_runtime_images(model_arguments, shape):
    arguments, _ = model_arguments
    model = build_descriptor_model("salad", **arguments)
    with pytest.raises(ValueError, match="images require"):
        model.encode(torch.zeros(shape))


@pytest.mark.parametrize("changes,aggregation", [
    ({"descriptor_dim": 512}, "salad"),
    ({"backbone_name": "dinov2_vitb14"}, "salad"),
    ({}, "mixvpr"),
    ({"model_family": "stable_night_salad"}, "salad"),
])
def test_salad_keeps_descriptor_backbone_and_other_family_constraints(model_arguments, changes, aggregation):
    arguments, _ = model_arguments
    with pytest.raises(ValueError):
        build_descriptor_model(aggregation, **{**arguments, **changes})


def _repeat_aggregation(head, value):
    """Independent original SALAD aggregation expression, used as an oracle."""
    features, token = value
    local = head.cluster_features(features).flatten(2)
    scores = head.score(features).flatten(2)
    token = head.token_features(token)
    weights = _optimal_transport(scores, head.dust_bin).exp()[:, :-1]
    weights = weights[:, None].repeat(1, head.cluster_dim, 1, 1)
    local = local[:, :, None].repeat(1, 1, head.num_clusters, 1)
    output = torch.cat((F.normalize(token, dim=-1), F.normalize((weights * local).sum(-1), dim=1).flatten(1)), dim=-1)
    return F.normalize(output, dim=-1)


def test_einsum_matches_original_aggregation_and_all_gradients():
    torch.manual_seed(7)
    head = CompactSALADHead(24).double()
    oracle = copy.deepcopy(head)
    features = torch.randn(2, 24, 4, 5, dtype=torch.float64, requires_grad=True)
    token = torch.randn(2, 24, dtype=torch.float64, requires_grad=True)
    oracle_features, oracle_token = features.detach().clone().requires_grad_(), token.detach().clone().requires_grad_()
    output = head((features, token))
    expected = _repeat_aggregation(oracle, (oracle_features, oracle_token))
    torch.testing.assert_close(output, expected, rtol=1e-10, atol=1e-12)
    probe = torch.randn_like(output)
    (output * probe).sum().backward()
    (expected * probe).sum().backward()
    for actual, reference in ((features.grad, oracle_features.grad), (token.grad, oracle_token.grad)):
        torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-12)
    for actual, reference in zip(head.parameters(), oracle.parameters()):
        torch.testing.assert_close(actual.grad, reference.grad, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_amp_preserves_original_aggregation_precision(dropout):
    head = CompactSALADHead(24, dropout=dropout)
    features, token = torch.randn(2, 24, 4, 5), torch.randn(2, 24)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        torch.manual_seed(7)
        actual = head((features, token))
        torch.manual_seed(7)
        reference = _repeat_aggregation(head, (features, token))
    assert actual.dtype == reference.dtype == torch.float32
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("dropout", [-0.1, 1.1, float("nan"), float("inf")])
def test_salad_rejects_invalid_dropout(dropout):
    with pytest.raises(ValueError, match="dropout"):
        CompactSALADHead(24, dropout=dropout)


def test_dropout_keeps_checkpoint_keys_and_eval_descriptors():
    head, with_dropout = CompactSALADHead(24), CompactSALADHead(24, dropout=0.3)
    assert head.state_dict().keys() == with_dropout.state_dict().keys()
    with_dropout.load_state_dict(head.state_dict(), strict=True)
    head.eval()
    with_dropout.eval()
    inputs = torch.randn(2, 24, 4, 5), torch.randn(2, 24)
    torch.testing.assert_close(head(inputs), with_dropout(inputs), rtol=0, atol=0)


@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_einsum_matches_local_upstream_salad_forward_and_gradients(dropout):
    # Compare the actual cached upstream implementation without modifying it.
    # Upstream release entry: serizba/salad v1.0.0; inspected cached source SHA256:
    # 7c24560a1653a167aeea248f4566b445aa5a39dacc252f13adbac2f1a4f68726.
    path = Path.home() / ".cache/torch/hub/serizba_salad_main/models/aggregators/salad.py"
    if not path.is_file():
        pytest.skip("Local official SALAD source is unavailable")
    upstream = types.ModuleType("salad_aggregation_oracle")
    exec(compile(path.read_text(), str(path), "exec"), upstream.__dict__)
    reference = upstream.SALAD(num_channels=24, num_clusters=16, cluster_dim=32, token_dim=32, dropout=dropout)
    actual = CompactSALADHead(24, dropout=dropout)
    actual.load_state_dict(reference.state_dict(), strict=True)
    features, token = torch.randn(2, 24, 4, 5, requires_grad=True), torch.randn(2, 24, requires_grad=True)
    reference_features, reference_token = features.detach().clone().requires_grad_(), token.detach().clone().requires_grad_()
    torch.manual_seed(11)
    output = actual((features, token))
    torch.manual_seed(11)
    expected = reference((reference_features, reference_token))
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=1e-6)
    probe = torch.randn_like(output)
    (output * probe).sum().backward()
    (expected * probe).sum().backward()
    for gradient, oracle in ((features.grad, reference_features.grad), (token.grad, reference_token.grad)):
        torch.testing.assert_close(gradient, oracle, rtol=2e-4, atol=1e-6)
    for parameter, oracle in zip(actual.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, oracle.grad, rtol=2e-4, atol=1e-6)
