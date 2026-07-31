import pytest
import torch

from a2b_lightvpr.generator.networks import PatchDiscriminator, ResnetGenerator
from a2b_lightvpr.generator.objectives import build_content_objective
from a2b_lightvpr.generator.trainer import _paired_charbonnier_loss


@pytest.mark.parametrize("name", ["negcut", "qsattn"])
def test_content_objective_produces_scalar_gradient(name):
    torch.manual_seed(0)
    generator = ResnetGenerator(channels=8, blocks=2)
    source = torch.randn(2, 3, 64, 64)
    fake, source_features = generator(source, return_features=True)
    _, fake_features = generator.encode_features(fake)
    if name == "negcut":
        # Small generator feature channels differ from the production contract.
        generator = ResnetGenerator()
        source = torch.randn(1, 3, 32, 32)
        fake, source_features = generator(source, return_features=True)
        _, fake_features = generator.encode_features(fake)
    objective = build_content_objective(
        name, num_patches=4, feature_dim=16
    )
    loss = objective.generator_loss(source, fake, source_features, fake_features)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in generator.parameters())


def test_qsattn_identity_nce_is_finite_and_differentiable():
    torch.manual_seed(2)
    generator = ResnetGenerator(channels=8, blocks=2)
    target = torch.randn(2, 3, 64, 64)
    identity, target_features = generator(target, return_features=True)
    _, identity_features = generator.encode_features(identity)
    objective = build_content_objective("qsattn", num_patches=4, feature_dim=16)
    loss = objective.identity_loss(target_features, identity_features)
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in generator.parameters())


def test_patch_discriminator_is_spatial():
    output = PatchDiscriminator(channels=8)(torch.randn(2, 3, 64, 64))
    assert output.shape[0] == 2
    assert output.shape[-1] > 1


def test_paired_charbonnier_only_updates_masked_samples():
    fake = torch.zeros(3, 3, 4, 4, requires_grad=True)
    target = torch.ones_like(fake)
    loss = _paired_charbonnier_loss(fake, target, torch.tensor([True, False, True]))
    loss.backward()
    assert fake.grad[0].abs().sum() > 0
    assert fake.grad[1].abs().sum() == 0
    assert fake.grad[2].abs().sum() > 0


def test_paired_charbonnier_empty_mask_is_differentiable_zero():
    fake = torch.randn(2, 3, 4, 4, requires_grad=True)
    loss = _paired_charbonnier_loss(
        fake, torch.randn_like(fake), torch.tensor([False, False])
    )
    assert loss.item() == 0.0
    loss.backward()
    assert torch.count_nonzero(fake.grad) == 0
