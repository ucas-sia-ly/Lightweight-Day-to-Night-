import torch

from a2b_lightvpr.generator.networks import PatchDiscriminator, ResnetGenerator
from a2b_lightvpr.generator.objectives import build_content_objective


def test_qsattn_content_objective_produces_scalar_gradient():
    torch.manual_seed(0)
    generator = ResnetGenerator(channels=8, blocks=2)
    source = torch.randn(2, 3, 64, 64)
    fake, source_features = generator(source, return_features=True)
    _, fake_features = generator.encode_features(fake)
    objective = build_content_objective("qsattn", num_patches=4, feature_dim=16)
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
