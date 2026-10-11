import torch
from torch import nn

from a2b_lightvpr.config import load_config
from a2b_lightvpr.vpr.salad import CompactSALADHead, DINOv2SALADBackbone


class FakeDINO(nn.Module):
    embed_dim = 384

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Linear(1, 1) for _ in range(12))
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward_features(self, images):
        batch, _, height, width = images.shape
        patches = (height // 14) * (width // 14)
        return {
            "x_norm_patchtokens": images.new_zeros(batch, patches, self.embed_dim),
            "x_norm_clstoken": images.new_zeros(batch, self.embed_dim),
        }


def test_compact_salad_head_has_544d_contract():
    head = CompactSALADHead(384)
    descriptor = head((torch.randn(2, 384, 4, 5), torch.randn(2, 384)))
    assert descriptor.shape == (2, 544)
    assert torch.allclose(descriptor.norm(dim=1), torch.ones(2), atol=1e-5)


def test_salad_backbone_preserves_only_last_two_trainable_blocks(tmp_path, monkeypatch):
    checkpoint = tmp_path / "dino.pt"
    torch.save(FakeDINO().state_dict(), checkpoint)
    repo = tmp_path / "dinov2"
    repo.mkdir()
    (repo / "hubconf.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: FakeDINO())
    backbone = DINOv2SALADBackbone(
        str(repo), "dinov2_vits14_reg", str(checkpoint), trainable_blocks=2
    )
    trainable = [
        index
        for index, block in enumerate(backbone.model.blocks)
        if any(parameter.requires_grad for parameter in block.parameters())
    ]
    assert trainable == [10, 11]
    features, token = backbone(torch.randn(1, 3, 322, 322))
    assert features.shape == (1, 384, 23, 23)
    assert token.shape == (1, 384)
    backbone.set_trainable(False)
    assert not any(parameter.requires_grad for parameter in backbone.parameters())


def test_salad_configs_are_strictly_paired():
    baseline = load_config("configs/experiments/students/salad_small_day_fake_544.yaml")
    candidate = load_config("configs/experiments/students/salad_small_cdnd_544.yaml")
    assert baseline.vpr.model_family == candidate.vpr.model_family == "salad"
    assert baseline.vpr.descriptor_dim == candidate.vpr.descriptor_dim == 544
    assert baseline.vpr.backbone_name == candidate.vpr.backbone_name == "dinov2_vits14_reg"
    assert baseline.vpr.backbone_trainable_blocks == candidate.vpr.backbone_trainable_blocks == 2
    assert baseline.vpr.cross_domain_aux == "none"
    assert candidate.vpr.cross_domain_aux == "cross_domain_relation"
