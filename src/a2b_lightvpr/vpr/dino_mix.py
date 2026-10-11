"""DINOv2-S patch grid + existing MixVPR head (not an official DINO-Mix checkpoint)."""
import torch
from torch import nn
from torch.nn import functional as F
from .salad import DINOv2SALADBackbone


class DINOMixStudent(nn.Module):
    descriptor_dim = 512
    aggregator_name = 'mixvpr'

    def __init__(self, repo, name, checkpoint, trainable_blocks=2):
        super().__init__()
        from .model import MixVPR
        self.backbone = DINOv2SALADBackbone(repo, name, checkpoint, trainable_blocks)
        if self.backbone.num_channels != 384:
            raise ValueError('DINO-Mix student requires 384-D Small backbone')
        self.aggregator = MixVPR(384, (23, 23), 512)

    def set_backbone_trainable(self, enabled):
        self.backbone.set_trainable(enabled)

    def encode(self, images):
        if tuple(images.shape[-2:]) != (322, 322):
            raise ValueError('DINO-Mix student requires 322x322 input')
        patches, _ = self.backbone(images)
        return F.normalize(self.aggregator(patches), dim=1)

    def forward(self, images):
        return self.encode(images)
