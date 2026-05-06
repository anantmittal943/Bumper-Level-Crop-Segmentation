"""Model definition: MobileNetV3 backbone with a lightweight decoder.

Designed to be fast for inference on cropped bottom-half inputs.
"""
from __future__ import annotations

from typing import Optional
import logging

import torch
import torch.nn as nn
import torchvision.models as models

from config import cfg

_LOG = logging.getLogger(__name__)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class LightweightDecoder(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        mid = max(64, in_channels // 2)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), ConvBNReLU(in_channels, mid))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), ConvBNReLU(mid, mid // 2))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), ConvBNReLU(mid // 2, mid // 4))
        self.classifier = nn.Conv2d(mid // 4, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.classifier(x)
        return x


class SegmentationModel(nn.Module):
    """MobileNetV3 encoder + lightweight decoder.

    Exposes methods to freeze/unfreeze backbone for fine-tuning.
    """

    def __init__(self, num_classes: int = cfg.NUM_CLASSES, backbone: str = cfg.BACKBONE, freeze_backbone: bool = cfg.FREEZE_BACKBONE):
        super().__init__()
        if backbone != "mobilenet_v3_large":
            _LOG.warning("Backbone %s requested but only mobilenet_v3_large is supported in this implementation.", backbone)
        # Load pretrained backbone
        bm = models.mobilenet_v3_large(pretrained=True)
        # Extract features up to final features layer
        self.encoder = bm.features
        # The last conv output channels (mobilenet_v3_large) is 960
        encoder_out_ch = 960
        self.decoder = LightweightDecoder(in_channels=encoder_out_ch, num_classes=num_classes)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder produces lower-resolution feature map; mobilenet reduces spatial dims
        feat = self.encoder(x)
        logits = self.decoder(feat)
        # logits shape: (N, C, H_out, W_out) - resize to input spatial size using bilinear
        logits = nn.functional.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits


def create_model(num_classes: int = cfg.NUM_CLASSES, freeze_backbone: bool = cfg.FREEZE_BACKBONE) -> SegmentationModel:
    model = SegmentationModel(num_classes=num_classes, freeze_backbone=freeze_backbone)
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = create_model()
    x = torch.randn(1, 3, cfg.RESIZE_HW[1], cfg.RESIZE_HW[0])
    y = m(x)
    print(y.shape)
