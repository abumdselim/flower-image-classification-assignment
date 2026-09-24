"""
models.py — The five models compared in this project.

1) SimpleCNN        — small CNN designed & trained FROM SCRATCH (baseline)
2) VGG16            — ImageNet-pretrained, frozen conv base (Simonyan & Zisserman 2014)
3) ResNet50         — ImageNet-pretrained, frozen conv base (He et al. 2016)
4) MobileNetV2      — ImageNet-pretrained, frozen conv base (Sandler et al. 2018)
5) EfficientNet-B0  — ImageNet-pretrained, frozen conv base (Tan & Le 2019)

Transfer-learning strategy (two-stage, CPU friendly):
  Stage 1: frozen backbone acts as a fixed feature extractor; features are
           pre-computed once (original + augmented passes).
  Stage 2: a small trainable MLP head (Dropout-Linear-ReLU-Dropout-Linear)
           is trained on those features under 5-fold cross-validation.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm

from config import CNN_INPUT_SIZE, HEAD_HIDDEN

# ------------------------------------------------------- 1) SimpleCNN ----
class SimpleCNN(nn.Module):
    """
    Compact from-scratch CNN baseline (3 conv blocks + classifier).

    Block: Conv3x3 -> BatchNorm -> ReLU -> MaxPool2x2  (repeated 3x)
      128 -> 64 -> 32 -> 16 spatial, channels 32/64/128
    Head: GlobalAveragePool -> FC(128->128) -> ReLU -> Dropout(0.3) -> FC(128->C)
    """

    def __init__(self, num_classes: int, in_size: int = CNN_INPUT_SIZE):
        super().__init__()
        channels = (32, 64, 128)
        blocks, in_ch = [], 3
        for c in channels:
            blocks += [nn.Conv2d(in_ch, c, kernel_size=3, padding=1),
                       nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            in_ch = c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], HEAD_HIDDEN), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(HEAD_HIDDEN, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# ------------------------------------- 2-5) frozen pretrained backbones ----
def build_backbone(name: str):
    """
    Return (frozen_backbone_module, feature_dim_after_GAP).
    The backbone is set to eval() mode so BatchNorm uses its ImageNet
    running statistics, and every parameter requires_grad=False.
    """
    name = name.lower()
    if name == "vgg16":
        net = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
        backbone = net.features                      # -> 512 x 7 x 7
        feat_dim = 512
    elif name == "resnet50":
        net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        backbone = nn.Sequential(*list(net.children())[:-1])  # -> 2048 x 1 x 1
        feat_dim = 2048
    elif name == "mobilenetv2":
        net = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
        backbone = net.features                      # -> 1280 x 7 x 7
        feat_dim = 1280
    elif name == "efficientnet-b0":
        net = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        backbone = net.features                      # -> 1280 x 7 x 7
        feat_dim = 1280
    else:
        raise ValueError(f"unknown backbone: {name}")

    for p in backbone.parameters():                  # freeze everything
        p.requires_grad = False
    backbone.eval()
    return backbone, feat_dim


def build_head(in_dim: int, num_classes: int) -> nn.Module:
    """Small trainable classifier head used on top of every frozen backbone."""
    return nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_dim, HEAD_HIDDEN), nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(HEAD_HIDDEN, num_classes),
    )


# ------------------------------------------------------- model registry ----
MODEL_REGISTRY = {
    "SimpleCNN": {
        "type": "scratch",
        "input_size": CNN_INPUT_SIZE,
        "desc": "3-block CNN (32-64-128) trained from scratch on the custom dataset",
    },
    "VGG16": {
        "type": "transfer",
        "input_size": 224,
        "desc": "13 conv layers of uniform 3x3 filters (~14.7M conv params), ImageNet-pretrained, frozen",
    },
    "ResNet50": {
        "type": "transfer",
        "input_size": 224,
        "desc": "bottleneck residual blocks with skip connections (~23.5M params), ImageNet-pretrained, frozen",
    },
    "MobileNetV2": {
        "type": "transfer",
        "input_size": 224,
        "desc": "depthwise-separable inverted residual blocks (~3.5M params), ImageNet-pretrained, frozen",
    },
    "EfficientNet-B0": {
        "type": "transfer",
        "input_size": 224,
        "desc": "compound-scaled mobile inverted bottleneck (~5.3M params), ImageNet-pretrained, frozen",
    },
}
