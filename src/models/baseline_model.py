"""
Building damage classification models — Phase 1 (custom CNN) and Phase 2 (ResNet-18).

Phase 1: BranchCNN — 4-block custom CNN, randomly initialised.
Phase 2: ResNetBranch — pretrained ResNet-18 backbone, SAR branch uses averaged first conv.

Classes (num_classes=3): 0=Intact, 1=Damaged, 2=Destroyed.
Inputs: pre-event optical (3-ch) + post-event SAR (1-ch), both 224×224.
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import ResNet18_Weights


# ── Phase 1: Custom 4-block CNN ───────────────────────────────────────────────

class BranchCNN(nn.Module):
    """
    Single-modality encoder with two output paths:
      - global_features (B, feat_dim)       → used by Phase 1 classification head
      - spatial_features (B, feat_dim, H, W) → reserved for Phase 3 decoder

    AdaptiveAvgPool2d is a separate module (not fused into the encoder),
    so encoder weights transfer to a Phase 3 UNet decoder without modification.
    """

    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        c = base_channels

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, c,   3, padding=1), nn.BatchNorm2d(c),   nn.ReLU(inplace=True),
            nn.Conv2d(c,          c,   3, padding=1), nn.BatchNorm2d(c),   nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(c,   2*c, 3, padding=1), nn.BatchNorm2d(2*c), nn.ReLU(inplace=True),
            nn.Conv2d(2*c, 2*c, 3, padding=1), nn.BatchNorm2d(2*c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(2*c, 4*c, 3, padding=1), nn.BatchNorm2d(4*c), nn.ReLU(inplace=True),
            nn.Conv2d(4*c, 4*c, 3, padding=1), nn.BatchNorm2d(4*c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(4*c, 8*c, 3, padding=1), nn.BatchNorm2d(8*c), nn.ReLU(inplace=True),
            nn.Conv2d(8*c, 8*c, 3, padding=1), nn.BatchNorm2d(8*c), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.feat_dim   = 8 * c   # 256 with base_channels=32
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        s1 = self.block1(x)
        s2 = self.block2(s1)
        s3 = self.block3(s2)
        s4 = self.block4(s3)
        return s1, s2, s3, s4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, _, spatial = self.encode(x)
        pooled      = self.global_pool(spatial)
        global_feat = self.head(pooled)
        return spatial, global_feat


class MultimodalDamageCNN(nn.Module):
    """Phase 1 dual-branch late-fusion classifier."""

    def __init__(self, num_classes: int = 3, base_channels: int = 32):
        super().__init__()
        self.optical_branch = BranchCNN(in_channels=3, base_channels=base_channels)
        self.sar_branch     = BranchCNN(in_channels=1, base_channels=base_channels)
        feat_dim = self.optical_branch.feat_dim

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        _, opt_feat = self.optical_branch(optical)
        _, sar_feat = self.sar_branch(sar)
        return self.classifier(torch.cat([opt_feat, sar_feat], dim=1))

    def encode_both(self, optical: torch.Tensor, sar: torch.Tensor):
        opt_skips = self.optical_branch.encode(optical)
        sar_skips = self.sar_branch.encode(sar)
        return opt_skips, sar_skips


class SingleModalDamageCNN(nn.Module):
    """Phase 1 single-modality baseline."""

    def __init__(self, in_channels: int, num_classes: int = 3, base_channels: int = 32):
        super().__init__()
        self.branch = BranchCNN(in_channels=in_channels, base_channels=base_channels)
        feat_dim = self.branch.feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, global_feat = self.branch(x)
        return self.classifier(global_feat)


# ── Phase 2: Pretrained ResNet-18 backbone ────────────────────────────────────

class ResNetBranch(nn.Module):
    """
    ResNet-18 encoder with pretrained ImageNet weights.

    SAR variant (in_channels=1): the pretrained 3-channel first conv weights are
    averaged across the channel dimension to initialise the 1-channel conv.
    This preserves ImageNet-learned low-level edge detectors for SAR intensity input
    (standard domain adaptation technique — He et al., 2016; Mou et al., 2019).

    AdaptiveAvgPool2d kept as a separate module (not inside the encoder sequence)
    so Phase 3 can replace it with a UNet decoder without touching encoder weights.
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base    = tv_models.resnet18(weights=weights)

        if in_channels != 3:
            old_conv = base.conv1
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                                 padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    # Average 3-channel pretrained weights → 1-channel init
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            base.conv1 = new_conv

        # Encoder: all layers except avgpool and fc
        self.encoder = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,  # [0–3]
            base.layer1, base.layer2, base.layer3, base.layer4,  # [4–7]
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512   # ResNet-18 layer4 output channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.encoder(x)).flatten(1)   # (B, 512)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (global_features, skip_list) for Phase 3 UNet decoder."""
        x = self.encoder[:4](x)                        # stem → (B, 64, 56, 56)
        s1 = self.encoder[4](x)                        # layer1 → (B, 64, 56, 56)
        s2 = self.encoder[5](s1)                       # layer2 → (B, 128, 28, 28)
        s3 = self.encoder[6](s2)                       # layer3 → (B, 256, 14, 14)
        s4 = self.encoder[7](s3)                       # layer4 → (B, 512,  7,  7)
        global_feat = self.pool(s4).flatten(1)
        return global_feat, [s1, s2, s3, s4]


class MultimodalDamageCNNv2(nn.Module):
    """
    Phase 2 dual-branch classifier using pretrained ResNet-18 backbones.

    Optical branch: standard ResNet-18 (ImageNet pretrained).
    SAR branch:     ResNet-18 with 1-channel first conv (weights averaged from pretrained).
    Fusion:         concat(opt_feat, sar_feat) [1024-dim] → Linear → num_classes.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.optical_branch = ResNetBranch(in_channels=3, pretrained=True)
        self.sar_branch     = ResNetBranch(in_channels=1, pretrained=True)

        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        opt_feat = self.optical_branch(optical)
        sar_feat = self.sar_branch(sar)
        return self.head(torch.cat([opt_feat, sar_feat], dim=1))

    def backbone_params(self):
        return (list(self.optical_branch.parameters()) +
                list(self.sar_branch.parameters()))

    def head_params(self):
        return list(self.head.parameters())

    def encode_both(self, optical: torch.Tensor, sar: torch.Tensor):
        """Phase 3 entry point — returns skip maps for both branches."""
        _, opt_skips = self.optical_branch.encode(optical)
        _, sar_skips = self.sar_branch.encode(sar)
        return opt_skips, sar_skips


class SingleModalDamageCNNv2(nn.Module):
    """Phase 2 single-modality classifier using pretrained ResNet-18."""

    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        self.branch = ResNetBranch(in_channels=in_channels, pretrained=True)

        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.branch(x))

    def backbone_params(self):
        return list(self.branch.parameters())

    def head_params(self):
        return list(self.head.parameters())


# ── Factory ───────────────────────────────────────────────────────────────────

NUM_CLASSES = 3

def create_model(model_type: str = "multimodal", num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    model_type options:
      Phase 1 (custom CNN):       'multimodal' | 'optical_only' | 'sar_only'
      Phase 2 (ResNet-18):        'multimodal_v2' | 'optical_only_v2' | 'sar_only_v2'
    """
    if model_type == "multimodal":
        return MultimodalDamageCNN(num_classes=num_classes)
    elif model_type == "optical_only":
        return SingleModalDamageCNN(in_channels=3, num_classes=num_classes)
    elif model_type == "sar_only":
        return SingleModalDamageCNN(in_channels=1, num_classes=num_classes)
    elif model_type == "multimodal_v2":
        return MultimodalDamageCNNv2(num_classes=num_classes)
    elif model_type == "optical_only_v2":
        return SingleModalDamageCNNv2(in_channels=3, num_classes=num_classes)
    elif model_type == "sar_only_v2":
        return SingleModalDamageCNNv2(in_channels=1, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. "
                         f"Valid: multimodal, optical_only, sar_only, "
                         f"multimodal_v2, optical_only_v2, sar_only_v2")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("── Phase 1 ──")
    m1 = create_model("multimodal").to(device)
    print(f"  MultimodalDamageCNN:    {sum(p.numel() for p in m1.parameters()):,} params")

    print("── Phase 2 ──")
    m2 = create_model("multimodal_v2").to(device)
    print(f"  MultimodalDamageCNNv2: {sum(p.numel() for p in m2.parameters()):,} params")

    optical = torch.randn(2, 3, 224, 224).to(device)
    sar     = torch.randn(2, 1, 224, 224).to(device)

    with torch.no_grad():
        print(f"  logits shape: {m2(optical, sar).shape}")
        _, sar_skips = m2.sar_branch.encode(sar)
        print(f"  SAR skip shapes: {[s.shape for s in sar_skips]}")

    print("Model test passed.")
