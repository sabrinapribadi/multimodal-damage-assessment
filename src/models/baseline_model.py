"""
Baseline dual-branch CNN for BRIGHT tile-level damage classification.

Phase 1: tile-level classification (current).
Phase 3 hook: BranchCNN.forward() returns (spatial_features, global_features) so
a UNet-style decoder can be added without rewriting the encoder.

Classes (num_classes=3): 0=Intact, 1=Damaged, 2=Destroyed.
Inputs: pre-event optical (3-ch) + post-event SAR (1-ch), both 224×224.
"""
import torch
import torch.nn as nn


class BranchCNN(nn.Module):
    """
    Single-modality encoder with two output paths:
      - global_features (B, feat_dim)     → used by Phase 1 classification head
      - spatial_features (B, feat_dim, H, W) → reserved for Phase 3 decoder

    AdaptiveAvgPool2d is a separate module (not fused into the encoder),
    so the encoder weights transfer to Phase 3 without modification.
    """

    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        c = base_channels

        # ── Encoder (spatial features preserved) ──────────────────────────
        # Each conv block doubles channels; MaxPool2d halves spatial dims.
        # Skip connection points (for Phase 3 UNet decoder):
        #   After block1: (B, c,   H/2,  W/2)   112×112
        #   After block2: (B, 2c,  H/4,  W/4)    56×56
        #   After block3: (B, 4c,  H/8,  W/8)    28×28
        #   After block4: (B, 8c,  H/16, W/16)   14×14  ← bottleneck
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

        self.feat_dim = 8 * c  # 256 with base_channels=32

        # ── CHANGE 1: global pool is a separate module, not inside encoder ──
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ── Classification sub-head (Phase 1 only) ────────────────────────
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    # ── CHANGE 2: encoder runs as sequential blocks (skip-connection ready) ─
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return feature maps at every block, pre-pool convention."""
        s1 = self.block1(x)   # (B, c,   112, 112)
        s2 = self.block2(s1)  # (B, 2c,   56,  56)
        s3 = self.block3(s2)  # (B, 4c,   28,  28)
        s4 = self.block4(s3)  # (B, 8c,   14,  14)  ← bottleneck
        return s1, s2, s3, s4

    # ── CHANGE 3: forward returns (spatial, global) — both paths available ──
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            spatial_features: (B, feat_dim, H/16, W/16)  — for Phase 3 decoder
            global_features:  (B, feat_dim)               — for Phase 1 classifier
        """
        _, _, _, spatial = self.encode(x)
        pooled = self.global_pool(spatial)   # (B, feat_dim, 1, 1)
        global_feat = self.head(pooled)      # (B, feat_dim)
        return spatial, global_feat


class MultimodalDamageCNN(nn.Module):
    """
    Dual-branch late-fusion model for tile-level damage classification (Phase 1).

    Fusion: concat(optical_global, sar_global) → Linear → num_classes.
    The spatial_features from each branch are forwarded through but unused here;
    Phase 3 decoder will consume them without modifying this class.
    """

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
        _, opt_feat = self.optical_branch(optical)   # (B, feat_dim)
        _, sar_feat = self.sar_branch(sar)           # (B, feat_dim)
        return self.classifier(torch.cat([opt_feat, sar_feat], dim=1))

    def encode_both(self, optical: torch.Tensor, sar: torch.Tensor):
        """
        Phase 3 entry point: returns all skip maps for both branches.
        Signature intentionally stable — Phase 3 decoder calls this unchanged.
        """
        opt_skips = self.optical_branch.encode(optical)  # (s1, s2, s3, s4)
        sar_skips = self.sar_branch.encode(sar)
        return opt_skips, sar_skips


class SingleModalDamageCNN(nn.Module):
    """Single-modality baseline (optical-only or SAR-only)."""

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


def create_model(model_type: str = "multimodal", num_classes: int = 3) -> nn.Module:
    """Factory. model_type: 'multimodal' | 'optical_only' | 'sar_only'"""
    if model_type == "multimodal":
        return MultimodalDamageCNN(num_classes=num_classes)
    elif model_type == "optical_only":
        return SingleModalDamageCNN(in_channels=3, num_classes=num_classes)
    elif model_type == "sar_only":
        return SingleModalDamageCNN(in_channels=1, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = create_model("multimodal").to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optical = torch.randn(2, 3, 224, 224).to(device)
    sar     = torch.randn(2, 1, 224, 224).to(device)

    with torch.no_grad():
        logits = model(optical, sar)
        print(f"Classification output: {logits.shape}")   # (2, 3)

        opt_skips, sar_skips = model.encode_both(optical, sar)
        print("Optical skip shapes:", [s.shape for s in opt_skips])
        # → [(2,32,112,112), (2,64,56,56), (2,128,28,28), (2,256,14,14)]

    print("Model test passed.")
