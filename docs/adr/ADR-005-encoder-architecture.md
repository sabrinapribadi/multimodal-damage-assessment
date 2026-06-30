# ADR-005: Encoder Architecture — Spatial/Global Feature Split

**Status:** Accepted  
**Date:** 2026-06-29  
**Deciders:** Sabrina Pribadi  
**Technical Story:** Grill-Me design interview — Q4 Architecture Patterns

---

## Context and Problem Statement

Phase 1 requires only a global feature vector per tile for classification. Phase 3 requires spatial feature maps (with skip connections at multiple resolutions) for a UNet-style segmentation decoder. The original design fused `AdaptiveAvgPool2d(1)` into the encoder — a one-way door: once pooled, spatial structure cannot be recovered.

The question: should Phase 1 be architected so Phase 3 is a surgical addition, or is a Phase 3 rewrite acceptable?

## Decision Drivers

- The decoupled model (Y^dam = Y^loc ⊙ Y^clf) requires a decoder that upsamples from bottleneck feature maps back to image resolution
- UNet-style decoders need skip connections at every encoder resolution level (112, 56, 28, 14 for 224×224 input)
- If `AdaptiveAvgPool2d` is inside the encoder's `nn.Sequential`, it cannot be bypassed without rewriting the encoder
- Phase 1 encoder weights are valuable: they represent ~20 epochs of gradient signal on Morocco data. Throwing them away in Phase 3 wastes that investment.

## Considered Options

**Option A — Keep AdaptiveAvgPool2d inside encoder (original)**
- `self.features = nn.Sequential(..., AdaptiveAvgPool2d(1))`
- Consequence: Phase 3 requires rewriting the entire encoder. Phase 1 weights are discarded.
- Rejected

**Option B — Separate global pool from encoder (chosen)**
- Three-line change:
  1. Rename `self.features` → `self.encoder`; remove `AdaptiveAvgPool2d` from it
  2. Add `self.global_pool = nn.AdaptiveAvgPool2d(1)` as a separate module
  3. `forward()` returns `(spatial_features, global_features)` — both paths available
- Phase 1 uses `global_features` for classification
- Phase 3 adds decoder consuming `spatial_features` — encoder untouched

## Decision Outcome

**Chosen: Option B.**

```python
# BranchCNN.forward() — src/models/baseline_model.py
def forward(self, x):
    _, _, _, spatial = self.encode(x)   # (B, 256, 14, 14) — preserved
    pooled = self.global_pool(spatial)  # (B, 256, 1, 1)
    global_feat = self.head(pooled)     # (B, 256)
    return spatial, global_feat         # both paths

# Phase 3 entry point — already stable
def encode_both(self, optical, sar):
    opt_skips = self.optical_branch.encode(optical)  # (s1, s2, s3, s4)
    sar_skips = self.sar_branch.encode(sar)
    return opt_skips, sar_skips
```

Skip connection shapes for 224×224 input:
- s1: (B, 32, 112, 112)
- s2: (B, 64,  56,  56)
- s3: (B, 128, 28,  28)
- s4: (B, 256, 14,  14) ← bottleneck

## Positive Consequences

- Phase 3 decoder attaches to `encode_both()` without modifying encoder
- Phase 1 encoder weights transfer directly to Phase 3
- `SingleModalDamageCNN` follows the same pattern — ablations are consistent

## Negative Consequences

- Slight overhead: forward pass always computes spatial features even when only global is needed
- The `(spatial, global_feat)` tuple return changes the interface — all callers must unpack

## Implementation Notes

```python
# Current Phase 1 usage (MultimodalDamageCNN.forward):
_, opt_feat = self.optical_branch(optical)
_, sar_feat = self.sar_branch(sar)
return self.classifier(torch.cat([opt_feat, sar_feat], dim=1))

# Phase 3 addition (does not modify above):
opt_skips, sar_skips = self.encode_both(optical, sar)
damage_map = self.decoder(opt_skips, sar_skips)  # new module
```

## Related Decisions

- ADR-001: Task framing (Phase 1 → Phase 3 roadmap)
- ADR-004: Backbone swap (encoder interface must be stable for ResNet-18 drop-in)

## References

- Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
- Chen et al. (2025). BRIGHT. ESSD 17(11). Section 3.1 (decoupled model Y^dam = Y^loc ⊙ Y^clf)
