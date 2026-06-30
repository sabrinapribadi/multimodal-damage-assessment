# ADR-004: ResNet-18 Backbone with Averaged First Conv for SAR Branch

**Status:** Accepted (Phase 2)  
**Date:** 2026-06-29  
**Deciders:** Sabrina Pribadi  
**Technical Story:** Grill-Me design interview — Q3 Technology Stack

---

## Context and Problem Statement

**Phase 1** uses a custom 4-layer CNN from random initialization. The confound is: if Phase 1 performance is low, we cannot distinguish between (a) weak multimodal signal and (b) random-init instability on 567 tiles.

**Phase 2** must isolate the multimodal signal by swapping to a pretrained backbone, fixing the architecture confound. The challenge: the SAR branch requires a 1-channel input, but ImageNet pretrained models expect 3-channel RGB.

## Decision Drivers

- BRIGHT benchmark: SiamCRNN and ChangeOS (both ResNet-18 backbone) achieve mIoU 65.73% and 65.98% — most direct comparison for Phase 2
- ResNet-18 (11M params) fits in 8 GB RAM; ResNet-50 (25M params) risks OOM on M2 MPS
- SAR backscatter (microwave reflectivity) and RGB color have different statistical domains — naïve pretrained application to SAR is incorrect
- Remote sensing literature validates averaged first conv as the standard adaptation technique for single-channel inputs

## Considered Options

**SAR adaptation options:**

**Option A — Repeat 1→3 (copy SAR channel 3×)**
- Rejected: First conv layer sees three identical channels — cross-channel patterns are undefined; ImageNet statistics are incorrect (model "thinks" it's seeing RGB)

**Option B — Average first conv weights (chosen)**
- Collapse `Conv2d(3,64,7)` → `Conv2d(1,64,7)` by `weight.mean(dim=1, keepdim=True)`
- Output shape (64, H, W) is identical — no change to subsequent layers
- Gradient magnitudes preserved; activation statistics maintained
- Empirically validated in remote sensing community (Sentinel-1, Landsat adaptations)

**Option C — Random-init SAR branch only**
- Pretrained optical, random SAR
- Clean domain separation, but SAR branch gets no transfer benefit
- SAR branch needs more data to converge from scratch
- Rejected for Phase 2; revisit if averaged-conv approach underperforms

**Backbone size:**

- **ResNet-18** (chosen): 11M params, paper-comparable (SiamCRNN, ChangeOS)
- **ResNet-50**: 25M params, memory risk on M2, overkill for 567-tile Phase 2

## Decision Outcome

**Phase 2 architecture:**
- Optical branch: ResNet-18, standard ImageNet pretrained weights
- SAR branch: ResNet-18, weights initialized by averaging first conv across input channel dim

```python
# Phase 2 implementation (not yet in code)
resnet = torchvision.models.resnet18(pretrained=True)
sar_conv1 = resnet.conv1.weight.mean(dim=1, keepdim=True)  # (64,1,7,7)
```

**Phase ordering:**
1. Phase 1: Custom CNN from scratch, Morocco only — validates pipeline
2. Phase 2: ResNet-18 backbone, Morocco first → full BRIGHT — validates signal
3. Phase 3: Add segmentation decoder — validates localization
4. Phase 4: ChangeMamba / DamageFormer — match paper benchmarks

Note: backbone swap happens before data expansion to keep variables isolated.

## Positive Consequences

- Removes architecture confound from multimodal signal evaluation
- ResNet-18 is a known quantity in BRIGHT literature — results are interpretable
- Averaged first conv is reversible — can ablate to random-init SAR if needed

## Negative Consequences

- ImageNet features (color edges, textures) are domain-mismatched for SAR — pretrained weights help initialization stability but the learned filters will need to adapt
- Requires torchvision dependency (already present)

## Implementation Notes

Phase 2 implementation is future work. `BranchCNN` in Phase 1 is designed so the encoder can be swapped without changing the classifier or fusion logic.

## Related Decisions

- ADR-001: Task framing (Phase roadmap)
- ADR-006: Encoder architecture (forward-compatible with backbone swap)

## References

- Chen et al. (2025). BRIGHT. ESSD 17(11). Table 5 (model comparison with backbones)
- He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
