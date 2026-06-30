# ADR-001: Tile-Level Classification as Stepping Stone to Decoupled Segmentation

**Status:** Accepted  
**Date:** 2026-06-29  
**Deciders:** Sabrina Pribadi  
**Technical Story:** Grill-Me design interview — Q1 Core Problem Definition

---

## Context and Problem Statement

The BRIGHT dataset provides pixel-level segmentation masks (0=background, 1=Intact, 2=Damaged, 3=Destroyed). The benchmark evaluation is pixel-level (mIoU, per-class F1). The end goal is an operational triage tool — within 6–72 hours of a disaster, rank affected tiles by severity for field coordinators.

The question is: should the system produce a **pixel-level damage map** (segmentation) or a **tile-level damage score** (classification)?

## Decision Drivers

- Validating that the multimodal signal (optical + SAR) is learnable is the Phase 1 goal
- Segmentation requires a full encoder–decoder with skip connections; classification only needs an encoder + pool + linear
- 567 Morocco tiles is a small dataset — simpler models train more reliably
- The BRIGHT paper (Section 3.1) itself uses a decoupled formula: Y^dam = Y^loc ⊙ Y^clf, confirming classification and localization are separable tasks
- The paper's direct-prediction models (UNet, DeepLabV3+) demonstrate tile-level output is a valid evaluation modality

## Considered Options

**Option A — Direct pixel-level segmentation**
- Pros: Matches benchmark evaluation (mIoU); per-building output; full damage map
- Cons: Requires decoder; more parameters; unstable on 567 tiles; overkill before signal validation
- Why considered: It's the paper's primary evaluation mode

**Option B — Tile-level classification (chosen)**
- Pros: Simpler; trains reliably on small datasets; ~10× faster; validates multimodal signal independently of decoder complexity; matches triage use case
- Cons: Cannot report mIoU; no per-building resolution; collapses spatial information
- Why considered: Paper validates this mode explicitly ("used for quick triage and validation")

**Option C — Building-level instance classification**
- Pros: True per-building output matching the damage response workflow
- Cons: Requires building footprint detection (separate model); dataset doesn't provide instance masks in BRIGHT format
- Why considered: Closest to operational need

## Decision Outcome

**Chosen: Option B — Tile-level classification.**

Phase 1 validates the multimodal fusion signal. Phase 3 adds a UNet-style decoder surgically (encoder weights reused, no retraining). The encoder architecture was specifically designed to make this transition a surgical addition, not a rewrite (see ADR-006).

## Positive Consequences

- Training converges reliably on limited data
- Clean experiment: if multimodal > optical-only, signal is real
- BranchCNN already exposes `encode()` for Phase 3 skip connections

## Negative Consequences

- Cannot compare directly to BRIGHT benchmark mIoU results
- Tile label collapses spatial heterogeneity within a tile
- Requires separate label-derivation logic (see ADR-002)

## Implementation Notes

- Label derived from segmentation mask using area-weighted thresholds (ADR-002)
- Encoder preserves spatial features for Phase 3 (ADR-006)
- Validation gate: F1(multimodal) > F1(optical-only) + 0.05 on val set (ADR-003)

## Related Decisions

- ADR-002: Label derivation
- ADR-003: Validation metric
- ADR-006: Encoder architecture

## References

- Chen et al. (2025). BRIGHT. ESSD 17(11). DOI: 10.5194/essd-17-6217-2025 — Section 3.1, Table 5
