# Architecture Decision Records

This directory documents the key design decisions made during the multimodal damage assessment
project. Each ADR captures the context, options considered, decision, and consequences —
written before implementation, not after.

The goal is to make the reasoning behind non-obvious choices auditable and reversible.

## Index

| # | Title | Status | Summary |
|---|-------|--------|---------|
| [ADR-001](ADR-001-task-framing.md) | Tile-level classification as stepping stone to pixel segmentation | Accepted | Start with tile classification (encoder only) to validate the SAR fusion signal before investing in a full encoder–decoder. BranchCNN exposes `encode()` and `encode_both()` so Phase 3 UNet decoder is a surgical addition, not a rewrite. |
| [ADR-002](ADR-002-tile-label-derivation.md) | Area-weighted label derivation | Amended | Tile label derived from pixel-level mask using building-pixel thresholds: Destroyed ≥1%, Damaged ≥5%, skip if <200 building px. Thresholds revised from 5%/20% after Morocco empirical analysis yielded zero Destroyed training examples. |
| [ADR-003](ADR-003-validation-strategy.md) | Macro F1 + relative improvement gate | Accepted | Validation metric is macro F1 with explicit `labels=list(range(3))` to prevent sklearn from averaging only present classes. Ablation gate: F1(multimodal) > F1(optical-only) + 0.05. Gate is falsifiable — Phase 1 result (+0.017) correctly flagged insufficient data rather than declaring success. |
| [ADR-004](ADR-004-backbone-and-sar-adaptation.md) | ResNet-18 backbone with averaged first conv (Phase 2) | Accepted | Phase 2 will replace the custom CNN with a pretrained ResNet-18 optical branch. SAR branch uses the same architecture with the 3-channel first conv weights averaged to 1 channel — standard domain adaptation technique that preserves ImageNet-learned low-level features. |
| [ADR-005](ADR-005-encoder-architecture.md) | Spatial/global split; AdaptiveAvgPool2d as separable module | Accepted | `AdaptiveAvgPool2d(1)` is a separate module (not fused into the final conv block) so Phase 3 can replace it with a UNet decoder without touching the encoder weights. The `encode()` method returns all 4 skip feature maps for decoder hook points. |
| [ADR-006](ADR-006-deployment-and-output.md) | GeoJSON/PNG/SMS output hierarchy for field deployment | Accepted | Phase 2+ operational output: GeoJSON (primary, QGIS-compatible), PNG overlay (secondary, field tablets), SMS summary (tertiary, no-internet fallback). Phase 1 uses Streamlit dashboard with pre-computed parquet — no model or TIF files at runtime. |

## How to Read an ADR

Each file follows the standard format:

- **Status** — Accepted / Proposed / Amended / Superseded
- **Context** — Why this decision was needed
- **Considered Options** — The realistic alternatives with pros/cons
- **Decision Outcome** — What was chosen and why
- **Consequences** — Positive and negative downstream effects
- **Implementation Notes** — Specific code constraints that enforce the decision

## Decision Dependencies

```
ADR-001 (tile classification)
  └── ADR-002 (label derivation — needed because classification requires tile-level labels)
  └── ADR-003 (validation gate — needed to interpret classification results)
  └── ADR-005 (encoder design — preserves upgrade path from ADR-001)

ADR-004 (Phase 2 backbone)
  └── ADR-005 (encoder must be swappable — drives the separable pool module choice)

ADR-006 (output format)
  └── ADR-001 (output granularity follows from tile vs. pixel classification decision)
```
