# ADR-002: Area-Weighted Tile Label Derivation

**Status:** Amended 2026-06-29 — thresholds revised after Morocco empirical analysis  
**Date:** 2026-06-29  
**Deciders:** Sabrina Pribadi  
**Technical Story:** Grill-Me design interview — Q1 follow-up on label quality

---

## Context and Problem Statement

A tile-level classification label must be derived from the pixel-level segmentation mask. The naive approach (`max(non-background)`) labels any tile containing even a single Destroyed pixel as Destroyed — producing corrupt training signal: a tile with 200 Intact buildings and 1 collapsed wall gets labeled Destroyed, forcing the model to learn "Intact rooftop texture" = Destroyed.

## Decision Drivers

- BRIGHT mask pixel distribution: Intact 82.8%, Damaged 10.7%, Destroyed 6.5% (Section 2.3, Figure 5d)
- SAR sensitivity is asymmetric: IoU >70% for Destroyed (strong backscatter), IoU <20% for Damaged (subtle signal) — Table 5
- Registration error in BRIGHT is ~1 pixel; a minimum building coverage filter prevents noise-label corruption
- The label must reflect operational priority: field coordinators need "is there destruction in this tile?" not "what exact % of pixels are destroyed?"

## Considered Options

**Option A — Max label (original code)**
- `label = max(building_pixels) - 1`
- Rejected: Any single Destroyed pixel dominates the entire tile label regardless of area

**Option B — Dominant class / mode**
- `label = mode(building_pixels) - 1`
- Rejected: Intact always wins (82.8% base rate); tiles with 51% Intact and 49% Destroyed labeled Intact — loses all Destroyed signal

**Option C — Area-weighted threshold (chosen)**
- Destroyed if destroyed_pixels / building_pixels ≥ 5%
- Damaged if damaged_pixels / building_pixels ≥ 20%
- Intact otherwise
- Min building pixel filter: skip tiles with <200 building pixels

**Option D — Soft multi-label**
- Label = proportion vector [p_intact, p_damaged, p_destroyed], trained with BCELoss or KL-divergence
- Deferred to Phase 3: requires architectural changes to the classification head

## Decision Outcome

**Chosen: Option C — Area-weighted threshold.**

Threshold justification grounded in BRIGHT paper:
- **Destroyed ≥ 5%**: 5% of building pixels ≈ 25 pixels in 512×512 ≈ 1 collapsed building footprint. Destroyed has strong SAR signal, so low threshold is appropriate — any visible collapse is operationally significant.
- **Damaged ≥ 20%**: 20% ≈ 100 pixels ≈ 2–3 buildings with partial damage. Damaged is hard to detect in SAR (IoU <20%), so higher threshold avoids false positives.
- **Min 200 building pixels**: BRIGHT registration error ~1 pixel; fewer than 200 building pixels leaves label vulnerable to misregistration noise. 200 px ≈ 3 small buildings at 0.5 m/px.

## Positive Consequences

- Label reflects damage severity proportional to area, not single-pixel accidents
- Threshold asymmetry (aggressive for Destroyed, conservative for Damaged) matches SAR sensitivity profile from paper
- Minimum pixel filter removes sparse tiles that would corrupt training

## Negative Consequences

- Thresholds are fixed hyperparameters without automatic event-specific calibration
- Some tiles with mixed damage may be ambiguously labeled

## Amendment: Morocco empirical calibration (2026-06-29)

Training on Morocco alone (392 train tiles) revealed the original thresholds were too strict for this event:

**Diagnosis**: Morocco training split has only 3 tiles with any Destroyed pixels and 11 with any Damaged pixels. At 5%/20% thresholds, 0 Destroyed and 2 Damaged tiles survived → the model never saw a minority class example during training.

**Root cause**: The paper thresholds (Section 2.3) were calibrated on the global BRIGHT distribution (6.5% Destroyed, 10.7% Damaged building pixels across 14 events). Morocco's damage is sparser within each tile — even heavily-hit areas have mostly intact surrounding buildings.

**Revised thresholds (current defaults):**

| Class | Original | Revised | Justification |
|-------|----------|---------|---------------|
| Destroyed | 5% | **1%** | Morocco tiles have ~1–3% destroyed fraction even in hit areas; 1% ≈ 2 collapsed buildings at 0.5 m/px |
| Damaged | 20% | **5%** | 20% was so strict only 2 tiles passed; 5% recovers 6/11 Damaged tiles while rejecting borderline noise |
| Min building px | 200 | 200 | Unchanged — dropping only 5 tiles anyway |

**Outcome**: train Intact=138 / Damaged=6 / Destroyed=2; val Intact=20 / Damaged=2 / Destroyed=2. All 3 classes now represented in both splits.

**Note for Phase 1.5+**: When expanding to all 14 events, revisit whether event-specific thresholds (or a single global threshold calibrated on the full 4,246-tile distribution) perform better.

## Implementation Notes

```python
# src/data/brighT_loader.py — _derive_tile_label()
destroyed_frac = (mask == 3).sum() / n_building
damaged_frac   = (mask == 2).sum() / n_building
if n_building < 200:    return None   # too sparse
if destroyed_frac >= 0.01: return 2  # Destroyed (revised from 5%)
if damaged_frac   >= 0.05: return 1  # Damaged   (revised from 20%)
return 0                              # Intact
```

## Related Decisions

- ADR-001: Why tile-level classification (not segmentation)
- ADR-003: Validation metric

## References

- Chen et al. (2025). BRIGHT. ESSD 17(11). Section 2.3, Figure 5d, Table 5
