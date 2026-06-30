# ADR-003: Validation Strategy — Macro F1 + Relative Improvement Gate

**Status:** Accepted  
**Date:** 2026-06-29  
**Deciders:** Sabrina Pribadi  
**Technical Story:** Grill-Me design interview — Q2 Data Strategy

---

## Context and Problem Statement

Training on Morocco-only data (567 tiles), the validation strategy must answer one question: **"Is the multimodal signal (optical + SAR) providing real value over optical alone?"** This requires both a metric that is insensitive to class imbalance and an exit criterion that isolates the modality contribution from other confounds.

## Decision Drivers

- Class distribution is highly imbalanced: Intact ~83%, Damaged ~11%, Destroyed ~6%
- A model that always predicts Intact achieves ~83% accuracy — making accuracy meaningless
- Training on 1 event limits what absolute performance thresholds can be calibrated against
- The paper's cross-event zero-shot results (~0.38–0.44 Macro F1 with 13 training events) are not directly comparable to same-event supervised training
- Failure modes need to be distinguishable: (a) weak multimodal signal, (b) noisy labels, (c) insufficient model capacity

## Considered Options

**Metric candidates:**
- **Overall accuracy (OA)** — Rejected: majority-class bias (82.8% Intact)
- **Micro F1** — Rejected: equivalent to accuracy under this imbalance; dominated by Intact class
- **Macro F1** — Chosen: equal weight to each class; the model must learn both Damaged and Destroyed to score well

**Threshold candidates:**
- **Absolute threshold 0.35** — Partially rejected as primary: borrowed from cross-event zero-shot; miscalibrated for same-event supervised (expected ~0.55+). Retained as minimum floor only.
- **Relative improvement gate** — Chosen as primary: `F1(multimodal) > F1(optical_only) + 0.05` controls for label noise, tile difficulty, and single-event limitations

## Decision Outcome

**Primary validation criterion:** `F1(multimodal) > F1(optical_only) + 0.05` on the val set.  
**Secondary floor:** Macro F1 ≥ 0.35 (minimum bar; below this the model hasn't learned the minority classes at all).

**Falsification conditions:**

| Result | Diagnosis | Next Action |
|--------|-----------|-------------|
| Multimodal ≈ Optical-only (both < 0.35) | Weak signal OR noisy labels | Qualitative analysis of predictions; inspect tile label distribution |
| Multimodal ≈ Optical-only (both > 0.35) | SAR adds no value | Go single-modality optical; revisit SAR preprocessing |
| Multimodal < Optical-only | SAR actively hurts | Check SAR normalization; consider random-init SAR branch |
| Optical-only >> 0.55, Multimodal >> 0.60 | Signal validated | Proceed to Phase 2 (pretrained backbone, full BRIGHT) |

## Positive Consequences

- Relative gate controls for all confounds that affect both models equally (label noise, dataset size)
- Running optical-only and multimodal simultaneously is a clean ablation
- The training script already supports `model_type` = "multimodal" | "optical_only"

## Negative Consequences

- Requires training two models per experiment (doubles compute)
- 5% improvement threshold is a judgment call — could be too small for noisy data

## Implementation Notes

```python
# scripts/train_model.py — already implemented
macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
```

To run ablation:
```bash
MODEL_TYPE=optical_only python scripts/train_model.py   # baseline
MODEL_TYPE=multimodal    python scripts/train_model.py   # candidate
# Accept if: multimodal_f1 > optical_only_f1 + 0.05
```

## Related Decisions

- ADR-001: Task framing
- ADR-002: Label derivation (affects label quality, which affects F1 reliability)

## References

- Chen et al. (2025). BRIGHT. ESSD 17(11). Table 10 (cross-event transfer results)
