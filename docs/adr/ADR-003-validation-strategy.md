# ADR-003: Validation Strategy — Macro F1 + Relative Improvement Gate

**Status:** Accepted — Research Complete (gate not cleared across 6 phases; best delta +0.022)  
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

## Empirical Gate Results — All Phases (2026-07-02)

| Phase | Events | Model | MM Val F1 | OPT Val F1 | SAR Delta | Gate |
|-------|--------|-------|-----------|------------|-----------|------|
| Phase 1 | Turkey | Custom CNN | 0.409 | 0.392 | +0.017 | Not passed |
| Phase 1.5 | Turkey + Beirut | Custom CNN | 0.421 | 0.443 | −0.023 | Not passed (negative) |
| Phase 2b | Turkey + Noto | ResNet-18 concat | 0.597 | 0.628 | −0.031 | Not passed (negative) |
| **Phase 3** | Turkey + Noto | ResNet-18 opt-gate | **0.580** | 0.558 | **+0.022** | **Not passed — best result** |
| Phase 4 | Turkey + Noto + Morocco | ResNet-18 soft labels | 0.396 | 0.492 | −0.096 | Not passed (negative) |
| Phase 3b | Turkey + Noto | ResNet-18 soft labels | 0.283 | 0.345 | −0.114 | Not passed (negative) |

**Gate verdict (final):** Not cleared. Closest approach: +0.022 (Phase 3). Required: +0.050.

**Gate interpretation across phases:**
- Phase 1 gate (+0.017): correctly identified that Damaged class (6 val samples) was the bottleneck — drove the event expansion decision
- Phase 1.5 gate (−0.023): correctly flagged cross-event-type SAR failure — drove the switch from explosion to earthquake domain
- Phase 2b gate (−0.031 macro, but Destroyed +6.4 pp, Damaged −16.7 pp): gate revealed SAR is class-conditional — drove the optical-gated fusion design in Phase 3
- Phase 3 gate (+0.022): gate confirmed the optical gate improved SAR signal; not yet sufficient — drove the data expansion and soft label experiments
- Phase 4 gate (−0.096): gate confirmed Morocco domain shift breaks SAR fusion more than optical — falsified the hypothesis that more earthquake data helps multimodal
- Phase 3b gate (−0.114): gate confirmed soft labels hurt both models at tile-level granularity — falsified the soft label hypothesis

**Research conclusion:** The gate worked correctly as a falsification instrument throughout. Each negative result was diagnosable and redirected experimentation. The +0.050 bar appropriately distinguishes noise from signal — the +0.022 Phase 3 result is likely real but insufficient to claim operational SAR value at this architecture scale. Clearing the gate likely requires pixel-level supervision (UNet decoder), explicit temporal change encoding (ChangeMamba), or domain-specific SAR pre-training.

## Related Decisions

- ADR-001: Task framing
- ADR-002: Label derivation (affects label quality, which affects F1 reliability)

## References

- Chen et al. (2025). BRIGHT. ESSD 17(11). Table 10 (cross-event transfer results)
