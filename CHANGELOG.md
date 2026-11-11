# Changelog

Newest first. Each entry uses STAR (Situation, Task, Action, Result) so it can be told as a story.

## [5.1.0] — 2026-11-11 — CI on every push
**Files:** tests/test_inference_results.py, .github/workflows/ci.yml
- **Situation:** No CI and no tests; the README's Phase 3 numbers had nothing checking them against the data.
- **Task:** CI on every push (portfolio rule R19) that fails if a headline number drifts from the committed results.
- **Action:** Contract tests recompute macro F1 per split from `data/inference_results.parquet` and check confidences are probabilities; CI runs them with the dashboard's light dependencies plus an error-level lint.
- **Result:** 5 tests pass; README values (val 0.580 vs 0.558, test 0.454 vs 0.512) reproduce exactly.
- **Talking point:** Every number in my README is now re-derived from the data on each push.

## [5.0.1] — 2026-10-12 — MIT license
**Files:** LICENSE, CHANGELOG.md (new)
- **Situation:** The repository had no license file, so by default nobody else could legally reuse the code.
- **Task:** Make the portfolio code reusable with credit, consistently across all portfolio repositories.
- **Action:** Added the standard MIT license (© 2026 Sabrina Pribadi). Data sources keep their own terms; the license covers code only.
- **Result:** GitHub now detects the license as MIT; no code changed.
- **Talking point:** "All my portfolio code is MIT-licensed, so anyone can reuse it with attribution."

