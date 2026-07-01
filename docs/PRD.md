PRODUCT REQUIREMENT DOCUMENT (PRD)
Project: Multimodal Building Damage Assessment — SAR + Optical Fusion with BRIGHT
Version: 5.0 (Research Complete — All Phases Concluded)
Author: Sabrina Pribadi
Date: July 2, 2026
Status: Research Complete — Phase 3 is best result; gate not cleared; findings documented


1. EXECUTIVE SUMMARY

Problem: After a major disaster, first responders and humanitarian coordinators need to
rapidly prioritise which areas to deploy to. Current workflows rely on manual aerial survey
or crowd-sourced damage reports — both slow and incomplete. Satellite imagery (optical + SAR)
is available within 6–72 hours of any event worldwide, but converting raw imagery into
actionable damage maps requires ML expertise that most response organisations lack.

Solution: A dual-modality deep learning system that fuses pre-event optical imagery with
post-event Synthetic Aperture Radar (SAR) to classify building damage at the tile level.
Built on the BRIGHT dataset (14 real disaster events, 4,246 tiles), the system uses a
custom dual-branch CNN to demonstrate that the SAR channel provides additive signal over
optical-only — then provides a Streamlit dashboard with pre-computed inference results,
filterable tile gallery, confusion matrices, and per-tile confidence inspection, deployable
to Streamlit Cloud without model weights or GeoTIFF files.

Value Proposition: Demonstrate end-to-end satellite ML — from selective range-request
downloading of multi-GB HuggingFace zips, to multi-modal feature fusion, to Streamlit
deployment via pre-computed parquet — as a cohesive portfolio project. Validate the SAR
signal with a rigorous ablation gate before investing in heavier architectures.

Success Metrics:
- Download pipeline fetches only the requested event's tiles (<5% of full zip size)
- Ablation: multimodal macro F1 > optical-only macro F1 (positive SAR signal confirmed)
- Dashboard loads from parquet in under 3 seconds with no model weights or TIFs needed
- All 6 Architecture Decision Records documented before coding each component
- Phase roadmap (Phase 1 → Phase 4) clearly defined and documented


2. PROJECT CONTEXT AND BACKGROUND

This project was built to demonstrate capabilities across:

1. Satellite ML and Remote Sensing: Working with GeoTIFF (rasterio), multi-band inputs
   (optical RGB + SAR intensity), geospatial tile alignment, and domain-specific
   preprocessing (percentile stretch for optical, normalisation for SAR).

2. Deep Learning Architecture: Designing a dual-branch CNN from first principles with
   deliberate choices (separate AdaptiveAvgPool2d module, spatial feature preservation
   for future UNet decoder, late fusion via concatenation) documented in ADRs.

3. Network Protocol Engineering: Implementing HTTP range requests to selectively download
   individual files from multi-GB zip archives without fetching the full archive —
   including zip64 EOCD parsing, CDN URL expiry handling, and exponential-backoff retry.

4. Ablation Methodology: Structuring an experiment with a pre-defined falsification gate
   (F1(multimodal) > F1(optical_only) + 0.05) so results are interpretable regardless of outcome.

5. Deployment Engineering: Building a Streamlit dashboard that is self-contained after a
   one-time export script — no model weights, no rasterio, no PyTorch at runtime.
   Pre-computed parquet with base64 thumbnails is committed to git and served on Streamlit Cloud.

6. Architectural Decision Making: Six ADRs written before implementation, covering task
   framing, label derivation, validation strategy, backbone choice, encoder design, and
   output format — following the same rigour expected in production ML systems.

Data Source: BRIGHT Dataset (Chen et al., ESSD 2025)
- HuggingFace: Kullervo/BRIGHT (pre-event.zip 9.9 GB, post-event.zip 3.3 GB, target.zip)
- 14 disaster events across earthquake, wildfire, hurricane, volcano, and conflict types
- 4,246 tiles total; each tile 512×512 pixels at very-high resolution (0.3–0.5 m/px)
- Pixel-level labels: 0=background, 1=Intact, 2=Damaged, 3=Destroyed


3. SCOPE

In-Scope:
- Data: BRIGHT dataset (turkey-earthquake event for Phase 1 PoC)
- Models: Custom dual-branch CNN (MultimodalDamageCNN, SingleModalDamageCNN)
- Download Pipeline: Selective range-request downloader for HuggingFace zip archives
- Label Derivation: Area-weighted tile label from pixel-level segmentation masks
- Training: Multi-event, dual-ablation (multimodal vs optical-only) via MODEL_TYPE env var
- Validation: Macro F1 with explicit class labels; relative improvement gate
- Export: Pre-computed inference parquet with 64x64 base64 thumbnails
- Dashboard: Streamlit 3-tab app reading parquet only (no model/TIF at runtime)
- Documentation: 6 ADRs, architecture diagrams, phase roadmap
- Phase Roadmap: Defined path from Phase 1 (custom CNN) to Phase 4 (DamageFormer)

Out-of-Scope (current version):
- Real-time inference pipeline (GeoTIFFTiler + live satellite feed)
- Pixel-level segmentation (Phase 3 — requires UNet decoder)
- ResNet-18 pretrained backbone (Phase 2)
- DamageFormer / ChangeMamba (Phase 4)
- GeoJSON/PNG/SMS output generation (Phase 2+)
- User authentication or multi-tenancy
- Cloud GPU training (local MPS / CPU only)


4. USER PERSONAS AND STORIES

| Persona               | Goal                                                          | Pain Point                               |
|-----------------------|---------------------------------------------------------------|------------------------------------------|
| Alex (Field Coord.)   | Know which tiles to deploy teams to in the first 24 hours    | Manual survey takes days                 |
| Maria (GIS Analyst)   | Generate a damage layer for QGIS within hours of an event    | Existing tools require optical only      |
| James (ML Engineer)   | Validate that fusing SAR with optical actually helps         | No rigorous ablation framework exists    |
| Dana (Data Scientist) | Explore per-tile model confidence and identify failure modes  | Raw .pt files are not browseable         |
| Sam (Donor Org.)      | See what fraction of an affected area is destroyed vs intact  | Cannot read GeoTIFF or run Python        |

User Stories Implemented:
- As a field coordinator, I can view a paginated tile gallery filtered by damage class to
  identify the highest-severity zones at a glance.
- As an ML engineer, I can run multimodal vs optical-only training in a single command and
  compare their val macro F1 against a pre-defined gate.
- As a data scientist, I can select any tile in the inspector and see exactly what probability
  each model assigned to each damage class.
- As a GIS analyst, I can read the confusion matrix and understand precisely where the model
  fails (e.g. Damaged class with too few training examples).
- As a non-technical user, I can view the Streamlit dashboard without installing PyTorch,
  rasterio, or downloading a single TIF file.
- As an ML engineer, I can add a new event in one command and retrain on the combined dataset.


5. FUNCTIONAL REQUIREMENTS

Module A: Data Download Pipeline
- A.1 Selective range-request download: read only the zip64 central directory from the
     HuggingFace CDN via HTTP Range header; extract only entries whose filename contains
     the requested event name; never download the full archive.
- A.2 Zip64 EOCD parsing: locate End of Central Directory using 8 KB tail read
     (reduced from 65 KB to reduce CDN throttling). Handle both zip64 (PK\x06\x07 locator)
     and regular zip (PK\x05\x06) footers.
- A.3 Zip64 extended info sequential field reading: per ZIP spec §4.5.3, zip64 extra
     fields (original_size, compressed_size, local_hdr_offset) are present ONLY when their
     32-bit placeholder is 0xFFFFFFFF, in order. Read with a sequential foff pointer, not
     by assumed index position.
- A.4 CDN URL refresh: HuggingFace pre-signed CDN URLs expire after ~1 hour.
     Raise CDNExpired on HTTP 403; catch in download loop; call _get_cdn(force_refresh=True)
     and retry once before skipping.
- A.5 Exponential-backoff retry: 5 attempts with delays 1s → 2s → 4s → 8s → 16s for
     transient network drops ("peer closed connection without sending complete message body").
- A.6 Central directory disk cache: after first successful CD read, write all entry
     metadata (filename, comp_method, compressed_size, local_hdr_offset) to
     data/.cd_cache/{zip_name}.{event}.json. On subsequent runs, load from cache and
     skip the EOCD + CD network reads entirely.
- A.7 Resume safety: check dest.exists() before downloading each tile; skip if present.
- A.8 Multi-event support: accept multiple event names as CLI arguments; download all.
     Usage: python scripts/download_events.py turkey-earthquake beirut-explosion --all

Module B: Data Loading (BRIGHTDataset)
- B.1 BRIGHT file structure: pre-event/{tile_id}_pre_disaster.tif (optical RGB),
     post-event/{tile_id}_post_disaster.tif (SAR 1-ch),
     target/{tile_id}_building_damage.tif (mask 0-3).
- B.2 Tile label derivation (_derive_tile_label): area-weighted thresholds on building
     pixels (mask > 0). Default thresholds (Turkey-calibrated):
     Destroyed if destroyed_px/building_px >= 1%; Damaged if damaged_px/building_px >= 5%;
     Intact otherwise; return None (skip tile) if building_px < 200.
- B.3 Corruption guard: _is_valid_tif() reads a 1x1 pixel window via rasterio to validate
     pixel data (not just header). Called in _load_split for all three files per tile.
     Corrupt tiles are skipped with a logger.warning; training never crashes mid-epoch.
- B.4 GeoTIFF reading: rasterio-first with PIL fallback for non-geo TIFs.
     Optical: 3-band read, percentile (2–98%) stretch to uint8 RGB.
     SAR: 1-band read, 0-clipped, max-normalised to uint8 grayscale.
- B.5 Normalisation: optical — ImageNet mean/std; SAR — mean=0.5, std=0.5.
- B.6 Multi-event: ConcatDataset of per-event BRIGHTDatasets when multiple events given.
- B.7 Split files: standard_ML split (train_set.txt, val_set.txt) from BRIGHT benchmark.
     Tiles not on disk are silently skipped (partial downloads are fine).

Module C: Model Architecture
- C.1 BranchCNN: single-modality encoder with 4 conv blocks (32→64→128→256 channels),
     each block = Conv→BN→ReLU×2→MaxPool2d. Final spatial feature map: (B, 256, 14, 14).
     AdaptiveAvgPool2d(1) is a separate module (not fused into block4) so encoder weights
     transfer to Phase 3 UNet decoder without modification.
- C.2 encode() method: returns all 4 skip feature maps (s1,s2,s3,s4) for Phase 3
     UNet decoder hook. Intentionally stable signature.
- C.3 MultimodalDamageCNN: two BranchCNN branches (optical in_ch=3, SAR in_ch=1).
     Late fusion: concat(opt_global, sar_global) → Linear(512→256) → ReLU → Dropout(0.3)
     → Linear(256→3). 2,611,459 parameters.
- C.4 SingleModalDamageCNN: one BranchCNN branch → Linear(256→128) → ReLU → Dropout(0.3)
     → Linear(128→3). 1,273,251 parameters. Used for optical_only and sar_only ablations.
- C.5 encode_both(): Phase 3 entry point on MultimodalDamageCNN; returns (opt_skips, sar_skips)
     without modifying the forward() path or requiring retraining.

Module D: Training Pipeline
- D.1 MODEL_TYPE environment variable: "multimodal" (default) | "optical_only" | "sar_only".
     Governs which model is created and which forward() path is used in train/eval loops.
- D.2 Loss function: CrossEntropyLoss with class weights [1.0, 5.0, 10.0] for
     Intact / Damaged / Destroyed. Weights reflect label scarcity in training split.
- D.3 Optimizer: AdamW lr=1e-3, weight_decay=1e-4.
- D.4 Scheduler: CosineAnnealingLR over 20 epochs.
- D.5 Macro F1 computation: sklearn f1_score with labels=list(range(NUM_CLASSES)) and
     zero_division=0. The explicit labels parameter ensures all 3 classes contribute to
     the average even when a class is absent from the val batch — prevents inflated F1
     when only 2 classes appear.
- D.6 Checkpoint: saved whenever val macro F1 exceeds the running best. Filename:
     outputs/best_{run_name}_{model_type}.pt. Stores epoch, model_state_dict,
     optimizer_state_dict, val_loss, val_acc, val_f1, events, model_type, num_classes.
- D.7 Final report: loads best checkpoint, runs classification_report with explicit
     labels=list(range(NUM_CLASSES)) and target_names. Avoids ValueError when val set
     contains fewer classes than target_names.
- D.8 Multi-event: run_name = "+".join(event.replace("-","_") for event in events).
     Checkpoint names disambiguate across event combinations.

Module E: Inference Export
- E.1 export_inference.py: loads both checkpoints (multimodal + optical_only), iterates
     over train and val splits, runs both models on each tile, and builds a records list.
- E.2 Parquet schema: tile_id, split, true_label, true_label_name, pred_multimodal,
     pred_optical, conf_mm_intact, conf_mm_damaged, conf_mm_destroyed, conf_opt_intact,
     conf_opt_damaged, conf_opt_destroyed, optical_thumb, sar_thumb.
- E.3 Thumbnails: each tile's pre-event and post-event TIF is read, resized to 64×64,
     saved as optimised PNG, and base64-encoded as a UTF-8 string column in the parquet.
     Enables the dashboard to display images without any file I/O.
- E.4 Output: data/inference_results.parquet (~10–15 MB). Safe to commit to git.

Module F: Streamlit Dashboard
- F.1 Data source: reads only data/inference_results.parquet. No model weights, no
     rasterio, no PyTorch required at dashboard runtime.
- F.2 Tab 1 — Ablation Results:
     KPI row: val tiles, multimodal macro F1, optical macro F1, SAR delta F1, accuracy delta.
     Per-class F1 grouped bar chart (multimodal vs optical-only, 3 classes).
     Side-by-side confusion matrices (normalised by true label, Plotly heatmap).
     True label distribution bar chart.
- F.3 Tab 2 — Tile Gallery:
     Sidebar filters: split (Val/Train/All), true label multiselect, model choice,
     misclassified-only checkbox.
     4-column grid, 24 tiles per page, with prev/next page navigation.
     Each card: optical thumbnail, SAR thumbnail, true label badge (colour-coded),
     predicted label badge, correct/incorrect indicator.
- F.4 Tab 3 — Tile Inspector:
     Tile ID selectbox filtered to current split selection.
     Optical and SAR thumbnails side by side.
     Dual grouped bar chart: confidence scores for both models on the selected tile.
- F.5 Metrics: all F1 and accuracy values computed at runtime from parquet columns using
     sklearn, ensuring dashboard metrics always reflect the actual parquet data.
- F.6 Deployment: requirements-streamlit.txt lists only streamlit, pandas, pyarrow,
     plotly, scikit-learn, numpy — no GPU or GeoTIFF dependencies.


6. NON-FUNCTIONAL REQUIREMENTS

- Performance: Dashboard loads from parquet in < 3 seconds. Training 20 epochs on 772
  tiles takes ~60–90 minutes on Apple MPS (M-series chip). Export script (884 tiles,
  both models) completes in ~10–15 minutes on MPS.
- Resilience: Corrupt TIF files (from partial CDN downloads) are caught by _is_valid_tif()
  during dataset init and skipped. Training never crashes mid-epoch due to bad files.
  CDN URL expiry is caught and refreshed. Transient network drops are retried automatically.
- Code Quality: Modular structure (src/data/, src/models/, src/ui/, scripts/).
  No emoji in any Python source file. All decisions documented in ADRs before coding.
- Reproducibility: Label derivation thresholds, class weights, optimizer hyperparameters,
  and split files are all fixed and documented. Checkpoint filenames encode event + model type.
- Version Control: data/processed/ (8 GB TIFs) and outputs/*.pt (719 MB) are gitignored.
  Committed: src/, scripts/, docs/, data/.cd_cache/ (500 KB JSON), pyproject.toml, README.
- Documentation: 6 ADRs cover every major design decision. ARCHITECTURE.md has Mermaid
  diagrams for Phase 1 model, Phase 3 extension, operational pipeline, and phase roadmap.


7. DATA STRATEGY

Data Dictionary (Key Fields):

| Field              | Type    | Description                                              | Example                          |
|--------------------|---------|----------------------------------------------------------|----------------------------------|
| tile_id            | string  | Unique tile identifier including event prefix            | "turkey-earthquake_00000384"     |
| true_label         | int     | Derived tile class: 0=Intact, 1=Damaged, 2=Destroyed    | 2                                |
| pred_multimodal    | int     | Multimodal model predicted class                         | 2                                |
| pred_optical       | int     | Optical-only model predicted class                       | 0                                |
| conf_mm_destroyed  | float   | Multimodal model confidence for Destroyed class          | 0.712                            |
| optical_thumb      | string  | Pre-event 64×64 RGB PNG, base64-encoded                 | "iVBOR..."                       |
| sar_thumb          | string  | Post-event 64×64 grayscale PNG, base64-encoded          | "iVBOR..."                       |
| split              | string  | Dataset split the tile belongs to                        | "val"                            |

Label Derivation Rules (_derive_tile_label):

| Class     | Rule                                          | Rationale                                           |
|-----------|-----------------------------------------------|-----------------------------------------------------|
| Destroyed | destroyed_px / building_px >= 1%             | BRIGHT SAR: IoU >70% for destroyed — low threshold safe |
| Damaged   | damaged_px  / building_px >= 5%             | Damaged harder to detect in SAR (IoU <20%) — higher threshold reduces noise |
| Intact    | neither threshold met                         | Default for well-surveyed areas with minor damage  |
| Skip      | building_px < 200                            | ~1 px BRIGHT registration error corrupts sparse tiles |

Threshold History:
- Original (paper defaults): Destroyed 5%, Damaged 20%
- Morocco empirical revision: Destroyed 1%, Damaged 5%
  (Morocco training had 0 Destroyed examples at paper thresholds — model never learned the class)
- Current defaults apply to all events; per-event calibration planned for Phase 1.5

Data Pipeline:
1. download_events.py: selective range-request fetch → data/processed/{event}/pre-event|post-event|target/
2. BRIGHTDataset._load_split: reads split .txt, validates 3 files per tile, derives label,
   skips corrupt/sparse tiles
3. torchvision transforms: resize to 224×224, ToTensor, Normalize
4. CrossEntropyLoss training with class weights; val macro F1 gating
5. export_inference.py: load checkpoints → parquet with base64 thumbnails


8. TECHNICAL ARCHITECTURE

+------------------------------------------------------------------+
|                STREAMLIT DASHBOARD (3 tabs)                       |
|  Ablation Results | Tile Gallery | Tile Inspector                 |
|  (reads inference_results.parquet — no model or TIF at runtime)   |
+------------------------------------------------------------------+
                            |
         +------------------+-------------------+
         |                                      |
+--------+--------+                   +---------+---------+
| export_inference.py              |  | train_model.py              |
| Loads both .pt checkpoints        |  | MODEL_TYPE=multimodal       |
| Runs inference on all tiles        |  | MODEL_TYPE=optical_only     |
| Generates 64×64 thumbnails         |  | CosineAnnealingLR / AdamW   |
| Saves parquet (~10-15 MB)          |  | Macro F1 + gate evaluation  |
+------------------------------------+  +-----------------------------+
                                                     |
                               +---------------------+
                               |
              +----------------+----------------+
              |                                 |
+-------------+----------+    +-----------------+-------+
| MultimodalDamageCNN    |    | SingleModalDamageCNN    |
| Optical BranchCNN (3ch)|    | Optical BranchCNN (3ch) |
| SAR BranchCNN     (1ch)|    | (or SAR for sar_only)   |
| Late fusion: concat→FC |    | FC: 256→128→3           |
| 2,611,459 params       |    | 1,273,251 params        |
+------------------------+    +-------------------------+
              |
+-------------+-------------+
| BRIGHTDataset              |
| _derive_tile_label()       |
| _is_valid_tif()            |
| rasterio GeoTIFF reader    |
| torchvision transforms     |
+-------------+-------------+
              |
+-------------+-------------------------------------+
| data/processed/{event}/                            |
|   pre-event/  {tile_id}_pre_disaster.tif  (optical)|
|   post-event/ {tile_id}_post_disaster.tif (SAR)    |
|   target/     {tile_id}_building_damage.tif (mask) |
| Downloaded by download_events.py                   |
| (gitignored — 8 GB; CDN cache committed at 500 KB) |
+----------------------------------------------------+

              +-------------------------------------------+
              | download_events.py                         |
              | HTTP Range → zip64 EOCD → central dir      |
              | CDN refresh (403) + backoff retry (5×)     |
              | CD cache: data/.cd_cache/*.json (committed) |
              | HuggingFace: Kullervo/BRIGHT               |
              +-------------------------------------------+


9. IMPLEMENTATION PLAN

| Sprint | Duration | Deliverables                                                         | Status    |
|--------|----------|----------------------------------------------------------------------|-----------|
| 1      | 1 day    | Morocco download (download_morocco.py), BRIGHTDataset loader,        | Completed |
|        |          | baseline_model.py (BranchCNN + MultimodalDamageCNN), smoke-test       |           |
| 2      | 1 day    | Morocco training — revealed threshold issue (0 Destroyed examples)   | Completed |
|        |          | at paper defaults 5%/20%; training confirmed pipeline end-to-end     |           |
| 3      | 0.5 days | Threshold calibration: Destroyed 5%→1%, Damaged 20%→5% (ADR-002     | Completed |
|        |          | amendment); Morocco retrain shows all 3 classes present but macro    |           |
|        |          | F1 stuck at 0.30 (all-Intact predictions) — insufficient training data|           |
| 4      | 1 day    | Multi-event downloader (download_events.py): zip64 EOCD, CDN URL    | Completed |
|        |          | expiry handling (CDNExpired exception + force_refresh), exponential  |           |
|        |          | backoff retry, central directory disk cache                          |           |
| 5      | 2 days   | Turkey earthquake download (1,114 tiles): hit CDN rate limiting      | Completed |
|        |          | (received 0 bytes, expected 65536 EOCD tail). Fixed: 8 KB tail       |           |
|        |          | (was 64 KB); CD cache prevents repeat EOCD reads. 1,109+/1,114      |           |
|        |          | tiles successfully downloaded across 3 sessions                      |           |
| 6      | 1 day    | Zip64 local_hdr_offset bug: original code read vals[2] assuming      | Completed |
|        |          | fixed 3-field array. Zip64 fields are conditional — only present     |           |
|        |          | when 32-bit placeholder = 0xFFFFFFFF. Fixed: sequential foff pointer |           |
|        |          | (recovered 327 previously-zero-offset entries in Morocco pre-event)  |           |
| 7      | 1 day    | Multimodal training: fixed sklearn f1_score labels bug (reported     | Completed |
|        |          | 0.4894 averaging 2 classes instead of 0.3262 for 3 classes);        |           |
|        |          | MODEL_TYPE env var; checkpoint naming collision fix                  |           |
| 8      | 0.5 days | Full Turkey ablation: multimodal macro F1=0.4091 (epoch 5 best),    | Completed |
|        |          | optical-only macro F1=0.3924 (epoch 5 best). SAR delta +0.0167.     |           |
|        |          | Gate not passed (+0.017 vs +0.05 required). Damaged class F1=0.00   |           |
|        |          | for both (6 val samples — class imbalance confirmed as blocker)      |           |
| 9      | 0.5 days | _is_valid_tif() corruption guard: reads 1×1 pixel window (not just  | Completed |
|        |          | header) to catch truncated-data TIFs from mid-transfer CDN drops    |           |
| 10     | 1 day    | Streamlit deployment pipeline: export_inference.py (inference        | Completed |
|        |          | parquet with base64 thumbnails), dashboard.py rewrite (3 tabs,      |           |
|        |          | reads parquet only), requirements-streamlit.txt for cloud deploy     |           |
| 11     | 0.5 days | GitHub: .gitignore (excludes data/processed/, outputs/*.pt, BRIGHT/),| Completed |
|        |          | initial commit (23 files, 10,078 insertions), push to               |           |
|        |          | sabrinapribadi/multimodal-damage-assessment                          |           |


10. TESTING STRATEGY

- Zip64 parsing: manually verified local_hdr_offset against hex dump for Morocco pre-event
  entries that previously yielded 0-byte files; confirmed 327 tiles now extract correctly.
- Macro F1 accuracy: cross-checked sklearn f1_score output with and without explicit labels
  parameter on a val set with only 2 active classes — confirmed inflation from 0.33 to 0.49.
- Corruption guard: tested _is_valid_tif() against known truncated files (race condition
  from simultaneous download + training) — confirmed skipping without crash.
- CDN retry: confirmed CDNExpired is raised on 403, URL is refreshed, and tile is retried
  — verified across multiple Turkey download sessions.
- Dashboard: manual QA of all 3 tabs on Python 3.12 / macOS; parquet loading confirmed
  with and without pre-existing parquet file (graceful error message shown if absent).
- Classification report: confirmed classification_report with labels=list(range(3)) and
  target_names=[...] does not raise ValueError when val set has fewer active classes.


11. RISKS AND MITIGATIONS

| Risk                              | Mitigation                                                  | Status   |
|-----------------------------------|-------------------------------------------------------------|----------|
| HuggingFace CDN URL expiry        | CDNExpired exception; force_refresh on _get_cdn;            | Resolved |
|   (HTTP 403 mid-download)         | catch-refresh-retry in download loop                        |          |
| Transient CDN drops               | 5-attempt exponential backoff (1→2→4→8→16s);               | Resolved |
|   ("peer closed connection")      | graceful SKIP with reason logged                            |          |
| CDN throttling on large reads     | EOCD tail reduced from 65 KB to 8 KB; CD cached to disk    | Resolved |
|   (0 bytes received on EOCD)      | so subsequent runs never re-read the EOCD                  |          |
| Zip64 wrong field index           | Sequential foff pointer (not assumed positional array);     | Resolved |
|   (local_hdr_offset = 0)          | only advances for fields whose 32-bit placeholder = 0xFFFF  |          |
| Corrupt tiles from partial download| _is_valid_tif() reads 1×1 pixel window; corrupt tiles       | Resolved |
|   crashing training mid-epoch     | skipped during _load_split, never hit __getitem__           |          |
| sklearn macro F1 inflation        | labels=list(range(NUM_CLASSES)) on all f1_score and         | Resolved |
|   (only averages present classes) | classification_report calls                                 |          |
| Morocco class imbalance           | Confirmed root cause: insufficient training tiles for        | Resolved |
|   (0 Destroyed training examples) | minority classes. Resolution: switch to Turkey (1,114 tiles)|          |
| Threshold too strict for Morocco  | Revised Destroyed 5%→1%, Damaged 20%→5% after empirical    | Resolved |
|   (0 examples survive 5%/20%)    | survey of building pixel fractions                          |          |
| MODEL_TYPE hardcoded             | Replaced with os.environ.get("MODEL_TYPE", "multimodal");   | Resolved |
|   (optical_only not loading)      | import os added to train_model.py                           |          |
| Checkpoint naming collision       | best_{run_name}_{MODEL_TYPE}.pt — both type and event       | Resolved |
|   (both models overwrite same .pt)| encoded in filename                                         |          |
| Dashboard needs model/TIF at      | export_inference.py pre-computes everything; parquet with   | Resolved |
|   runtime (not deployable)        | base64 thumbnails; requirements-streamlit.txt for Cloud     |          |
| Freeze/unfreeze patience logic    | Phase 3b attempt 1: patience_counter not reset at unfreeze  | Resolved |
|   (early stop before backbone     | caused frozen-phase stagnation to exhaust all 7 patience   |          |
|   ever trained)                   | slots before epoch 9. Fix: only count patience after        |          |
|                                   | backbone unfrozen; reset to 0 at unfreeze point.            |          |
| Ablation gate not passed          | Final state: best SAR delta +0.022 (Phase 3). Gate requires | Final    |
|   (all phases)                    | +0.050. Remaining levers (soft labels, Morocco) made it     |          |
|                                   | worse. Gate requires pixel-level supervision or ChangeMamba. |          |


12. SUCCESS CRITERIA

Download pipeline (all achieved):
- Selective download fetches only the event's tiles (<5% of zip size)
- CDN URL expiry handled transparently with auto-refresh
- Central directory cached to disk; subsequent runs skip EOCD entirely
- Resume-safe: re-runs pick up from where previous run stopped

Training pipeline (all achieved):
- Multimodal and optical-only models train from a single script via MODEL_TYPE env var
- Macro F1 computed correctly with explicit class labels (no inflation)
- Checkpoints disambiguated by event + model type
- Classification report includes all 3 classes even when absent from val batch

Ablation results (Phase 1):
- Positive SAR signal confirmed: multimodal F1 0.4091 > optical-only F1 0.3924 (+0.017)
- Gate (+0.05) not passed — documented with diagnosis (Damaged class imbalance)
- Destroyed recall = 0.78 for both models — backbone learning structural features

Dashboard (all achieved):
- Loads from parquet in < 3 seconds; no model or TIF files needed at runtime
- All 3 tabs functional: ablation results, tile gallery (paginated + filtered), tile inspector
- Deployable to Streamlit Cloud with requirements-streamlit.txt (no GPU dependencies)

Documentation (all achieved):
- 6 ADRs covering all major architecture decisions
- ARCHITECTURE.md with Mermaid diagrams for Phase 1, Phase 3, operational pipeline, roadmap
- PRD and README in sync with implemented state


13. KEY INSIGHTS FROM DATA

Phase 1 — Turkey Earthquake only:
- Total tiles (on disk): 1,109 / 1,114 (99.6% complete)
- Train: 772 tiles | Val: 112 tiles | Val distribution: Intact 74 / Damaged 6 / Destroyed 32
- SAR signal: +0.017 F1 lift (multimodal 0.4091 vs optical 0.3924)
- Destroyed recall = 0.78 (both models) — SAR backscatter is learnable on homogeneous data
- Damaged class: 6 val samples, F1=0.00 both models. Primary driver of macro F1 below gate.

Phase 1.5 — Turkey Earthquake + Beirut Explosion:
- Beirut label distribution: Intact 88.7% / Damaged 6.0% / Destroyed 5.3%
  (localized port explosion, not widespread earthquake — hypothesis of 15-20% Damaged was wrong)
- Combined: Train 861 tiles | Val 129 tiles | Test 248 tiles
- Multimodal val F1: 0.4207 | Optical-only val F1: 0.4432 | SAR delta: −0.023
- SAR hurts on mixed-event-type data: Beirut explosion backscatter fundamentally different
  from earthquake rubble. Custom CNN cannot generalise the SAR feature extractor.

Phase 2b — Turkey Earthquake + Noto Earthquake (ResNet-18 pretrained backbone):
- Noto label distribution: Intact 20.0% / Damaged 8.9% / Destroyed 71.1% (45 labelled tiles)
  (2024 Noto peninsula earthquake — widespread structural collapse, few partial damages)
- Combined: Train 824 tiles | Val 121 tiles | Test 266 tiles
- Multimodal (multimodal_v2) val F1: 0.597 | Optical-only (optical_only_v2) val F1: 0.628
- SAR delta: −0.031 (gate not passed)
- Per-class SAR delta:
    Intact:    +0.009 (marginal, noise level)
    Damaged:   −0.167 (SAR actively misleads — ambiguous backscatter for partially standing)
    Destroyed: +0.064 (SAR provides clear signal — rubble scattering consistent across events)
- Key finding: SAR is class-conditionally useful, not universally harmful or beneficial.
  The macro gate fails because macro averaging weights the Damaged class loss (−0.167)
  equally with the Destroyed gain (+0.064). The signal is real but needs class-aware fusion.
- Test macro F1: multimodal 0.607 vs optical 0.479 — multimodal wins on test set despite
  losing on val. This is the first phase where multimodal outperforms optical on test data.
- Early stopping: multimodal_v2 at epoch 6; optical_only_v2 at epoch 7 (of 20 configured)


Phase 3 — Turkey Earthquake + Noto Earthquake (ResNet-18 + Optical-Gated SAR):
- Train: 802 tiles (Turkey 772 + Noto 30) | Val: 118 tiles | Test: 231 tiles
- FREEZE_EPOCHS=5 (head only), then full backbone at LR×0.1 with patience reset
- Augmentation: paired hflip/vflip/rotation (both modalities) + optical-only colour jitter
- Loss: CrossEntropyLoss(weights=[1,5,10]) + Lovász-Softmax surrogate IoU
- Multimodal val F1: 0.580 | Optical-only val F1: 0.558 | SAR delta: +0.022
- Per-class SAR delta: Intact +0.038, Damaged +0.047, Destroyed −0.019
- Key finding: optical-gated scalar gate (α = sigmoid(Linear(opt_feat→1))) prevents circular
  conditioning — optical features decide SAR trust. Gate correctly suppresses SAR noise on
  Damaged (ambiguous partially-standing) tiles. First positive SAR delta on two-event run.
  Phase 3 is the best result across all phases. Ablation gate (+0.050) not cleared.
- XAI: Grad-CAM dual-branch heatmaps (pytorch-grad-cam) integrated into Tile Inspector;
  opt_gradcam_thumb and sar_gradcam_thumb columns added to inference parquet.

Phase 4 — Turkey + Noto + Morocco Earthquakes (soft pixel-distribution labels):
- Additional event: morocco-earthquake (2023 High Atlas Mountains, rural mud-brick construction)
- Morocco tiles: 567 on disk; 146 training-eligible (391 filtered: sparse building coverage)
- Train: 948 tiles | Val: 142 tiles | Test: 278 tiles
- Soft labels: _derive_soft_label() returns pixel fraction vector [p_intact, p_damaged, p_destroyed]
  instead of hard majority-vote label. Loss: weighted soft-CE + Lovász.
- Multimodal val F1: 0.396 | Optical-only val F1: 0.492 | SAR delta: −0.096
- Notable: optical-only learned Damaged class for first time (F1=0.200) with Morocco data
- Key finding: SAR domain shift across construction types breaks fusion. Morocco mud-brick
  collapse produces different backscatter signatures than Turkish RC or Japanese residential
  structures. Optical features are more construction-type-agnostic; SAR is not.
  Optical improved; multimodal degraded — the SAR branch is more sensitive to domain shift.

Phase 3b — Turkey + Noto only (soft labels, freeze=8, backbone LR×0.05):
- Same events as Phase 3 (Morocco removed to isolate soft-label effect)
- FREEZE_EPOCHS=8, patience only counts after unfreeze, backbone LR×0.05
- Multimodal val F1: 0.283 | Optical-only val F1: 0.345 | SAR delta: −0.114
- Key finding: soft pixel-distribution labels alone hurt both models. Most tiles are >80%
  dominated by one class — soft labels flatten the gradient for these cases, producing a
  shallower loss landscape and slower/weaker convergence. Hard one-hot labels are better
  for tile-level classification. Soft labels would be appropriate for pixel-level tasks.
- Secondary finding: freeze/unfreeze patience bug discovered and fixed. Not resetting
  patience_counter at unfreeze allows frozen-phase stagnation (7+ epochs with identical F1)
  to exhaust all patience before backbone ever trains. Correct design: patience counts only
  in unfrozen phase; reset to 0 at unfreeze.


14. LESSONS LEARNED — PHASE 1

1. The pipeline works end-to-end on Apple MPS without GPU. Selective range-request
   downloading, corruption-guarded data loading, custom CNN training, and Streamlit
   deployment all function correctly. Hardware constraint became a design asset.

2. The falsification gate correctly identified insufficient data. The ablation gate
   (+0.05 F1) was not passed. The root cause was diagnosable: 6 Damaged val samples
   makes the Damaged class F1 undefined for both models, correctly driving macro F1
   below the gate threshold. The framework worked as designed.

3. The Damaged class is the hardest to classify. Intact (structurally stable) and
   Destroyed (rubble signature) have distinct visual and SAR signatures. Damaged buildings
   are partially standing with subtle SAR shadow changes — ambiguous at 512×512 tile
   resolution. Class imbalance compounds this: only 6 val examples out of 112.

4. SAR provides a real but small signal. +1.7 pp macro F1 at epoch 5, reproduced across
   runs. Destroyed recall = 0.78 for both models — the backbone learns backscatter loss
   from structural collapse. The signal exists; a stronger backbone and more balanced data
   will amplify it.

5. Pre-computed parquet is the right deployment architecture for a portfolio demo. The
   dashboard loads from a committed file with 5 pure-Python dependencies. Zero inference
   overhead, no rasterio, no GPU. Any reviewer can run it in under 60 seconds with pip install.

6. SAR signal is class-conditional, not universally present or absent. Phase 2b (ResNet-18,
   two earthquakes) reveals the clearest finding yet: SAR helps Destroyed (+6.4 pp) and hurts
   Damaged (−16.7 pp). The macro gate fails because macro averaging treats these equally.
   The right next question is class-conditional fusion, not whether SAR helps overall.

7. Test set generalisation favours multimodal. Phase 2b multimodal test F1 = 0.607 vs
   optical test F1 = 0.479 — a +12.8 pp advantage on held-out data despite trailing on val
   (−3.1 pp). Multimodal overfits less: the SAR branch regularises the optical branch by
   forcing the fusion head to represent both modalities.

8. Early stopping is essential. Phase 2b optimal epochs: multimodal at 6, optical at 7
   (out of 20 configured). Best checkpoints consistently arrive early; patience=5 prevented
   wasted compute without stopping on epoch-to-epoch noise.

9. The gate design for class-conditional fusion matters more than the backbone.
   Phase 2b simple concat (SAR decides its own trust): −0.031. Phase 3 optical-only gate
   (optical decides SAR trust): +0.022. Same backbone, same events, same data. The circular
   conditioning problem — using SAR to gate SAR — was the critical flaw in prior designs.
   A 513-parameter Linear(512→1) gate conditioned on optical features alone changed the sign
   of the SAR delta and improved the Damaged class delta by 21.4 pp.

10. Soft labels are wrong for tile-level classification with dominant-class tiles.
    Pixel-distribution soft labels [0.85, 0.10, 0.05] provide richer supervision in theory
    but flatten gradients for tiles where one class is dominant (>80% of building pixels).
    Both models degraded in Phase 3b vs. Phase 3 hard-label baseline. Soft labels are
    appropriate for pixel-level (segmentation) tasks where the label itself is a distribution;
    at tile level they introduce unnecessary ambiguity. This is a granularity mismatch.

11. SAR backscatter is less domain-transferable than optical representations.
    Phase 4 added Morocco (mud-brick) to Turkey (RC) + Noto (Japanese residential).
    Optical-only improved (learned Damaged class for first time: F1=0.200).
    Multimodal degraded severely (SAR delta −0.096). The optical backbone learned
    construction-type-agnostic damage patterns; the SAR branch did not. Collapsed mud-brick
    produces fundamentally different backscatter than collapsed RC. Cross-construction-type
    SAR fusion requires explicit domain adaptation or event-type conditioning.

12. Freeze/unfreeze patience must be decoupled from frozen-phase stagnation.
    During the frozen phase, the head has no backbone gradient signal — identical F1 across
    8 frozen epochs is expected behaviour, not an early-stopping signal. Phase 3b attempt 1
    (patience not reset at unfreeze) caused early stopping at epoch 8, exactly at the
    freeze boundary, before the backbone ever received a gradient. Correct design:
    patience_counter is reset to 0 at unfreeze AND only increments after backbone unfreezes.
    This was implemented and verified in Phase 3b final run.

13. +2.2 pp is the best SAR delta achievable with the current architecture.
    Across 6 experimental configurations (Phase 1, 1.5, 2b, 3, 4, 3b), Phase 3 achieved
    the global optimum. The ablation gate (+5.0 pp) was not cleared. Research conclusion:
    SAR adds marginal, class-conditional, domain-sensitive value at tile level with ResNet-18.
    Clearing the gate likely requires: (a) pixel-level supervision with UNet decoder,
    (b) explicit temporal change encoding (ChangeMamba), or (c) domain-specific pre-training
    on earthquake-only SAR imagery. These require cloud GPU compute beyond current scope.


15. APPENDIX

- Data Source: https://huggingface.co/datasets/Kullervo/BRIGHT
- GitHub Repository: https://github.com/sabrinapribadi/multimodal-damage-assessment
- Tech Stack: PyTorch, rasterio, BRIGHTDataset, httpx, huggingface_hub, Streamlit,
  Plotly, scikit-learn, Pandas, pyarrow, Pillow
- Models: MultimodalDamageCNN (2.6M params), SingleModalDamageCNN (1.3M params)
- Phase Roadmap (all tile-level phases complete):
    Phase 1  (complete): Custom 4-layer CNN, Turkey — SAR delta +0.017, gate not passed
    Phase 1.5 (complete): Custom CNN, Turkey + Beirut — SAR delta −0.023, cross-event-type failure
    Phase 2b (complete): ResNet-18, Turkey + Noto — SAR delta −0.031 macro, Destroyed +0.064
    Phase 3  (complete): ResNet-18 + optical gate + Lovász + augmentation — SAR delta +0.022 (BEST)
    Phase 4  (complete): Turkey + Noto + Morocco + soft labels — SAR delta −0.096 (domain shift)
    Phase 3b (complete): Turkey + Noto + soft labels only — SAR delta −0.114 (soft labels hurt)
    Phase 5  (deferred): ChangeMamba / pixel-level UNet — needs CUDA cloud GPU
- ADR Index (see docs/adr/README.md for full index with decision dependency graph):
    ADR-001: Tile classification as stepping stone to pixel segmentation
    ADR-002: Area-weighted label derivation (1%/5% thresholds; Morocco amendment)
             Decision drove: _derive_tile_label() thresholds; Morocco→Turkey switch
    ADR-003: Macro F1 + relative improvement gate (+0.05)
             Decision drove: sklearn labels=list(range(3)) fix; falsification of Phase 1 gate
    ADR-004: ResNet-18 backbone (Phase 2); averaged first conv for SAR domain adaptation
    ADR-005: Spatial/global split; AdaptiveAvgPool2d as separate module for Phase 3 transfer
             Decision drove: BranchCNN.pool as standalone module; encode() returning all skips
    ADR-006: GeoJSON (primary) / PNG (secondary) / SMS (tertiary) output hierarchy
- Citation:
    Chen et al. (2025). BRIGHT: a globally distributed multimodal building damage assessment
    dataset with very-high-resolution for all-weather disaster response.
    Earth System Science Data, 17(11), 6217-6253. https://doi.org/10.5194/essd-17-6217-2025
