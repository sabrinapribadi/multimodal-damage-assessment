# Multimodal Building Damage Assessment

An end-to-end deep learning system for satellite-based building damage classification after
natural disasters. Fuses pre-event optical imagery with post-event SAR data in a dual-branch
CNN, trained and evaluated on the BRIGHT dataset. Includes an HTTP range-request downloader,
ablation training pipeline, and a Streamlit dashboard powered by pre-computed inference
(no model weights or GeoTIFF files needed at runtime).

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-BRIGHT-00BFFF?style=for-the-badge)](https://huggingface.co/datasets/Kullervo/BRIGHT)

## Features

### Dashboard (3 tabs — no model or TIF files required at runtime)

| Tab | What it shows |
|-----|--------------|
| **Ablation Results** | KPI row (val tiles, multimodal macro F1, optical F1, SAR delta, accuracy); per-class F1 grouped bar chart; side-by-side confusion matrices (multimodal vs optical-only); true label distribution chart |
| **Tile Gallery** | Paginated 4-column grid (24 tiles/page) of optical + SAR thumbnail pairs; filterable by split, true label, model, and correct/misclassified only; per-tile true vs predicted label badge |
| **Tile Inspector** | Select any tile by ID; view optical and SAR thumbnails side by side; dual confidence bar chart comparing multimodal and optical-only predictions on the same tile |

### Download Pipeline

Selective HTTP range-request downloader — reads only the zip central directory from
HuggingFace, then fetches only the tiles needed for a given event. Never downloads the
full 9.9 GB pre-event.zip or 3.3 GB post-event.zip.

- Zip64 EOCD and central directory parsing with sequential field reading (per ZIP spec §4.5.3)
- CDN pre-signed URL refresh on HTTP 403 (HuggingFace URL expiry)
- Exponential-backoff retry (5 attempts: 1s → 2s → 4s → 8s → 16s) for transient drops
- Central directory cached to `data/.cd_cache/` — subsequent runs skip the EOCD read entirely
- Resume-safe: already-downloaded tiles are skipped

### Training Pipeline

- Dual ablation via `MODEL_TYPE` environment variable: `multimodal` (optical+SAR) or `optical_only`
- Multi-event support: pass multiple event names to train on a `ConcatDataset`
- Area-weighted tile label derivation from pixel-level masks (configurable thresholds)
- CrossEntropyLoss with class weights `[1.0, 5.0, 10.0]` (Intact / Damaged / Destroyed)
- Macro F1 with explicit `labels=list(range(3))` to avoid sklearn class-presence bias
- Best checkpoint saved by val macro F1 (not val loss)

## Model Performance — Turkey Earthquake (Phase 1)

| Metric | Multimodal (Optical + SAR) | Optical-only |
|--------|---------------------------|--------------|
| **Val Macro F1** | **0.4091** | 0.3924 |
| **Val Accuracy** | 0.61 | 0.58 |
| Intact F1 | 0.67 | 0.65 |
| Damaged F1 | 0.00 | 0.00 |
| Destroyed F1 | 0.56 | 0.53 |
| Parameters | 2,611,459 | 1,273,251 |
| SAR signal (ΔF1) | **+0.0167** | — |
| Train tiles | 772 | 772 |
| Val tiles | 112 | 112 |

> Phase 1 ablation gate: F1(multimodal) > F1(optical-only) + 0.05. Gate not yet passed (+0.017).
> Damaged class (6 val samples) drags macro F1 for both. Phase 1.5 will add Beirut explosion
> data (15–20% Damaged rate) to address the class imbalance.

## Project Structure

```
multimodal-damage-assessment/
├── .streamlit/
│   └── config.toml                    # Dark theme + coral accent
├── src/
│   ├── data/
│   │   └── brighT_loader.py           # BRIGHTDataset, _derive_tile_label, _is_valid_tif
│   ├── models/
│   │   └── baseline_model.py          # BranchCNN, MultimodalDamageCNN, SingleModalDamageCNN
│   └── ui/
│       └── dashboard.py               # Streamlit 3-tab dashboard (reads parquet only)
├── scripts/
│   ├── download_events.py             # Selective HuggingFace downloader (range requests)
│   ├── download_morocco.py            # Legacy Morocco-specific downloader
│   ├── train_model.py                 # Multi-event training with ablation support
│   └── export_inference.py            # Pre-compute inference → parquet with thumbnails
├── docs/
│   ├── ARCHITECTURE.md                # Mermaid diagrams: Phase 1 model, Phase 3 extension,
│   │                                  #   operational pipeline, phase roadmap
│   ├── BRIGHT_SUMMARY.md              # Dataset summary and key statistics
│   └── adr/
│       ├── ADR-001-task-framing.md    # Tile classification as stepping stone to segmentation
│       ├── ADR-002-tile-label-derivation.md  # Area-weighted thresholds + Morocco amendment
│       ├── ADR-003-validation-strategy.md    # Macro F1 + relative improvement gate
│       ├── ADR-004-backbone-and-sar-adaptation.md  # ResNet-18, averaged first conv
│       ├── ADR-005-encoder-architecture.md   # Spatial/global split, separate pool module
│       └── ADR-006-deployment-and-output.md  # GeoJSON/PNG/SMS output hierarchy
├── data/
│   ├── .cd_cache/                     # Zip central directory caches (committed, ~500 KB)
│   │   ├── post-event.zip.turkey-earthquake.json
│   │   ├── pre-event.zip.turkey-earthquake.json
│   │   └── target.zip.turkey-earthquake.json
│   └── inference_results.parquet      # Pre-computed predictions + thumbnails (committed)
├── outputs/                           # Model checkpoints (.pt — gitignored)
├── BRIGHT/                            # BRIGHT benchmark repo (gitignored — clone separately)
├── requirements-streamlit.txt         # Minimal deps for Streamlit Cloud deployment
└── pyproject.toml                     # Full Poetry dependency spec
```

## Setup

**Prerequisites:** Python 3.12, [Poetry](https://python-poetry.org/).

```bash
git clone https://github.com/sabrinapribadi/multimodal-damage-assessment.git
cd multimodal-damage-assessment
poetry install
```

## Data Setup

```bash
# Download Turkey earthquake tiles (~3.5 GB — only the Turkey slices, not the full zips)
python scripts/download_events.py turkey-earthquake

# Optional: add Beirut explosion data for Phase 1.5
python scripts/download_events.py turkey-earthquake beirut-explosion
```

The downloader reads the central directory from each HuggingFace zip via HTTP range requests
and fetches only the tiles belonging to the requested event. Central directory JSON caches
in `data/.cd_cache/` so subsequent runs skip the EOCD read and jump straight to missing tiles.

Event names available: `bata-explosion`, `beirut-explosion`, `congo-volcano`, `haiti-earthquake`,
`hawaii-wildfire`, `la`, `libya-flood`, `marshall-wildfire`, `mexico-hurricane`,
`morocco-earthquake`, `myanmar-hurricane`, `noto-earthquake`, `turkey-earthquake`, `ukraine-conflict`

## Training

```bash
# Multimodal (optical + SAR) — default
python scripts/train_model.py turkey-earthquake

# Optical-only ablation
MODEL_TYPE=optical_only python scripts/train_model.py turkey-earthquake

# Multi-event (Phase 1.5)
python scripts/train_model.py turkey-earthquake beirut-explosion
```

Training prints a per-epoch table with train loss/acc, val loss/acc, val macro F1, and
checkpoint saves. Best checkpoint is saved to `outputs/best_{events}_{model_type}.pt`.

## Exporting Inference & Running the Dashboard

```bash
# 1. Run once after training — generates the dashboard parquet (~10-15 MB)
python scripts/export_inference.py

# 2. Launch the dashboard (reads parquet only — no model weights or TIF files needed)
PYTHONPATH=. streamlit run src/ui/dashboard.py
```

Dashboard opens at `http://localhost:8501`.

### Streamlit Cloud Deployment

Commit `data/inference_results.parquet` and use `requirements-streamlit.txt` as the
requirements file. No GPU, no rasterio, no PyTorch needed on the cloud server.

```bash
git add data/inference_results.parquet
git commit -m "Add pre-computed inference parquet for Streamlit deployment"
```

## Tech Stack

| Layer | Library / Tool |
|-------|---------------|
| Deep Learning | PyTorch 2.x, torchvision |
| GeoTIFF I/O | rasterio |
| Data | NumPy, Pandas, pyarrow (parquet) |
| Frontend | Streamlit 1.35+, Plotly Graph Objects |
| Charts | Plotly Express / Graph Objects (dark theme) |
| Metrics | scikit-learn (F1, confusion matrix) |
| Download | httpx (async-capable), huggingface_hub |
| Image processing | Pillow |
| Language | Python 3.12 |
| Hardware | Apple MPS / CUDA / CPU (auto-detected) |

## Key Dataset Facts (BRIGHT — Turkey Earthquake Subset)

- **Total tiles:** 1,114
- **Train tiles (after label filtering):** 772
- **Val tiles (after label filtering):** 112
- **Val label distribution:** Intact 74 / Damaged 6 / Destroyed 32
- **Tile size:** 512×512 pixels (resized to 224×224 for training)
- **Pre-event:** Optical RGB GeoTIFF (3 channels)
- **Post-event:** SAR GeoTIFF (1 channel, intensity)
- **Target:** Pixel-level mask (0=background, 1=Intact, 2=Damaged, 3=Destroyed)
- **Tile label rule:** Destroyed if ≥1% building pixels destroyed; Damaged if ≥5%; Intact otherwise; skip if <200 building pixels

## Data Source

[BRIGHT Dataset — HuggingFace (Kullervo/BRIGHT)](https://huggingface.co/datasets/Kullervo/BRIGHT)

Chen et al. (2025). BRIGHT: a globally distributed multimodal building damage assessment
dataset with very-high-resolution for all-weather disaster response.
*Earth System Science Data*, 17(11), 6217–6253. https://doi.org/10.5194/essd-17-6217-2025
