# System Architecture — Morocco Earthquake Damage Assessment

## Phase 1 Architecture (Current)

```mermaid
flowchart TD
    subgraph INPUT["Data Ingestion"]
        A1[Pre-event Optical\nGeoTIFF 512×512 RGB] --> L
        A2[Post-event SAR\nGeoTIFF 512×512 1-ch] --> L
        A3[Target Mask\nGeoTIFF 512×512\n0=bg,1=intact,2=dmg,3=destr] --> LABEL
        LABEL[_derive_tile_label\n5% destroyed threshold\n20% damaged threshold\nmin 200 building px] --> L
        L[BRIGHTDataset\n567 Morocco tiles\nstandard_ML split]
    end

    subgraph MODEL["MultimodalDamageCNN — Phase 1"]
        direction LR
        OPT[Optical Branch\nBranchCNN\nin_ch=3] --> |"(B, 256)"| FUSE
        SAR[SAR Branch\nBranchCNN\nin_ch=1] --> |"(B, 256)"| FUSE
        FUSE[Concat → Linear 512→256\n→ ReLU → Dropout\n→ Linear 256→3] --> OUT
        OUT["Logits (B, 3)\n0=Intact 1=Damaged 2=Destroyed"]
    end

    subgraph BRANCH["BranchCNN Detail"]
        B1["block1: Conv→BN→ReLU×2 → MaxPool\n(B, 32, 112, 112)  ← skip s1"]
        B2["block2: Conv→BN→ReLU×2 → MaxPool\n(B, 64,  56,  56)   ← skip s2"]
        B3["block3: Conv→BN→ReLU×2 → MaxPool\n(B, 128, 28,  28)   ← skip s3"]
        B4["block4: Conv→BN→ReLU×2 → MaxPool\n(B, 256, 14,  14)   ← bottleneck s4"]
        GP["global_pool: AdaptiveAvgPool2d(1)\n(separate module — not fused)"]
        B1 --> B2 --> B3 --> B4 --> GP
    end

    subgraph TRAIN["Training"]
        T1["CrossEntropyLoss\nweights=[1.0, 5.0, 10.0]\nIntact / Damaged / Destroyed"]
        T2["AdamW lr=1e-3, wd=1e-4\nCosineAnnealingLR"]
        T3["Best checkpoint:\nval Macro F1 — not val loss"]
    end

    subgraph VALIDATE["Validation Gate"]
        V1["Run optical_only model"]
        V2["Run multimodal model"]
        V3{"F1_multimodal >\nF1_optical + 0.05?"}
        V3 --> |Yes| PASS["Signal validated\n→ Phase 2: ResNet-18 backbone"]
        V3 --> |No| DIAG["Diagnose:\na) weak SAR signal → go single-modal\nb) noisy labels → revisit ADR-002\nc) capacity → increase model size"]
    end

    L --> MODEL
    MODEL --> TRAIN
    TRAIN --> VALIDATE
```

---

## Phase 3 Extension (Future — Surgical Addition)

```mermaid
flowchart TD
    subgraph ENCODER["Encoder — Reused from Phase 1"]
        OE["Optical BranchCNN\nencode() → s1,s2,s3,s4"]
        SE["SAR BranchCNN\nencode() → s1,s2,s3,s4"]
    end

    subgraph DECODER["Decoder — New in Phase 3"]
        D1["Upsample s4 → 28×28\nCat with s3 → ConvBlock"]
        D2["Upsample → 56×56\nCat with s2 → ConvBlock"]
        D3["Upsample → 112×112\nCat with s1 → ConvBlock"]
        D4["Upsample → 224×224\nConv1×1 → 3 classes"]
        D1 --> D2 --> D3 --> D4
    end

    subgraph HEADS["Dual Heads"]
        CLF["Classification Head\n(unchanged from Phase 1)"]
        SEG["Segmentation Head\nY_loc: building mask\nY_clf: per-building damage"]
        FUSE["Y_dam = Y_loc ⊙ Y_clf"]
    end

    ENCODER --> DECODER
    ENCODER --> CLF
    DECODER --> SEG
    SEG --> FUSE
    CLF --> FUSE
```

---

## End-to-End Operational Pipeline (Phase 2+)

```
  [Capella/Umbra SAR overpass]           [Pre-event optical archive]
           │                                        │
           ▼                                        ▼
   GeoTIFFTiler                           GeoTIFFTiler
   ─ sliding window 512×512              ─ aligned to SAR extent
   ─ overlap=64 for edge continuity      ─ same tile grid
   ─ despeckle → dB → 3σ clip → norm     ─ percentile stretch → norm
   ─ preserve (minx,miny,maxx,maxy)      ─ preserve geobounds
           │                                        │
           └──────────────┬─────────────────────────┘
                          ▼
              MultimodalDamageCNN
              ─ batch_size=32 (MPS)
              ─ ~42 seconds for 567 tiles
              ─ outputs: class + confidence per tile
                          │
           ┌──────────────┼──────────────────┐
           ▼              ▼                  ▼
      GeoJSON          PNG map          SMS priority list
  (primary output)  (secondary)          (tertiary)
   QGIS / ArcGIS     WhatsApp         Field responders
   Google Earth       email           without internet
```

---

## Phase Roadmap

| Phase | Model | Data | What It Proves |
|-------|-------|------|----------------|
| **1 (current)** | Custom 4-layer CNN, random init | Morocco (567 tiles) | Pipeline works; multimodal > optical-only? |
| **1.5** | Custom CNN, random init | Full BRIGHT (14 events, ~4246 tiles) | Can the model generalize across events? |
| **2** | ResNet-18, pretrained optical / averaged-conv SAR | Full BRIGHT | Multimodal signal isolated from architecture confound |
| **3** | ResNet-18 + UNet decoder | Full BRIGHT | Pixel-level segmentation; comparable to paper mIoU |
| **4** | DamageFormer / ChangeMamba | Full BRIGHT | Match paper benchmark; production candidate |

---

## Key ADR Decisions in One View

| Decision | Choice | Alternative Rejected |
|----------|--------|---------------------|
| Task type | Tile classification (stepping stone) | Direct segmentation |
| Tile label | Area-weighted: Destroyed≥5%, Damaged≥20%, min 200px | max() — too noisy |
| Metric | Macro F1 + relative improvement gate | Accuracy — majority-class bias |
| SAR backbone | ResNet-18, averaged first conv | Repeat 1→3 — domain mismatch |
| Encoder design | Spatial/global split; AdaptiveAvgPool2d separate | Fused pool (one-way door) |
| LLM stack | Removed (Phase 4, after signal validated) | Keep as placeholder |
| Output format | GeoJSON primary, PNG secondary, SMS tertiary | Streamlit-only |
