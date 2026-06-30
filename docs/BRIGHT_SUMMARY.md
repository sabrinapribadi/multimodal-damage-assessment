# BRIGHT Dataset: Summary & Analysis

## Page 1: Executive Summary & Dataset Overview

### What is BRIGHT?

**BRIGHT** (Building damage assessment dataset using veRy-hIGH-resolution optical and SAR imagery) is the **first globally distributed, open-access multimodal dataset** for building damage assessment. It combines pre-event optical imagery with post-event Synthetic Aperture Radar (SAR) imagery to enable all-weather, day-and-night disaster response.

### Key Statistics

| Metric | Value |
|--------|-------|
| Total Disaster Events | 14 (5 natural + 2 man-made) |
| Geographic Coverage | 23 regions across 14 countries |
| Total Image Pairs | 4,246 multimodal tiles |
| Building Instances | 384,596 |
| Spatial Resolution | 0.3 - 1 meter/pixel |
| Disaster Types | Earthquakes, storms, wildfires, floods, volcanic eruptions, explosions, armed conflicts |

### Why This Matters

Traditional building damage assessment relies on optical imagery, which fails in:
- **Cloudy conditions** (storms, hurricanes)
- **Nighttime** (no solar illumination)
- **Smoke/haze** (wildfires)

**SAR imagery** penetrates clouds and works in darkness, making it ideal for rapid disaster response. BRIGHT is the first dataset to combine both modalities at scale.

---

## Page 2: Dataset Construction & Statistics

### Disaster Events Included

| Region | Disaster Type | Date | Buildings | Tiles |
|--------|--------------|------|-----------|-------|
| Beirut, Lebanon | Explosion | Aug 2020 | 25,496 | 133 |
| Bata, Equatorial Guinea | Explosion | Mar 2021 | 8,893 | 10 |
| Goma, DR Congo | Volcano | May 2021 | 18,741 | 123 |
| Les Cayes, Haiti | Earthquake | Aug 2021 | 18,918 | 73 |
| La Palma, Spain | Volcano | Sep 2021 | 30,239 | 93 |
| Boulder, USA | Wildfire | Dec 2021 | 8,365 | 77 |
| Ukraine | Armed Conflict | Mar 2022 | 56,770 | 513 |
| Turkey | Earthquake | Feb 2023 | 135,033 | 1,114 |
| Kyaukpyu, Myanmar | Cyclone | May 2023 | 8,052 | 126 |
| Maui, Hawaii | Wildfire | Aug 2023 | 3,995 | 65 |
| Morocco | Earthquake | Sep 2023 | 6,269 | 56 |
| Derna, Libya | Flood | Sep 2023 | 10,979 | 124 |
| Acapulco, Mexico | Hurricane | Oct 2023 | 18,437 | 212 |
| Noto, Japan | Earthquake | Jan 2024 | 8,153 | 79 |
| **TOTAL** | **14 events** | | **384,596** | **4,246** |

### Data Sources
- **Optical**: Maxar Open Data Program, Google Earth, NOAA, GSI Japan, IGN Spain
- **SAR**: Capella Space, Umbra Space (both CC-BY-4.0 licensed)

### Damage Categories

| Label | Definition |
|-------|------------|
| **Intact (1)** | No visible structural damage |
| **Damaged (2)** | Partial damage (cracks, missing roof, partial collapse) |
| **Destroyed (3)** | Complete collapse, burned, or covered by debris/water |

### Class Imbalance Challenge
- Building pixels: ~12.5% of total
- Destroyed buildings: ~6.5% of building pixels
- Damaged buildings: ~10.7% of building pixels
- Intact buildings: ~82.8% of building pixels

---

## Page 3: Model Performance & Key Findings

### Supervised Learning Results

**Best Performing Model: ChangeMamba**
- **Overall Accuracy**: 96.22%
- **mIoU**: 67.63%
- **F1 (Localization)**: 90.90%
- **F1 (Classification)**: 72.70%

**Key Insight**: Task decoupling models (separating building localization from damage classification) significantly outperform direct segmentation approaches.

| Model | OA (%) | mIoU (%) | F1 (Localization) | F1 (Classification) |
|-------|--------|----------|-------------------|---------------------|
| ChangeMamba | 96.22 | 67.63 | 90.90 | 72.70 |
| DamageFormer | 96.13 | 67.09 | 90.29 | 72.51 |
| ChangeOS | 95.84 | 65.98 | 89.60 | 71.88 |
| UNet | 95.47 | 64.94 | 87.97 | 72.24 |

### Cross-Event Generalization

**Zero-Shot Setting**: Models trained on 13 events, tested on 1 unseen event
- Average mIoU drops from ~50% to ~35%
- Shows significant challenge of generalization

**One-Shot Setting**: Single labeled sample from target event
- Average mIoU improves from ~35% to ~41%
- Minimal supervision helps adaptation

### Best Performing Events by Disaster Type

| Disaster Type | Performance | Explanation |
|--------------|-------------|-------------|
| **Wildfire** | Best (IoU >70% for "Destroyed") | Clear SAR signatures (burn scars, debris) |
| **Volcano** | Best (IoU >70% for "Destroyed") | Lava flows create strong SAR contrast |
| **Flood/Hurricane** | Good (IoU ~50-60% for "Damaged") | Water inundation visible in SAR |
| **Earthquake** | Challenging | Complex, heterogeneous damage patterns |
| **Conflict** | Poor | Limited samples, varied destruction |

---

## Page 4: Multimodal Analysis & Insights

### Role of Pre-Event Optical Data

Pre-event optical imagery provides more than just building localization:
- **Optical-only**: 69.76% mIoU
- **SAR-only**: 65.56% mIoU
- **Optical+SAR**: 70.79% mIoU (best performance)

**Key Finding**: SAR is a viable alternative when optical is unavailable, and fusion provides complementary information.

### The "Damaged" Class Challenge

The "Damaged" category is significantly harder to detect than "Destroyed":
- Damaged IoU: ~20-40% across most disaster types
- Destroyed IoU: ~55-70% for wildfires/volcanoes

**Why?** Partial damage is subtle in SAR imagery, whereas total destruction creates clear backscatter anomalies.

### Why Cross-Event Transfer is Hard

1. **Inconsistent damage signatures across events**:
   - SAR backscatter for "destroyed" buildings differs between wildfires, earthquakes, and floods
   - Same damage level looks different depending on building materials, sensor angle, terrain

2. **No target supervision for model selection**:
   - In real disasters, you can't tune hyperparameters on labeled target data
   - This creates a significant performance gap between "best possible" and "achievable with source-only validation"

---

## Page 5: Broader Impact & Future Directions

### What BRIGHT Enables

| Application | Description |
|-------------|-------------|
| **Multimodal AI** | Train models combining optical + SAR data |
| **Cross-Event Transfer** | Test generalization across disaster types/regions |
| **Unsupervised Domain Adaptation** | Align feature distributions across domains |
| **Semi-Supervised Learning** | Leverage unlabeled disaster imagery |
| **Foundation Models** | Pre-train large models on diverse EO data |
| **Multimodal Change Detection** | Detect disaster-induced changes without labels |
| **Image Registration** | Align optical and SAR imagery automatically |

### Key Limitations

1. **Registration Error**: ~1 pixel average, but SAR distortions exist
2. **Label Quality**: Manual polygon annotation has minor errors
3. **Sample Imbalance**: Some events dominate (Turkey: 1,114 tiles vs Hawaii: 65 tiles)
4. **Geographic Bias**: Northern hemisphere only (no events in southern hemisphere)
5. **Single-Polarization SAR**: Lacks richer polarimetric information
6. **Temporal Scope**: Limited to 2020-present (Capella/Umbra availability)

### Future Directions

1. **Add More Modalities**: Fully polarimetric SAR, LiDAR, multi-spectral
2. **Expand Geographic Coverage**: Southern hemisphere, more developing countries
3. **Improve Cross-Event Generalization**: Better domain adaptation methods
4. **Real-Time Deployment**: Models that run on edge devices for rapid response
5. **Foundation Models**: Pre-train on BRIGHT, fine-tune for specific events

### Significance

BRIGHT is the first dataset that enables:
- **All-weather disaster response** (not limited to clear skies)
- **Global coverage** with sub-meter resolution
- **Open access** to the research community
- **Realistic evaluation** of cross-event generalization

**Quote from the authors**: *"We hope that this effort will promote the development of AI-driven methods in support of people in disaster-affected areas... BRIGHT, true to its name, will bring even a glimmer of brightness to people in disaster-stricken areas by enabling more prompt and effective disaster response and relief."*

---

## Summary Table of Key Metrics

| Aspect | Result |
|--------|--------|
| **Best Model** | ChangeMamba (mIoU: 67.63%) |
| **Zero-Shot mIoU** | ~35% (avg across models) |
| **One-Shot mIoU** | ~41% (avg across models) |
| **Registration Error** | ~1 pixel |
| **Dataset Size** | 4,246 tiles, 384,596 buildings |
| **Resolution** | 0.3-1 m/pixel |
| **Licenses** | CC-BY-NC-4.0 (optical), CC-BY-4.0 (SAR) |
