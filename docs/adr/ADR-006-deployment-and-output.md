# ADR-006: Deployment Architecture — GeoTIFF Tiler + Multi-Format Field Output

**Status:** Accepted  
**Date:** 2026-06-29  
**Deciders:** Sabrina Pribadi  
**Technical Story:** Grill-Me design interview — Q5 Deployment & Scaling

---

## Context and Problem Statement

The end-to-end pipeline has two gaps between the trained model and operational field use:

1. **Input gap**: The model expects pre-cropped 512×512 GeoTIFFs. Real SAR deliveries from Capella/Umbra are scene-level GeoTIFFs (100+ km², multiple polarizations, various projections, 0.3–1 m/px resolution).

2. **Output gap**: The model produces class logits. Field coordinators use QGIS, Google Earth, or receive WhatsApp messages — not Streamlit bar charts.

The Streamlit dashboard is a development tool, not an operational interface.

## Decision Drivers

- Target latency: 45–50 seconds for 567 Morocco tiles end-to-end (disk I/O is the bottleneck, not inference)
- Field coordinators use GIS tools (QGIS free, ArcGIS paid); GeoJSON is natively supported
- WhatsApp/SMS reaches field teams without internet access to a server
- Output must carry geolocation (lat/lon bounds per tile) to be actionable on a map
- Minimum viable delivery must not require the field team to install software

## Considered Options

**Input preprocessing:**
- **Manual tiling**: Researcher tiles each new SAR scene by hand → not scalable, not repeatable. Rejected.
- **GeoTIFFTiler class (chosen)**: Automated sliding-window tiling with configurable size (512) and overlap (64 pixels for edge continuity); SAR-specific preprocessing (despeckling, dB conversion, 3σ clipping, normalization); preserves geolocation metadata per tile.

**Output format:**
- **Streamlit dashboard only** — Rejected: not usable offline, no geolocation, not shareable via WhatsApp
- **GeoJSON (chosen as primary)**: Opens in QGIS/ArcGIS/Google Earth; contains damage label + confidence + geobounds per tile; shareable via email or USB
- **PNG map (chosen as secondary)**: Static damage map image shareable via WhatsApp/SMS; no software required
- **SMS priority list (chosen as tertiary)**: Top-N tiles by severity with lat/lon coordinates; works without internet on feature phones
- **PDF report (deferred)**: Formal documentation; implement when operational handoff requires it

## Decision Outcome

**Three-tier output strategy:**

| Tier | Format | Tool | Who Gets It |
|------|--------|------|-------------|
| Primary | GeoJSON | QGIS, Google Earth | GIS-trained field coordinator |
| Secondary | PNG map | WhatsApp, email | Any responder with a phone |
| Tertiary | SMS priority list | SMS | First responder without smartphone |

**Input:** `GeoTIFFTiler` handles raw SAR → tiled tensors with geolocation metadata.

**Inference latency breakdown (567 tiles, MPS, batch_size=32):**

| Stage | Time | Note |
|-------|------|------|
| Data loading | ~5s | Pre-load all tiles into RAM first |
| Preprocessing | ~2s | Normalize, resize |
| Inference | ~28s | MPS batch inference |
| Post-processing | ~5s | Rank, threshold |
| GeoJSON + PNG generation | ~2s | |
| **Total** | **~42s** | Well within 6h response window |

## Positive Consequences

- GeoJSON output is immediately loadable in free tools (QGIS) — no paid software required
- PNG map enables sharing via WhatsApp to teams without GIS skills
- GeoTIFFTiler makes the pipeline repeatable for any new disaster event
- Total latency (~42s for Morocco) is well within the 6–72h operational window

## Negative Consequences

- GeoTIFFTiler needs optical reference alignment — pre-event optical must be available to validate SAR geolocation
- PNG map generation requires a basemap (OpenStreetMap tile fetch) for visual reference
- SMS priority list loses spatial context (only lat/lon point, not polygon)

## Implementation Notes

**Files to create in Phase 2:**
- `src/data/geotiff_tiler.py` — `GeoTIFFTiler` class
- `src/output/field_outputs.py` — `FieldOutputGenerator` class
- `scripts/infer_scene.py` — end-to-end pipeline: raw SAR → GeoJSON + PNG + SMS

**Dependency additions needed:**
- `geopandas` for GeoJSON creation
- `matplotlib` + `contextily` for basemap PNG generation (contextily fetches OSM tiles)

## Related Decisions

- ADR-001: Task framing (operational triage tool — defines output requirements)
- ADR-003: Validation strategy (per-tile confidence score feeds into GeoJSON properties)

## References

- Capella Space Open Data Gallery; Umbra Space Open Data Program (SAR data format)
- Chen et al. (2025). BRIGHT. ESSD 17(11). Section 2 (data format, GeoTIFF structure)
