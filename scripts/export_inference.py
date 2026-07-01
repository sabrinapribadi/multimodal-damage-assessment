"""
Pre-compute inference results for Streamlit deployment.

Run once locally after training — output is a self-contained parquet that
includes 64×64 thumbnails (base64 PNG) so the dashboard needs no TIF files
or model weights at runtime.

Usage:
    python scripts/export_inference.py                                   # default events
    python scripts/export_inference.py turkey-earthquake beirut-explosion

Output:
    data/inference_results.parquet  (~20-30 MB, safe to commit)
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.append(".")
from src.data.brighT_loader import DAMAGE_CLASSES, NUM_CLASSES, BRIGHTDataset
from src.models.baseline_model import create_model

DEFAULT_EVENTS = ["turkey-earthquake", "noto-earthquake"]
SPLIT_DIR  = Path("BRIGHT/bda_benchmark/dataset/splitname/standard_ML")
BASE_DIR   = Path("data/processed")
OUTPUT_DIR = Path("outputs")
OUT_FILE   = Path("data/inference_results.parquet")
IMAGE_SIZE = 224
THUMB_SIZE = 64
DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DAMAGE_NAMES = [DAMAGE_CLASSES[i] for i in range(NUM_CLASSES)]


def _checkpoint_name(events: list[str]) -> str:
    return "+".join(e.replace("-", "_") for e in events)


def _load_model(model_type: str, events: list[str]) -> torch.nn.Module:
    run_name = _checkpoint_name(events)
    path = OUTPUT_DIR / f"best_{run_name}_{model_type}.pt"
    model = create_model(model_type, num_classes=NUM_CLASSES).to(DEVICE)
    if path.exists():
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        f1 = ckpt.get("val_f1", None)
        print(f"  {model_type}: {path.name}  val F1={f1:.4f}" if f1 else f"  {model_type}: {path.name}")
    else:
        print(f"  WARNING: {path} not found — using random weights")
    model.eval()
    return model


def _thumb_b64(path: Path, n_channels: int) -> str:
    from src.data.brighT_loader import _read_tif
    img = _read_tif(path, n_channels=n_channels).resize(
        (THUMB_SIZE, THUMB_SIZE), Image.LANCZOS
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def _infer(model: torch.nn.Module, sample: dict, model_type: str) -> np.ndarray:
    optical = sample["images"]["optical"].unsqueeze(0).to(DEVICE)
    sar     = sample["images"]["sar"].unsqueeze(0).to(DEVICE)
    is_mm = model_type in ("multimodal", "multimodal_v2", "multimodal_v3")
    logits  = model(optical, sar) if is_mm else model(optical)
    return torch.softmax(logits, dim=1)[0].cpu().numpy()


def main():
    events = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_EVENTS
    print(f"Device : {DEVICE}")
    print(f"Events : {events}")
    print("Loading models ...")
    model_mm  = _load_model("multimodal_v3",  events)
    model_opt = _load_model("optical_only_v3", events)

    records = []
    for split in ["train", "val", "test"]:
        split_file = SPLIT_DIR / f"{split}_set.txt"
        if not split_file.exists():
            print(f"\n  Skipping {split}: {split_file} not found")
            continue

        for event in events:
            data_dir = BASE_DIR / event
            if not data_dir.exists():
                print(f"\n  Skipping {event}/{split}: data directory not found")
                continue

            ds = BRIGHTDataset(
                data_dir=data_dir,
                split_file=split_file,
                event=event,
                image_size=IMAGE_SIZE,
                synthetic_fallback=False,
            )
            if len(ds) == 0:
                print(f"\n  Skipping {event}/{split}: no tiles found")
                continue

            print(f"\n{split} / {event}  ({len(ds)} tiles) ...")

            for i, s in enumerate(ds.samples):
                sample   = ds[i]
                true_lbl = sample["label"].item()

                probs_mm  = _infer(model_mm,  sample, "multimodal_v3")
                probs_opt = _infer(model_opt, sample, "optical_only_v3")

                records.append({
                    "tile_id":            s["tile_id"],
                    "event":              event,
                    "split":              split,
                    "true_label":         true_lbl,
                    "true_label_name":    DAMAGE_NAMES[true_lbl],
                    "pred_multimodal":    int(probs_mm.argmax()),
                    "pred_optical":       int(probs_opt.argmax()),
                    "conf_mm_intact":     float(probs_mm[0]),
                    "conf_mm_damaged":    float(probs_mm[1]),
                    "conf_mm_destroyed":  float(probs_mm[2]),
                    "conf_opt_intact":    float(probs_opt[0]),
                    "conf_opt_damaged":   float(probs_opt[1]),
                    "conf_opt_destroyed": float(probs_opt[2]),
                    "optical_thumb":      _thumb_b64(s["pre_path"],  n_channels=3),
                    "sar_thumb":          _thumb_b64(s["post_path"], n_channels=1),
                })

                if (i + 1) % 25 == 0:
                    print(f"  {i+1}/{len(ds)}", end="\r", flush=True)

            print(f"  {len(ds)}/{len(ds)} done")

    df = pd.DataFrame(records)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)
    mb = OUT_FILE.stat().st_size / 1e6
    print(f"\nSaved {len(df)} tiles → {OUT_FILE}  ({mb:.1f} MB)")

    for split_name in ["val", "test"]:
        split_df = df[df["split"] == split_name]
        if split_df.empty:
            continue
        print(f"\n{split_name} summary ({len(split_df)} tiles):")
        for col, name in [("pred_multimodal", "multimodal"), ("pred_optical", "optical_only")]:
            acc = (split_df[col] == split_df["true_label"]).mean()
            print(f"  {name} accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
