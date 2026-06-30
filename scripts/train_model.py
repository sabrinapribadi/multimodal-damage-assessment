"""
Training script for BRIGHT building damage classification.

Run with one or more events:
    python scripts/train_model.py                          # turkey-earthquake (default)
    python scripts/train_model.py turkey-earthquake beirut-explosion
    MODEL_TYPE=optical_only python scripts/train_model.py turkey-earthquake

Data must be downloaded first:
    python scripts/download_events.py turkey-earthquake
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import ConcatDataset, DataLoader

sys.path.append(".")
from src.data.brighT_loader import DAMAGE_CLASSES, NUM_CLASSES, BRIGHTDataset
from src.models.baseline_model import create_model

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("data/processed")
SPLIT_DIR  = Path("BRIGHT/bda_benchmark/dataset/splitname/standard_ML")
OUTPUT_DIR = Path("outputs")
DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

BATCH_SIZE = 8
IMAGE_SIZE = 224
EPOCHS     = 20
LR         = 1e-3
PATIENCE   = 5          # early stopping — epochs without val F1 improvement
GRAD_CLIP  = 1.0        # max gradient norm

MODEL_TYPE = os.environ.get("MODEL_TYPE", "multimodal")

CLASS_WEIGHTS = torch.tensor([1.0, 5.0, 10.0], dtype=torch.float)
CLASS_NAMES   = [DAMAGE_CLASSES[i] for i in range(NUM_CLASSES)]

DEFAULT_EVENTS = ["turkey-earthquake"]
# ──────────────────────────────────────────────────────────────────────────────


def make_loader(events: list[str], split: str, shuffle: bool) -> DataLoader:
    datasets = []
    for event in events:
        data_dir = BASE_DIR / event
        if not data_dir.exists():
            print(f"  WARNING: data directory not found for {event} — run download_events.py first")
            continue
        ds = BRIGHTDataset(
            data_dir=data_dir,
            split_file=SPLIT_DIR / f"{split}_set.txt",
            event=event,
            image_size=IMAGE_SIZE,
            synthetic_fallback=False,
        )
        if len(ds) > 0:
            datasets.append(ds)
            print(f"  {split:5s} {event}: {len(ds)} tiles")
        else:
            print(f"  WARNING: no {split} tiles found for {event} — skipped")

    if not datasets:
        raise RuntimeError(
            f"No tiles found for any event in '{split}' split.\n"
            f"  Events requested: {events}\n"
            f"  Run: python scripts/download_events.py {' '.join(events)}"
        )

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    return DataLoader(combined, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def _forward(model, batch):
    optical = batch["images"]["optical"].to(DEVICE)
    sar     = batch["images"]["sar"].to(DEVICE)
    if MODEL_TYPE == "multimodal":
        return model(optical, sar)
    elif MODEL_TYPE == "optical_only":
        return model(optical)
    else:
        return model(sar)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = total_correct = total_samples = 0

    for batch in loader:
        labels = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        logits = _forward(model, batch)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()

        total_loss    += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = total_correct = total_samples = 0
    all_preds, all_labels = [], []

    for batch in loader:
        labels = batch["label"].to(DEVICE)
        logits = _forward(model, batch)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(1)

        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss      = total_loss / total_samples
    acc           = total_correct / total_samples
    macro_f1      = f1_score(all_labels, all_preds, labels=list(range(NUM_CLASSES)),
                             average="macro", zero_division=0)
    per_class_f1  = f1_score(all_labels, all_preds, labels=list(range(NUM_CLASSES)),
                             average=None, zero_division=0)
    return avg_loss, acc, macro_f1, per_class_f1, all_preds, all_labels


def main():
    events   = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_EVENTS
    run_name = "+".join(e.replace("-", "_") for e in events)

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Device     : {DEVICE}")
    print(f"Events     : {events}")
    print(f"Model type : {MODEL_TYPE}")
    print(f"Early stop : patience={PATIENCE} epochs")
    print(f"Grad clip  : max_norm={GRAD_CLIP}\n")

    print("Loading datasets ...")
    train_loader = make_loader(events, "train", shuffle=True)
    val_loader   = make_loader(events, "val",   shuffle=False)
    print(f"\nTrain total: {len(train_loader.dataset)} tiles")
    print(f"Val   total: {len(val_loader.dataset)} tiles\n")

    model = create_model(MODEL_TYPE, num_classes=NUM_CLASSES).to(DEVICE)
    print(f"Model : {model.__class__.__name__}  ({sum(p.numel() for p in model.parameters()):,} params)\n")

    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1         = 0.0
    best_path       = OUTPUT_DIR / f"best_{run_name}_{MODEL_TYPE}.pt"
    patience_counter = 0

    header = (f"{'Epoch':>5}  {'TrLoss':>8}  {'TrAcc':>7}  "
              f"{'VlLoss':>8}  {'VlAcc':>7}  {'MacroF1':>8}  "
              f"{'I':>6} {'D':>6} {'R':>6}")
    print(header)
    print("-" * len(header))

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc                          = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_f1, pc_f1, _, _     = eval_epoch(model, val_loader,   criterion)
        scheduler.step()

        flag = ""
        if vl_f1 > best_f1:
            best_f1          = vl_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": vl_loss, "val_acc": vl_acc, "val_f1": vl_f1,
                "val_per_class_f1": pc_f1.tolist(),
                "events": events, "model_type": MODEL_TYPE, "num_classes": NUM_CLASSES,
            }, best_path)
            flag = "  ✓"
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(
                    f"{epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  "
                    f"{vl_loss:>8.4f}  {vl_acc:>7.4f}  {vl_f1:>8.4f}  "
                    f"{pc_f1[0]:>6.3f} {pc_f1[1]:>6.3f} {pc_f1[2]:>6.3f}"
                    f"  ({time.time()-t0:.1f}s)"
                )
                print(f"\nEarly stopping — no improvement for {PATIENCE} epochs.")
                break

        print(
            f"{epoch:>5}  {tr_loss:>8.4f}  {tr_acc:>7.4f}  "
            f"{vl_loss:>8.4f}  {vl_acc:>7.4f}  {vl_f1:>8.4f}  "
            f"{pc_f1[0]:>6.3f} {pc_f1[1]:>6.3f} {pc_f1[2]:>6.3f}"
            f"  ({time.time()-t0:.1f}s){flag}"
        )

    # ── Final evaluation on best checkpoint ───────────────────────────────────
    print(f"\nLoading best checkpoint (val MacroF1={best_f1:.4f}) ...")
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    _, _, val_macro_f1, _, val_preds, val_labels = eval_epoch(model, val_loader, criterion)
    print("\nClassification Report (val set):")
    print(classification_report(val_labels, val_preds, labels=list(range(NUM_CLASSES)),
                                target_names=CLASS_NAMES, zero_division=0))

    # ── Test set evaluation ───────────────────────────────────────────────────
    test_macro_f1 = None
    try:
        test_loader = make_loader(events, "test", shuffle=False)
        print(f"\nTest total: {len(test_loader.dataset)} tiles")
        _, _, test_macro_f1, _, test_preds, test_labels = eval_epoch(model, test_loader, criterion)
        print("\nClassification Report (test set):")
        print(classification_report(test_labels, test_preds, labels=list(range(NUM_CLASSES)),
                                    target_names=CLASS_NAMES, zero_division=0))
    except RuntimeError as e:
        print(f"\nTest set skipped: {e}")

    # ── Save run metadata ─────────────────────────────────────────────────────
    metadata = {
        "events":          events,
        "model_type":      MODEL_TYPE,
        "run_name":        run_name,
        "batch_size":      BATCH_SIZE,
        "image_size":      IMAGE_SIZE,
        "epochs_config":   EPOCHS,
        "epochs_run":      ckpt["epoch"],
        "learning_rate":   LR,
        "patience":        PATIENCE,
        "grad_clip":       GRAD_CLIP,
        "class_weights":   CLASS_WEIGHTS.tolist(),
        "best_val_f1":     round(best_f1, 6),
        "best_val_acc":    round(ckpt["val_acc"], 6),
        "best_val_per_class_f1": {
            "Intact":    round(ckpt["val_per_class_f1"][0], 6),
            "Damaged":   round(ckpt["val_per_class_f1"][1], 6),
            "Destroyed": round(ckpt["val_per_class_f1"][2], 6),
        },
        "test_macro_f1":   round(test_macro_f1, 6) if test_macro_f1 is not None else None,
        "device":          str(DEVICE),
        "timestamp":       datetime.now().isoformat(),
    }
    meta_path = OUTPUT_DIR / f"{run_name}_{MODEL_TYPE}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nBest checkpoint : {best_path}")
    print(f"Metadata        : {meta_path}")


if __name__ == "__main__":
    main()
