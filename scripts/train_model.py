"""
Training script for BRIGHT building damage classification.

Run with one or more events:
    python scripts/train_model.py                          # turkey-earthquake (default)
    python scripts/train_model.py turkey-earthquake beirut-explosion
    MODEL_TYPE=optical_only python scripts/train_model.py turkey-earthquake

Data must be downloaded first:
    python scripts/download_events.py turkey-earthquake
"""
import os
import sys
import time
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
MODEL_TYPE = os.environ.get("MODEL_TYPE", "multimodal")

CLASS_WEIGHTS = torch.tensor([1.0, 5.0, 10.0], dtype=torch.float)

DEFAULT_EVENTS = ["turkey-earthquake"]
# ──────────────────────────────────────────────────────────────────────────────


def make_loader(events: list[str], split: str, shuffle: bool) -> DataLoader:
    datasets = []
    for event in events:
        data_dir = BASE_DIR / event
        ds = BRIGHTDataset(
            data_dir=data_dir,
            split_file=SPLIT_DIR / f"{split}_set.txt",
            event=event,
            image_size=IMAGE_SIZE,
            synthetic_fallback=False,
        )
        if len(ds) > 0:
            datasets.append(ds)
        else:
            print(f"  WARNING: no {split} tiles found for {event} — skipped")

    if not datasets:
        raise RuntimeError(f"No tiles found for any event in {split} split")

    combined = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    return DataLoader(combined, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = total_correct = total_samples = 0
    for batch in loader:
        optical = batch["images"]["optical"].to(DEVICE)
        sar     = batch["images"]["sar"].to(DEVICE)
        labels  = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        if MODEL_TYPE == "multimodal":
            logits = model(optical, sar)
        elif MODEL_TYPE == "optical_only":
            logits = model(optical)
        else:
            logits = model(sar)

        loss = criterion(logits, labels)
        loss.backward()
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
        optical = batch["images"]["optical"].to(DEVICE)
        sar     = batch["images"]["sar"].to(DEVICE)
        labels  = batch["label"].to(DEVICE)

        if MODEL_TYPE == "multimodal":
            logits = model(optical, sar)
        elif MODEL_TYPE == "optical_only":
            logits = model(optical)
        else:
            logits = model(sar)

        loss  = criterion(logits, labels)
        preds = logits.argmax(1)

        total_loss    += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total_samples
    acc      = total_correct / total_samples
    macro_f1 = f1_score(all_labels, all_preds, labels=list(range(NUM_CLASSES)),
                        average="macro", zero_division=0)
    return avg_loss, acc, macro_f1, all_preds, all_labels


def main():
    events = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_EVENTS
    run_name = "+".join(e.replace("-", "_") for e in events)

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Device     : {DEVICE}")
    print(f"Events     : {events}")
    print(f"Model type : {MODEL_TYPE}\n")

    train_loader = make_loader(events, "train", shuffle=True)
    val_loader   = make_loader(events, "val",   shuffle=False)

    print(f"Train : {len(train_loader.dataset)} tiles")
    print(f"Val   : {len(val_loader.dataset)} tiles\n")

    model = create_model(MODEL_TYPE, num_classes=NUM_CLASSES).to(DEVICE)
    print(f"Model : {model.__class__.__name__}  ({sum(p.numel() for p in model.parameters()):,} params)\n")

    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1   = 0.0
    best_path = OUTPUT_DIR / f"best_{run_name}_{MODEL_TYPE}.pt"

    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}  {'Val F1':>7}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc, vl_f1, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        flag = ""
        if vl_f1 > best_f1:
            best_f1 = vl_f1
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": vl_loss, "val_acc": vl_acc, "val_f1": vl_f1,
                "events": events, "model_type": MODEL_TYPE, "num_classes": NUM_CLASSES,
            }, best_path)
            flag = "  ✓ saved"

        print(
            f"{epoch:>5}  {tr_loss:>10.4f}  {tr_acc:>9.4f}  "
            f"{vl_loss:>9.4f}  {vl_acc:>8.4f}  {vl_f1:>7.4f}"
            f"  ({time.time()-t0:.1f}s){flag}"
        )

    print(f"\nLoading best checkpoint (val F1={best_f1:.4f}) for final report ...")
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    _, _, _, preds, labels = eval_epoch(model, val_loader, criterion)

    class_names = [DAMAGE_CLASSES[i] for i in range(NUM_CLASSES)]
    print("\nClassification Report (val set):")
    print(classification_report(labels, preds, labels=list(range(NUM_CLASSES)),
                                target_names=class_names, zero_division=0))
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
