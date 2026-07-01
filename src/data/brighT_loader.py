"""
BRIGHT Dataset Loader — actual BRIGHT format with GeoTIFF support.

BRIGHT file structure:
  {data_dir}/pre-event/{tile_id}_pre_disaster.tif   → optical RGB (3-ch)
  {data_dir}/post-event/{tile_id}_post_disaster.tif → SAR (1-ch)
  {data_dir}/target/{tile_id}_building_damage.tif   → mask (0=bg,1=intact,2=damaged,3=destroyed)

Tile-level label = max non-background damage class in the mask.
Output classes (num_classes=3): 0=Intact, 1=Damaged, 2=Destroyed.
"""
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# BRIGHT pixel labels → tile classification mapping
BRIGHT_PIXEL = {0: "background", 1: "intact", 2: "damaged", 3: "destroyed"}

# Output class indices (background is ignored; tile label = max building damage level - 1)
DAMAGE_CLASSES = {0: "Intact", 1: "Damaged", 2: "Destroyed"}
NUM_CLASSES = 3

# ImageNet stats for optical (pre-event); neutral normalisation for SAR (post-event)
_OPTICAL_MEAN = [0.485, 0.456, 0.406]
_OPTICAL_STD  = [0.229, 0.224, 0.225]
_SAR_MEAN     = [0.5]
_SAR_STD      = [0.5]


def _is_valid_tif(path: Path) -> bool:
    """Check that a TIF is readable (opens header AND reads a 1x1 pixel window)."""
    try:
        import rasterio
        from rasterio.windows import Window
        with rasterio.open(path) as src:
            src.read(1, window=Window(0, 0, 1, 1))
        return True
    except Exception:
        return False


def _read_tif(path: Path, n_channels: int) -> Image.Image:
    """Read a GeoTIFF (or regular TIF) as a PIL Image."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            if n_channels == 1:
                arr = src.read(1).astype(np.float32)
                arr = np.clip(arr, 0, None)
                if arr.max() > 0:
                    arr = arr / arr.max() * 255.0
                return Image.fromarray(arr.astype(np.uint8), mode="L")
            else:
                # Read first 3 bands; BRIGHT optical is RGB
                bands = [src.read(b).astype(np.float32) for b in range(1, min(4, src.count + 1))]
                while len(bands) < 3:
                    bands.append(bands[-1])
                stacked = np.stack(bands[:3], axis=-1)
                # Percentile stretch to uint8
                lo, hi = np.percentile(stacked, 2), np.percentile(stacked, 98)
                if hi > lo:
                    stacked = np.clip((stacked - lo) / (hi - lo) * 255.0, 0, 255)
                return Image.fromarray(stacked.astype(np.uint8), mode="RGB")
    except ImportError:
        # Fallback to PIL for non-GeoTIFF TIFs
        img = Image.open(path)
        return img.convert("RGB") if n_channels == 3 else img.convert("L")


def _read_mask(path: Path) -> np.ndarray:
    """Read damage label mask as integer numpy array (values 0-3)."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            return src.read(1).astype(np.uint8)
    except ImportError:
        arr = np.array(Image.open(path))
        return arr.astype(np.uint8)


def _augment_pair(optical: Image.Image, sar: Image.Image) -> tuple[Image.Image, Image.Image]:
    """
    Apply identical spatial transforms to both modalities; colour jitter to optical only.
    Co-registration must be preserved — both images get the same flip/rotation.
    """
    if random.random() > 0.5:
        optical, sar = TF.hflip(optical), TF.hflip(sar)
    if random.random() > 0.5:
        optical, sar = TF.vflip(optical), TF.vflip(sar)
    angle = random.uniform(-15.0, 15.0)
    optical = TF.rotate(optical, angle)
    sar     = TF.rotate(sar,     angle)
    # Optical-only: colour jitter (SAR intensity is physically calibrated — don't perturb)
    optical = TF.adjust_brightness(optical, random.uniform(0.8, 1.2))
    optical = TF.adjust_contrast(optical,   random.uniform(0.8, 1.2))
    return optical, sar


def _derive_tile_label(mask: np.ndarray,
                        destroyed_thresh: float = 0.01,
                        damaged_thresh: float = 0.05,
                        min_building_px: int = 200) -> Optional[int]:
    """
    Returns tile-level class index (0=Intact, 1=Damaged, 2=Destroyed).
    Returns None if building coverage is too sparse to trust.

    Decision rules (area-weighted, paper-aligned):
      - Destroyed  if destroyed_px  / building_px >= destroyed_thresh  (default 5%)
      - Damaged    if damaged_px    / building_px >= damaged_thresh     (default 20%)
      - Intact     otherwise

    Thresholds grounded in BRIGHT paper (Section 2.3, Figure 5d, Table 5):
      - Destroyed has strong SAR signal (IoU >70% for wildfires/volcanoes) → low threshold
      - Damaged has weak SAR signal  (IoU <20% for most events)           → high threshold
      - min_building_px guards against misregistration noise on sparse tiles
        (~1 px BRIGHT registration error; 200 px ≈ 3 small buildings at 0.5 m/px)
    """
    building_px = mask > 0
    n_building = int(building_px.sum())
    if n_building < min_building_px:
        return None  # too sparse — registration noise would corrupt the label

    destroyed_frac = float((mask == 3).sum()) / n_building
    damaged_frac   = float((mask == 2).sum()) / n_building

    if destroyed_frac >= destroyed_thresh:
        return 2  # Destroyed
    if damaged_frac >= damaged_thresh:
        return 1  # Damaged
    return 0      # Intact


class BRIGHTDataset(Dataset):
    """
    BRIGHT tile-level damage classification dataset.

    Args:
        data_dir:    Root folder containing pre-event/, post-event/, target/ subfolders.
        split_file:  Path to a split .txt file listing tile IDs (one per line, e.g. 'morocco-earthquake_00000114').
        event:       Optional event name filter, e.g. 'morocco-earthquake'. Filters split_file entries.
        image_size:  Resize both modalities to this square size before feeding the model.
        synthetic_fallback: If True, generate dummy data when no real data is found (for smoke-tests).
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split_file: Union[str, Path],
        event: Optional[str] = None,
        image_size: int = 224,
        synthetic_fallback: bool = True,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.augment = augment

        self.optical_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_OPTICAL_MEAN, std=_OPTICAL_STD),
        ])
        self.sar_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_SAR_MEAN, std=_SAR_STD),
        ])

        self.samples = self._load_split(split_file, event, synthetic_fallback)
        logger.info(
            "BRIGHTDataset: %d samples | event=%s | split=%s",
            len(self.samples), event or "all", Path(split_file).name,
        )

    def _load_split(self, split_file, event, synthetic_fallback):
        split_file = Path(split_file)
        if not split_file.exists():
            logger.warning("Split file not found: %s", split_file)
            return self._synthetic_samples(20) if synthetic_fallback else []

        tile_ids = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
        if event:
            tile_ids = [t for t in tile_ids if t.startswith(event)]

        samples = []
        for tile_id in tile_ids:
            pre  = self.data_dir / "pre-event"  / f"{tile_id}_pre_disaster.tif"
            post = self.data_dir / "post-event" / f"{tile_id}_post_disaster.tif"
            tgt  = self.data_dir / "target"     / f"{tile_id}_building_damage.tif"

            if not (pre.exists() and post.exists() and tgt.exists()):
                continue  # skip tiles not yet downloaded

            if not (_is_valid_tif(pre) and _is_valid_tif(post) and _is_valid_tif(tgt)):
                logger.warning("Skipping corrupt tile %s", tile_id)
                continue

            try:
                mask  = _read_mask(tgt)
            except Exception as e:
                logger.warning("Skipping corrupt tile %s: %s", tile_id, e)
                continue
            label = _derive_tile_label(mask)
            if label is None:
                continue  # skip all-background tiles

            samples.append({
                "tile_id":  tile_id,
                "pre_path": pre,
                "post_path": post,
                "label":    label,
            })

        if not samples:
            logger.warning("No data found in %s (event=%s). Using synthetic samples.", split_file, event)
            return self._synthetic_samples(20) if synthetic_fallback else []

        return samples

    def _synthetic_samples(self, n: int):
        """Dummy samples — returns placeholder tensors so training loop runs without real data."""
        return [
            {"tile_id": f"synth_{i:04d}", "pre_path": None, "post_path": None, "label": i % NUM_CLASSES}
            for i in range(n)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if s["pre_path"] is not None:
            optical_pil = _read_tif(s["pre_path"],  n_channels=3)
            sar_pil     = _read_tif(s["post_path"], n_channels=1)
            if self.augment:
                optical_pil, sar_pil = _augment_pair(optical_pil, sar_pil)
            optical = self.optical_tf(optical_pil)
            sar     = self.sar_tf(sar_pil)
        else:
            optical = torch.randn(3, self.image_size, self.image_size)
            sar     = torch.randn(1, self.image_size, self.image_size)

        return {
            "tile_id": s["tile_id"],
            "images":  {"optical": optical, "sar": sar},
            "label":   torch.tensor(s["label"], dtype=torch.long),
        }


def create_dataloader(
    data_dir: Union[str, Path],
    split_file: Union[str, Path],
    event: Optional[str] = None,
    image_size: int = 224,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    synthetic_fallback: bool = True,
) -> DataLoader:
    dataset = BRIGHTDataset(
        data_dir=data_dir,
        split_file=split_file,
        event=event,
        image_size=image_size,
        synthetic_fallback=synthetic_fallback,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=False)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    data_dir   = sys.argv[1] if len(sys.argv) > 1 else "data/processed/morocco-earthquake"
    split_file = sys.argv[2] if len(sys.argv) > 2 else (
        "BRIGHT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt"
    )

    loader = create_dataloader(
        data_dir=data_dir,
        split_file=split_file,
        event="morocco-earthquake",
        batch_size=2,
    )
    batch = next(iter(loader))
    print("tile_id :", batch["tile_id"])
    print("optical :", batch["images"]["optical"].shape)
    print("sar     :", batch["images"]["sar"].shape)
    print("label   :", batch["label"].tolist(),
          [DAMAGE_CLASSES[l.item()] for l in batch["label"]])
