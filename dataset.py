"""Dataset and DataLoader utilities for bottom-50% crop segmentation.

Implements safe image loading and an albumentations augmentation pipeline.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple, List
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import cfg


ImageFile.LOAD_TRUNCATED_IMAGES = True
_LOG = logging.getLogger(__name__)


def default_train_transforms(resize_hw: Tuple[int, int]):
    return A.Compose(
        [
            A.RandomBrightnessContrast(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Resize(height=resize_hw[1], width=resize_hw[0]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def default_val_transforms(resize_hw: Tuple[int, int]):
    return A.Compose(
        [
            A.Resize(height=resize_hw[1], width=resize_hw[0]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


class BumperDataset(Dataset):
    """PyTorch Dataset that crops the bottom `CROP_RATIO` of images and loads masks.

    The dataset handles corrupted files by logging a warning and returning a
    zero image and all-IGNORE mask so training is not blocked.
    """

    def __init__(
        self,
        images_dir: Path,
        masks_dir: Path,
        crop_ratio: float = cfg.CROP_RATIO,
        transforms: Optional[Callable] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.crop_ratio = float(crop_ratio)
        self.transforms = transforms

        self.samples = self._gather_samples()
        if len(self.samples) == 0:
            _LOG.warning("No samples found in %s / %s", images_dir, masks_dir)

    def _gather_samples(self) -> List[Tuple[Path, Path]]:
        images = sorted(self.images_dir.glob("**/*.*"))
        pairs = []
        for img_path in images:
            # Assume mask filename matches image name
            mask_path = self.masks_dir / img_path.name
            if mask_path.exists():
                pairs.append((img_path, mask_path))
            else:
                _LOG.debug("Mask missing for image: %s", img_path)
        return pairs

    def __len__(self) -> int:
        return len(self.samples)

    def _safe_load_image(self, path: Path) -> Image.Image:
        try:
            img = Image.open(path).convert("RGB")
            return img
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Failed to load image %s (%s). Returning black image.", path, exc)
            # safe fallback: black image of a standard size
            return Image.new("RGB", (cfg.RESIZE_HW[0], int(cfg.RESIZE_HW[1] / cfg.CROP_RATIO)), color=(0, 0, 0))

    def _safe_load_mask(self, path: Path) -> Image.Image:
        try:
            # Some masks may be stored as single-channel TIFFs or pngs
            mask = Image.open(path)
            return mask
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Failed to load mask %s (%s). Returning IGNORE mask.", path, exc)
            arr = np.full((int(cfg.RESIZE_HW[1] / cfg.CROP_RATIO), cfg.RESIZE_HW[0]), fill_value=cfg.IGNORE_INDEX, dtype=np.uint8)
            return Image.fromarray(arr)

    def _bottom_crop(self, img: Image.Image, mask: Image.Image):
        w, h = img.size
        crop_h = int(h * self.crop_ratio)
        top = h - crop_h
        img_c = img.crop((0, top, w, h))
        mask_c = mask.crop((0, top, w, h))
        return img_c, mask_c

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]
        img = self._safe_load_image(img_path)
        mask = self._safe_load_mask(mask_path)

        img_c, mask_c = self._bottom_crop(img, mask)

        img_np = np.array(img_c)
        mask_np = np.array(mask_c)

        # Map SKY label value to IGNORE_INDEX
        mask_np = np.where(mask_np == cfg.SKY_LABEL_VALUE, cfg.IGNORE_INDEX, mask_np)
        # Ensure mask is integer class ids
        mask_np = mask_np.astype(np.uint8)

        if self.transforms:
            augmented = self.transforms(image=img_np, mask=mask_np)
            img_t = augmented["image"]
            mask_t = augmented["mask"].long()
        else:
            # Minimal tensor conversion if no transforms provided
            img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask_np).long()

        return img_t, mask_t


def make_dataloader(
    images_dir: Path,
    masks_dir: Path,
    batch_size: int,
    shuffle: bool,
    transforms: Optional[Callable],
    num_workers: int = cfg.NUM_WORKERS,
    pin_memory: bool = cfg.PIN_MEMORY,
):
    ds = BumperDataset(images_dir=images_dir, masks_dir=masks_dir, transforms=transforms)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tr = default_train_transforms(cfg.RESIZE_HW)
    dl = make_dataloader(cfg.TRAIN_IMAGES_DIR, cfg.TRAIN_MASKS_DIR, batch_size=8, shuffle=True, transforms=tr)
    for imgs, masks in dl:
        print(imgs.shape, masks.shape)
        break
