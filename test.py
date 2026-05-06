"""Inference and evaluation script measuring per-image latency and IoU.

This script strictly times inference using time.perf_counter(). It also builds a
confusion matrix and reports per-class IoU. Sky class is ignored during eval.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from config import cfg
from dataset import default_val_transforms, make_dataloader
from model import create_model


def set_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def evaluate_checkpoint(checkpoint_path: Path, batch_size: int = 1):
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=cfg.NUM_CLASSES, freeze_backbone=False)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()

    val_transforms = default_val_transforms(cfg.RESIZE_HW)
    dl = make_dataloader(cfg.TEST_IMAGES_DIR, cfg.TEST_MASKS_DIR, batch_size=batch_size, shuffle=False, transforms=val_transforms)

    all_preds = []
    all_targets = []
    times = []
    with torch.no_grad():
        for imgs, masks in dl:
            imgs = imgs.to(device)
            start = time.perf_counter()
            logits = model(imgs)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()
            times.append((end - start) / imgs.size(0))

            preds = torch.argmax(logits, dim=1).cpu().numpy().ravel()
            targets = masks.cpu().numpy().ravel()
            # mask out ignored pixels
            mask = targets != cfg.IGNORE_INDEX
            all_preds.append(preds[mask])
            all_targets.append(targets[mask])

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(cfg.NUM_CLASSES)))

    per_class_iou = []
    for c in range(cfg.NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else float("nan")
        per_class_iou.append(iou)

    mean_iou = np.nanmean([iou for iou in per_class_iou if not np.isnan(iou)])
    mean_latency_ms = float(np.mean(times) * 1000)
    median_latency_ms = float(np.median(times) * 1000)

    return {
        "mean_iou": mean_iou,
        "per_class_iou": per_class_iou,
        "confusion_matrix": cm,
        "mean_latency_ms": mean_latency_ms,
        "median_latency_ms": median_latency_ms,
    }


def main(argv: Tuple[str, ...]):
    parser = argparse.ArgumentParser("Test segmentation model")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args(list(argv))
    set_logging()
    logger = logging.getLogger("test")

    res = evaluate_checkpoint(Path(args.ckpt), batch_size=args.batch_size)
    logger.info("Mean IoU: %.4f", res["mean_iou"])
    logger.info("Per-class IoU: %s", res["per_class_iou"])
    logger.info("Mean latency (ms): %.2f", res["mean_latency_ms"])
    logger.info("Median latency (ms): %.2f", res["median_latency_ms"])

    if res["mean_latency_ms"] * 1.0 > cfg.TARGET_INFERENCE_MS:
        logger.warning("Target inference %.1fms not met (observed %.2fms). Consider further optimizations.", cfg.TARGET_INFERENCE_MS, res["mean_latency_ms"])


if __name__ == "__main__":
    main(tuple())
