"""Training script for Bumper-Level Crop Segmentation.

Usage: python train.py --config ./config.py --epochs 30

Implements logging, AMP, checkpointing, and best-model saving by validation IoU.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from config import cfg
from dataset import default_train_transforms, default_val_transforms, make_dataloader
from model import create_model


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_iou_batch(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> float:
    """Compute mean IoU across classes excluding ignore_index."""
    pred = torch.argmax(pred, dim=1).cpu().numpy().ravel()
    target = target.cpu().numpy().ravel()
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    if target.size == 0:
        return 0.0
    ious = []
    for c in range(num_classes):
        tp = np.logical_and(pred == c, target == c).sum()
        fp = np.logical_and(pred == c, target != c).sum()
        fn = np.logical_and(pred != c, target == c).sum()
        denom = tp + fp + fn
        if denom == 0:
            continue
        ious.append(tp / denom)
    return float(np.mean(ious)) if len(ious) > 0 else 0.0


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: str, scaler: GradScaler, use_amp: bool):
    model.train()
    running_loss = 0.0
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            logits = model(imgs)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def validate(model: nn.Module, loader: DataLoader, device: str, ignore_index: int):
    model.eval()
    iou_scores = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            iou = compute_iou_batch(logits, masks, cfg.NUM_CLASSES, ignore_index)
            iou_scores.append(iou)
    return float(np.mean(iou_scores)) if iou_scores else 0.0


def main(argv: Tuple[str, ...]):
    parser = argparse.ArgumentParser("Train segmentation model")
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(list(argv))

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("train")

    set_seed(42)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_transforms = default_train_transforms(cfg.RESIZE_HW)
    val_transforms = default_val_transforms(cfg.RESIZE_HW)

    train_loader = make_dataloader(cfg.TRAIN_IMAGES_DIR, cfg.TRAIN_MASKS_DIR, batch_size=args.batch_size, shuffle=True, transforms=train_transforms)
    val_loader = make_dataloader(cfg.VAL_IMAGES_DIR, cfg.VAL_MASKS_DIR, batch_size=args.batch_size, shuffle=False, transforms=val_transforms)

    model = create_model(num_classes=cfg.NUM_CLASSES, freeze_backbone=cfg.FREEZE_BACKBONE)
    model = model.to(device)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.IGNORE_INDEX)
    scaler = GradScaler(enabled=cfg.USE_AMP)

    best_val_iou = 0.0
    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state.get("optimizer_state", {}))
        start_epoch = state.get("epoch", 0)
        best_val_iou = state.get("best_val_iou", 0.0)
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, cfg.USE_AMP)
        val_iou = validate(model, val_loader, device, cfg.IGNORE_INDEX)
        t1 = time.time()

        logger.info("Epoch %d/%d: train_loss=%.4f val_iou=%.4f epoch_time=%.1fs", epoch + 1, args.epochs, train_loss, val_iou, t1 - t0)

        # Save checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_iou": best_val_iou,
        }
        ckpt_path = cfg.CHECKPOINT_DIR / f"{cfg.EXP_NAME}_epoch{epoch+1}.pt"
        torch.save(ckpt, ckpt_path)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_path = cfg.CHECKPOINT_DIR / f"{cfg.EXP_NAME}_best.pt"
            torch.save({**ckpt, "best_val_iou": best_val_iou}, best_path)
            logger.info("Saved best model to %s (val_iou=%.4f)", best_path, best_val_iou)


if __name__ == "__main__":
    main(tuple())
