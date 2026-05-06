"""Central configuration for Bumper-Level Crop Segmentation project.

Replace the placeholder PROJECT_NAME and architecture fields as needed.
All hyperparameters and file paths are centralized here to avoid magic numbers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict


@dataclass(frozen=True)
class Config:
    """Typed configuration container.

    Update these values for your environment. Paths may be absolute or
    relative to the repository root.
    """

    # Project metadata
    PROJECT_NAME: str = "Bumper-Level Crop Segmentation"
    DESCRIPTION: str = (
        "UGV-focused segmentation using bottom-50% crops to ignore sky and "
        "accelerate inference."
    )

    # Data paths (update these to your dataset location)
    DATA_ROOT: Path = Path("./data")
    TRAIN_IMAGES_DIR: Path = Path("./data/train/images")
    TRAIN_MASKS_DIR: Path = Path("./data/train/masks")
    VAL_IMAGES_DIR: Path = Path("./data/val/images")
    VAL_MASKS_DIR: Path = Path("./data/val/masks")
    TEST_IMAGES_DIR: Path = Path("./data/test/images")
    TEST_MASKS_DIR: Path = Path("./data/test/masks")

    # Model / dataset
    NUM_CLASSES: int = 10
    # The dataset uses an explicit value for Sky class; we map it to IGNORE_INDEX
    SKY_LABEL_VALUE: int = 10000
    IGNORE_INDEX: int = 255

    # Image handling
    # NOTE: Images are cropped to keep only the bottom `CROP_RATIO` portion.
    CROP_RATIO: float = 0.5  # bottom 50%
    # Resize width/height AFTER cropping for predictable model input
    RESIZE_HW: Tuple[int, int] = (320, 192)  # width, height (small, wide for speed)

    # Training hyperparameters
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 50
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    PIN_MEMORY: bool = True
    NUM_WORKERS: int = min(8, (os.cpu_count() or 1))

    # Hardware toggles
    USE_AMP: bool = True
    DEVICE: str = "cuda" if (os.environ.get("CUDA_VISIBLE_DEVICES") is not None) else "cpu"

    # Checkpoints / logging
    CHECKPOINT_DIR: Path = Path("./checkpoints")
    LOG_DIR: Path = Path("./logs")
    EXP_NAME: str = "mobilenetv3_bottom50_fast"

    # Backbone selection and training behavior
    BACKBONE: str = "mobilenet_v3_large"
    FREEZE_BACKBONE: bool = True

    # Inference target (documented constraint): aim for < 20ms/image on target hardware.
    TARGET_INFERENCE_MS: int = 20

    # Mapping from human labels -> class ids (example names)
    CLASS_MAP: Dict[int, str] = None


def get_default_config() -> Config:
    cls_map = {
        0: "background",
        1: "road",
        2: "vegetation",
        3: "vehicle",
        4: "person",
        5: "sign",
        6: "curb",
        7: "obstacle",
        8: "water",
        9: "sky",  # implicitly ignored in training/inference
    }
    cfg = Config()
    # object.__setattr__ because dataclass is frozen
    object.__setattr__(cfg, "CLASS_MAP", cls_map)
    return cfg


cfg = get_default_config()
