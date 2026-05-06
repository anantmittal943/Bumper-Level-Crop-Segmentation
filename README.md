# Bumper-Level Crop Segmentation

Project: Bumper-Level Crop Segmentation — fast semantic segmentation focused on the bottom 50% of images for UGV path planning.

This repository provides a modular, production-oriented implementation to train and evaluate a segmentation model that ignores the sky region and prioritizes extremely fast inference by cropping the bottom half of images.

**Project Name:** Bumper-Level Crop Segmentation

**Core Architecture/Approach:** MobileNetV3 encoder + lightweight decoder (custom). Uses CrossEntropyLoss with an IGNORE_INDEX mapped from a dataset-provided Sky label value (10000).

**Dataset:** Example expects train/val/test folders under `./data` with `images/` and `masks/` subfolders. Masks using value `10000` denote Sky and are mapped to an IGNORE label.

## Files
- `config.py` — central typed config and hyperparameters
- `dataset.py` — `BumperDataset` with bottom-50% crop, albumentations transforms, safe loading
- `model.py` — `SegmentationModel` (MobileNetV3 encoder + decoder)
- `train.py` — training loop, AMP, logging, checkpointing
- `test.py` — inference, timing (time.perf_counter), IoU and confusion matrix reporting

## Requirements

Recommended minimal environment (python 3.9+):

Install dependencies:
```bash
pip install torch torchvision albumentations scikit-learn pillow numpy
```

Optional: for faster backbone loading and more backbones, `timm` can be used.

## Setup dataset

Prepare your dataset structure:

```
data/
  train/
    images/
    masks/
  val/
    images/
    masks/
  test/
    images/
    masks/
```

Masks must be integer per-pixel class ids. Sky pixels must be encoded with value `10000` (or change `SKY_LABEL_VALUE` in `config.py`). The code maps that value to `IGNORE_INDEX` so the model ignores sky in loss and metrics.

## Training

Basic training command:

```bash
python train.py --epochs 30 --batch-size 32
```

Checkpoints are written to `./checkpoints`. The best model (by validation IoU) is saved as `{EXP_NAME}_best.pt`.

Notes on meeting the <20ms target:
- Crop to the bottom 50% reduces input pixels by half.
- Use `cfg.RESIZE_HW` to reduce input resolution (default small wide 320x192) — lowers computation.
- Freeze backbone weights and train only decoder for rapid fine-tune.
- Use `torchscript` or `torch.backends.quantized` and ONNX -> TensorRT for further inference speedups.

## Testing / Inference

Measure latency and IoU on test set:

```bash
python test.py --ckpt ./checkpoints/mobilenetv3_bottom50_fast_best.pt --batch-size 1
```

This script reports mean/median latency per image (ms) and per-class IoU. If observed latency exceeds `cfg.TARGET_INFERENCE_MS`, the script logs a warning and you should apply additional optimizations (lower resolution, quantization, TensorRT, or smaller backbone).

## Reproducibility

`train.py` includes a `set_seed()` helper to stabilize runs. For full reproducibility, pin `NUM_WORKERS=0` and set environment variables like `PYTHONHASHSEED`.

## Next steps and optimization suggestions

- Convert the trained model to TorchScript and run on the target hardware, measuring real-world latency.
- Apply 8-bit quantization (dynamic/static) for further speedups.
- Use TensorRT / ONNXRuntime for deployment to GPUs or inference accelerators.

## Contact

This implementation was produced as a modular reference for optimizing ROI-based segmentation for UGVs. For tailored changes (e.g., specific backbone swaps, quantization workflows, or dataset converters), open a new task referencing the file to change.
