"""
train.py — DPR-RIA Backbone Training Script
=============================================
Trains the Dual-Path shared backbone (SPDConv + BMS-SPPF + P2 head)
on the RDD2022 dataset.

Dataset:  /Users/vjkiran/Documents/RoadDamageDetection/datasets
           ├── train/images  (37,230 images)
           ├── valid/images  (3,286 images)
           └── test/images   (3,285 images)

Usage
-----
  # Full training (recommended)
  python3 src/scripts/train.py

  # Quick sanity check (2 epochs, 640 images/run)
  python3 src/scripts/train.py --fast

  # Resume from last checkpoint
  python3 src/scripts/train.py --resume
"""

import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.custom_model import build_dual_path_model

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_YAML   = os.path.abspath("src/data/rdd2022.yaml")
MODEL_YAML  = "src/models/yolo11-spd-bmssppf-p2.yaml"
PRETRAINED  = "yolo11n.pt"          # backbone warm-start (downloaded automatically)
RUNS_DIR    = "runs/train"

# ─── Detect best available device ─────────────────────────────────────────────
def get_device():
    import torch
    if torch.cuda.is_available():
        return "0"          # NVIDIA GPU
    try:
        if torch.backends.mps.is_available():
            return "mps"    # Apple Silicon
    except AttributeError:
        pass
    return "cpu"


def run_training(fast: bool = False, resume: bool = False):
    device = get_device()
    print(f"[Train] Device: {device}")
    print(f"[Train] Data:   {DATA_YAML}")
    print(f"[Train] Model:  {MODEL_YAML}")

    # ── Memory-aware batch size ───────────────────────────────────────────────
    # M1 Pro unified memory is shared between system and GPU.
    # P2 head adds ~40% more output anchors, so we halve the safe baseline.
    if device == "mps":
        batch = 4 if fast else 6          # MPS safe: avoids 11+ GB OOM
    elif device == "cpu":
        batch = 4
    else:
        batch = 16                         # NVIDIA GPU (Jetson / cloud)

    model = build_dual_path_model(yaml_file=MODEL_YAML, weights=PRETRAINED)

    run_name = "dual_path_rdd2022_fast" if fast else "dual_path_rdd2022"

    train_kwargs = dict(
        data         = DATA_YAML,
        epochs       = 5 if fast else 100,
        imgsz        = 640,
        batch        = batch,
        device       = device,
        workers      = 0,          # 0 = main-process loading; avoids macOS fork() OOM
        # ── Augmentations (blueprint requirements) ──────────────────────────
        mosaic       = 1.0,        # 100% mosaic (exposes model to small objects)
        perspective  = 0.001,      # simulates 40 m viewing angle distortion
        degrees      = 10.0,
        translate    = 0.1,
        scale        = 0.5,
        fliplr       = 0.5,
        flipud       = 0.0,
        # ── Validation NMS fix ───────────────────────────────────────────────
        # The P2 head produces ~20K extra anchors. Raise conf threshold so
        # NMS only processes high-quality boxes → avoids 3s+ timeout warning.
        conf         = 0.15,       # filter low-conf anchors before NMS
        iou          = 0.5,        # slightly tighter NMS for cleaner dedup
        # ── Output ──────────────────────────────────────────────────────────
        project      = RUNS_DIR,
        name         = run_name,
        exist_ok     = True,
        patience     = 20,
        save         = True,
        save_period  = 1,          # save every epoch so resume is always possible
        plots        = True,
    )

    if resume:
        last_ckpt = os.path.join(
            "runs", "detect", RUNS_DIR, "dual_path_rdd2022", "weights", "last.pt"
        )
        # Also check the actual output dir
        alt_ckpt = os.path.join(
            "runs", "detect", RUNS_DIR, run_name, "weights", "last.pt"
        )
        ckpt = last_ckpt if os.path.exists(last_ckpt) else alt_ckpt
        if os.path.exists(ckpt):
            print(f"[Train] Resuming from {ckpt}")
            train_kwargs["resume"] = True
            model = build_dual_path_model(weights=ckpt)
        else:
            print(f"[Train] No checkpoint found. Starting fresh.")

    results = model.train(**train_kwargs)

    print("\n[Train] ✓ Training complete!")
    best = os.path.join("runs", "detect", RUNS_DIR, run_name, "weights", "best.pt")
    print(f"[Train]   Best weights: {best}")
    print(f"[Train]   mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",   action="store_true", help="Quick 5-epoch sanity check")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    run_training(fast=args.fast, resume=args.resume)
