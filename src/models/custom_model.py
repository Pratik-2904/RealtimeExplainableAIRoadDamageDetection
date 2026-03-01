"""
custom_model.py — Dual-Path Shared Backbone Builder
====================================================
Registers both custom modules (SPDConv, BMS-SPPF) into the Ultralytics
parser before building from our YAML spec.

Monkey-patch strategy
---------------------
Ultralytics `parse_model` resolves module names via `globals()` in tasks.py.
We hijack two existing names whose signature matches ours:
  • `Focus`  → SPDConv   (both take c1, c2 and go through Conv internally)
  • `Focus2` → BMS_SPPF  (we patch a unused slot; 'Focus2' triggers no conflict)
"""

import sys
import os
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ultralytics import YOLO
import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks

from src.models.spd_conv import SPDConv
from src.models.bms_sppf import BMS_SPPF


# ── Register SPDConv as 'Focus' ──────────────────────────────────────────────
# parse_model will call Focus(c1, c2) which matches SPDConv.__init__(c1, c2, ...).
setattr(modules, "Focus", SPDConv)
setattr(tasks, "Focus", SPDConv)

# ── Register BMS_SPPF as 'SPPF' ─────────────────────────────────────────────
# SPPF is in base_modules so parse_model injects (c1, c2, k) correctly.
setattr(modules, "SPPF", BMS_SPPF)
setattr(tasks, "SPPF", BMS_SPPF)


def build_dual_path_model(
    yaml_file: str = "src/models/yolo11-spd-bmssppf-p2.yaml",
    weights: Optional[str] = None,
) -> YOLO:
    """
    Constructs the full Dual-Path backbone:
      • SPD-Conv layers (Focus proxy) for range-preserving downsampling.
      • BMS-SPPF layer (Focus2 proxy) for multi-scale crack + pothole receptive fields.
      • P2/P3/P4/P5 detection heads — 40 m range enabled by P2.

    Parameters
    ----------
    yaml_file : Model architecture spec.
    weights   : Optional .pt file; partial-weight load attempted if supplied.
    """
    print(f"[DualPath] Building from {yaml_file}")
    model = YOLO(yaml_file)

    if weights and os.path.exists(weights):
        print(f"[DualPath] Loading weights from {weights}")
        try:
            model.load(weights)
        except Exception as e:
            print(f"[DualPath] Warning — partial load: {e}")

    return model


# Alias kept for backward compatibility with existing scripts
def build_yolo_spd_p2(yaml_file="src/models/yolo11-spd-p2.yaml", weights=None, scale="s"):
    return build_dual_path_model(yaml_file=yaml_file, weights=weights)


if __name__ == "__main__":
    model = build_dual_path_model()
    print("[DualPath] Model built successfully ✓")
    model.info()
