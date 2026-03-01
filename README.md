# DPR-RIA: Dual-Path Road Infrastructure Analytics System

> **A research-grade road damage detection system** that identifies and audits potholes and cracks in real time, built on a custom YOLOv11 backbone with explainable AI, GPS asset tracking, and edge deployment support.

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4%2B-blue)](https://ultralytics.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🏗️ Architecture Overview

The system uses a **Dual-Path** design — one shared backbone feeds two specialized branches:

```
Camera Frame
    │
    ▼
┌─────────────────────────────────────────┐
│       SHARED BACKBONE (YOLOv11)         │
│  • SPD-Conv (no detail loss at 40 m)   │
│  • BMS-SPPF (cracks + potholes)        │
│  • P2/P3/P4/P5 detection heads         │
└──────────────┬──────────────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌───────────┐    ┌──────────────────┐
│  PATH 1   │    │     PATH 2       │
│ Real-Time │    │ Auditor Reports  │
│ Safety    │    │ (Explainable AI) │
│ Branch    │    │                  │
│           │    │ • DepthAnything  │
│ NMS-Free  │    │ • Grad-CAM       │
│ TensorRT  │    │ • MBTP Area Calc │
│ INT8      │    │ • IRC:82-2015    │
│ Jetson    │    │ • GeoJSON Export │
└───────────┘    └──────────────────┘
                         │
                         ▼
                   ┌──────────┐
                   │  TSCM    │
                   │ Kalman   │
                   │ Tracker  │
                   │          │
                   │ 1 pothole│
                   │ = 1 pin  │
                   └──────────┘
```

### Key Innovations

| Component | What It Solves | Where |
|---|---|---|
| **SPD-Conv** | Standard strided conv deletes distant potholes. SPD rearranges pixels into channels instead. | `src/models/spd_conv.py` |
| **BMS-SPPF** | Standard SPPF uses one pooling size. BMS adds avg→max branch to capture both elongated cracks and circular potholes simultaneously. | `src/models/bms_sppf.py` |
| **P2 Head** | Extra 160×160 detection head captures fine details at 15-40 m range that P3–P5 heads miss. | `src/models/yolo11-spd-bmssppf-p2.yaml` |
| **SAHI** | High-res images (up to 3650×2044) lose distant potholes when resized to 640. SAHI slices instead of resizes. | `src/data/sahi_inference.py` |
| **TSCM / CDKF** | Same pothole counted 30× as vehicle drives past. Kalman Filter aggregates into one unique asset. | `src/utils/tscm.py` |
| **GeoJSON Export** | Auditor-ready map of every unique defect with GPS, severity, and metric area. | `src/utils/geojson_exporter.py` |

---

## 📁 Project Structure

```
RoadDamageDetector/
├── requirements.txt
├── DPR_RIA_GPU_Training.ipynb     ← Colab/Kaggle notebook for GPU training
├── src/
│   ├── models/
│   │   ├── spd_conv.py            ← Space-to-Depth convolution layer
│   │   ├── bms_sppf.py            ← Bidirectional Multi-Scale SPPF
│   │   ├── yolo11-spd-bmssppf-p2.yaml  ← Full model architecture
│   │   ├── custom_model.py        ← Model builder (patches YOLO parser)
│   │   ├── depth_estimation.py    ← DepthAnything V2 wrapper
│   │   └── explainability.py      ← LayerCAM / Grad-CAM heatmaps
│   ├── utils/
│   │   ├── analytics.py           ← MBTP area + IRC:82-2015 severity + GPS
│   │   ├── tscm.py                ← CDKF Kalman tracker (deduplication)
│   │   └── geojson_exporter.py    ← GeoJSON FeatureCollection output
│   ├── data/
│   │   ├── rdd2022.yaml           ← Dataset config (point to your data)
│   │   ├── convert_rdd2022.py     ← VOC XML → YOLO TXT converter
│   │   └── sahi_inference.py      ← High-res SAHI sliced inference
│   └── scripts/
│       ├── train.py               ← Main training script
│       └── export_tensorrt.py     ← TensorRT INT8 export for Jetson
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd RoadDamageDetector
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset

This project uses the **[RDD2022 dataset](https://github.com/sekilab/RoadDamageDetector)** (Road Damage Detection 2022).

### Expected structure

```
datasets/
  train/
    images/    ← .jpg files
    labels/    ← .txt files (YOLO format, one per image)
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

### If your data is in Pascal VOC (XML) format

Run the converter first:

```bash
python3 src/data/convert_rdd2022.py \
    --xml_dir /path/to/annotations \
    --txt_dir /path/to/labels
```

### Update `src/data/rdd2022.yaml`

Open the file and set `path:` to your dataset root:

```yaml
path: /absolute/path/to/datasets
train: train/images
val:   valid/images
test:  test/images
nc: 4
names:
  0: D00_Longitudinal_Crack
  1: D10_Transverse_Crack
  2: D20_Alligator_Crack
  3: D40_Pothole
```

---

## 🚀 Training

### Quick sanity check (5 epochs)
```bash
python3 src/scripts/train.py --fast
```

### Full training (100 epochs)
```bash
python3 src/scripts/train.py
```

### Resume from checkpoint (after crash / interruption)
```bash
python3 src/scripts/train.py --resume
```

> **Memory note for Apple Silicon (M1/M2):** The script auto-detects MPS and sets `batch=6` to stay within unified memory limits. On NVIDIA GPUs, it uses `batch=16`.

### Train on Google Colab / Kaggle (GPU — recommended)

Open `DPR_RIA_GPU_Training.ipynb` directly in Colab or Kaggle. The notebook is fully self-contained — all custom modules are written inline. Expected training time:
- **Colab A100** → ~2–3 hrs
- **Colab/Kaggle T4** → ~8–10 hrs

---

## 🔍 Inference

### High-resolution SAHI inference (Auditor Branch)

Handles images up to 4K by slicing into overlapping 640×640 tiles instead of resizing:

```bash
python3 src/data/sahi_inference.py \
    --weights runs/train/dual_path_rdd2022/weights/best.pt \
    --image /path/to/road_image.jpg

# Or run on a whole directory:
python3 src/data/sahi_inference.py \
    --weights runs/train/dual_path_rdd2022/weights/best.pt \
    --image_dir /path/to/images/ \
    --output_json results.json
```

---

## 📊 Severity Grading (IRC:82-2015)

The system grades every detection automatically:

| Level | Pothole Depth | Action |
|---|---|---|
| 🟢 Low | < 25 mm | Routine Monitoring |
| 🟡 Medium | 25–50 mm | Patching / Sealing |
| 🔴 High | > 50 mm | Emergency Repair |

---

## 🗺️ GeoJSON Export

Every unique tracked pothole is exported as a GPS-tagged GeoJSON pin:

```bash
python3 src/utils/geojson_exporter.py
# Output: /tmp/road_assets.geojson
```

The output is compatible with any GIS software (QGIS, ArcGIS, Mapbox, Google Maps).

Sample output:
```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "geometry": { "type": "Point", "coordinates": [-122.419, 37.774] },
    "properties": {
      "id": 0,
      "class": "D40_Pothole",
      "severity_irc82": "High",
      "maintenance_action": "Emergency Repair",
      "area_cm2": 128.45,
      "depth_mm": 62.3
    }
  }]
}
```

---

## 🚢 Edge Deployment (NVIDIA Jetson Orin Nano)

Export the trained model to TensorRT INT8 for <10ms latency:

```bash
python3 src/scripts/export_tensorrt.py \
    --weights runs/train/dual_path_rdd2022/weights/best.pt \
    --int8 \
    --data src/data/rdd2022.yaml
```

This produces a `.engine` file ready for deployment on the Jetson Orin Nano.

---

## 🧠 How TSCM Works (The Novelty)

Traditional detectors count the same pothole 30–40 times as a vehicle drives past.
TSCM (Temporal-Spatial Consistency Module) solves this:

1. **Frame N (15 m away):** Pothole detected at low confidence → weak Kalman update.
2. **Frame N+15 (5 m away):** Same pothole, high confidence, precise depth → strong update dominates.
3. **Frame N+30 (past it):** Track goes stale → pruned.
4. **Output:** One GPS-tagged asset with depth/area averaged from all frames, weighted by `confidence / distance`.

```python
from src.utils.tscm import CDKFTracker
from src.utils.geojson_exporter import assets_to_geojson

tracker = CDKFTracker(match_threshold_meters=3.0)

# Feed detections frame by frame
for frame_detections in video_stream:
    tracker.update(frame_detections)

# Export unique inventory to GeoJSON
inventory = tracker.get_inventory()
assets_to_geojson(inventory, output_path="road_report.geojson")
```

---

## 📋 Requirements

- Python 3.9+
- macOS / Linux / Windows (WSL2 recommended)
- GPU: NVIDIA (CUDA) recommended for training; Apple MPS supported
- For edge deployment: NVIDIA Jetson Orin Nano + TensorRT 8.x

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
