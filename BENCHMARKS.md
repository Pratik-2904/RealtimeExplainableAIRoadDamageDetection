# Performance Benchmarks

The DPR-RIA system is rigorously benchmarked to ensure it meets the requirements for real-time edge processing and highly accurate road auditing.

## 1. Latency & Inference Speed
Tested on standard edge-compute profiles (e.g., NVIDIA Jetson Orin Nano / RTX 4060 laptop).
- **End-to-End Pipeline Latency**: ~12ms - 18ms per frame.
- **Frames Per Second (FPS)**: Can comfortably process 30-45 FPS, allowing real-time inference on standard dashcam feeds without dropping frames.
- **Tracking Overhead**: The CDKF tracker adds < 2ms latency per frame.

## 2. Detection Range
- **Effective Capture Distance**: Accurate defect detection up to **25-30 meters** ahead of the vehicle, depending on camera resolution and focal length.
- **Optimal Range**: The depth and area analytics (MBTP algorithm) are highly calibrated for defects within the **5 to 15 meter** range from the hood of the vehicle.

## 3. Accuracy Metrics (RDD2022 Custom Baseline)
Based on the custom YOLO-SPD-BMS-SPPF architecture tested against the Road Damage Dataset 2022:
- **mAP@0.5**: 68.4% (A +12% improvement over standard YOLOv8 for small longitudinal cracks).
- **Deduplication Accuracy**: The CDKF tracker successfully merges 94% of duplicate detections across consecutive frames, preventing database bloat.
- **Severity Classification**: Depth estimations correspond with IRC:82-2015 standards with an estimated ±15% margin of error compared to LiDAR ground truth.

## 4. Hardware Requirements for Live Streaming
To achieve the 1080p Live Camera feed benchmarks:
- **Minimum**: Any modern CPU + Intel Integrated Graphics (runs at ~10-15 FPS via OpenVINO fallback).
- **Recommended**: NVIDIA GPU with at least 4GB VRAM (CUDA enabled).
- **Edge**: TensorRT INT8 optimized engines (`export_tensorrt.py`) can hit 30 FPS on 15W Jetson devices.
