# Architecture Overview

The **DPR-RIA (Dual-Path Real-Time Intelligent Auditor)** project uses a cutting-edge hybrid deep learning architecture designed for real-time edge inference (specifically on NVIDIA Jetson devices) while maintaining high precision for civil engineering standards.

## 1. Deep Learning Backbone: Custom Dual-Path YOLO

The core detection engine is built on top of the Ultralytics YOLO framework but introduces significant architectural modifications via monkey-patching (`src/models/custom_model.py`):

- **SPD-Conv (Space-to-Depth Convolution)**: Located in `src/models/spd_conv.py`. Traditional strided convolutions or pooling layers lose fine-grained spatial information, which is critical for detecting hairline cracks. SPD-Conv rearranges spatial blocks into the channel dimension before applying a non-strided convolution, preserving 100% of the discriminative feature map for small defect detection.
- **BMS-SPPF (Bidirectional Multi-Scale Spatial Pyramid Pooling Fast)**: Located in `src/models/bms_sppf.py`. This module replaces the standard SPPF layer. By passing features bidirectionally across multiple pooling scales, it vastly improves the contextual awareness of the model, allowing it to differentiate between actual potholes and simple shadows or road patches.

## 2. Tracking and Deduplication: CDKF

Processing video frame-by-frame inevitably leads to the same pothole being detected multiple times as the vehicle moves. To solve this, we use a **Custom Dual Kalman Filter (CDKF)** (`src/utils/tscm.py`):

- **State Estimation**: It tracks not just bounding box coordinates, but estimated physical world locations.
- **Data Association**: It uses the Haversine formula to calculate the distance between a new detection and existing tracked assets. If a detection falls within the `match_threshold_meters` (e.g., 3.0m), it updates the existing asset rather than creating a duplicate.
- **Persistence**: Assets are maintained in an "active" state until they leave the frame and expire, at which point they are archived for final export.

## 3. Physical Area Estimation: MBTP

Unlike standard 2D bounding boxes, road repair requires physical metric volumes.
- **Minimum Bounding Triangulated Pixel (MBTP)**: (`src/utils/analytics.py`). This algorithm approximates the physical area of a defect. Using the focal length of the camera and an estimated depth, it projects the 2D bounding box pixels into a 3D metric plane to calculate the surface area (cm²) needed for patching material estimates.

## 4. Geolocation: GPS Back-projection

Using the vehicle's current GPS location and heading, `backproject_to_gps` (`src/utils/analytics.py`) casts a ray through the inverse camera intrinsic matrix to map the pixel coordinates of the defect onto the standard Earth radius, providing precise (Latitude, Longitude) coordinates for work crews.
