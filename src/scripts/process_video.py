"""
Live Video Pipeline — Dual-Path Road Infrastructure Analytics
=============================================================
Runs the full End-to-End Safety + Auditor pipeline on a video file.

Process per frame:
1. Run YOLO Dual-Path Model (SPDConv + BMS-SPPF) -> Detections
2. Calculate depth (mocked from bounding box y-pos for speed, or DepthAnything)
3. Calculate physical area (MBTP algorithm)
4. Feed to CDKF Tracker -> Deduplicates into master assets across frames
5. Render bounding boxes, IDs, and severity onto the output frame

Final output:
- Annotated video (.mp4)
- GeoJSON map of all unique defects
"""

import os
import sys
import argparse
import cv2s
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.custom_model import build_dual_path_model
from src.utils.tscm import CDKFTracker, irc_severity
from src.utils.analytics import backproject_to_gps
from src.utils.geojson_exporter import assets_to_geojson

CLASS_NAMES = {
    0: "D00_Longitudinal_Crack",
    1: "D10_Transverse_Crack",
    2: "D20_Alligator_Crack",
    3: "D40_Pothole"
}
CLASS_COLORS = {
    0: (0, 255, 136),   # Green
    1: (0, 170, 255),   # Orange
    2: (0, 68, 255),    # Red
    3: (255, 0, 170)    # Purple
}


def process_video(video_path: str, weights_path: str, output_path: str):
    print(f"[Video] Loading model from {weights_path}...")
    model = build_dual_path_model(weights=weights_path)
    
    # Initialize Tracker (matches objects passing by over 40 frames)
    tracker = CDKFTracker(match_threshold_meters=3.0, max_frames_unmatched=40)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Mock Vehicle Telemetry (San Francisco start)
    base_lat, base_lon = 37.7749, -122.4194
    speed_mps = 15.0  # 54 km/h

    print(f"[Video] Processing {total_frames} frames at {width}x{height} @ {fps} FPS")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Simulate car moving (0.0001 deg lat ~= 11 meters)
        current_lat = base_lat + (frame_idx * speed_mps / fps) / 111320.0
        current_lon = base_lon

        # 1. Inference
        # iou=0.6 prevents large close-up boxes from being suppressed by NMS
        results = model.predict(frame, conf=0.15, imgsz=1280, iou=0.6, max_det=300, verbose=False)
        frame_detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = CLASS_NAMES.get(cls, f"Class_{cls}")

                # 2. Analytics (Approximations for fast video rendering)
                # Distance: lower in image = closer
                center_y = (y1 + y2) / 2
                distance_m = max(3.0, 40.0 * (1.0 - (center_y / height)))
                
                # Area and Depth (Simulated using box size and distance)
                box_sq_px = (x2 - x1) * (y2 - y1)
                area_cm2 = (box_sq_px / 1000.0) * (distance_m * 0.5)
                depth_mm = min(80.0, area_cm2 * 0.4) 

                # Geometry -> GPS
                det_lat, det_lon = backproject_to_gps(
                    u=(x1+x2)/2, v=center_y, z=distance_m,
                    K_inv=np.eye(3), camera_height_m=1.5,
                    vehicle_gps=(current_lat, current_lon), heading_rad=0.0
                )

                frame_detections.append({
                    "gps": (det_lat, det_lon),
                    "area": area_cm2,
                    "depth": depth_mm,
                    "confidence": conf,
                    "distance": distance_m,
                    "class_name": class_name,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                })

        # 3. TSCM Tracking
        tracker.update(frame_detections)

        # 4. Render current frame tracked assets
        # To display IDs, we match the box back to the tracker's current active assets
        for det in frame_detections:
            x1, y1, x2, y2 = det["bbox"]
            color = CLASS_COLORS.get(list(CLASS_NAMES.values()).index(det["class_name"]), (255, 255, 255))
            
            # Find closest asset ID
            best_id = "?"
            best_sev = "Unknown"
            min_d = float('inf')
            for track in tracker.assets:
                d = CDKFTracker._haversine(det["gps"], (track.state[0], track.state[1]))
                if d < min_d:
                    min_d = d
                    best_id = str(track.asset_id)
                    best_sev, _ = irc_severity(track.state[3])

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"ID:{best_id} | {det['class_name']} | Sev: {best_sev}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw Telemetry Overlay
        cv2.putText(frame, f"DPR-RIA Live | Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Active Master Assets: {len(tracker.assets)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            print(f"[Video] Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    
    # 5. Export GeoJSON
    inventory = tracker.get_inventory()
    geojson_path = output_path.replace(".mp4", "_assets.geojson")
    assets_to_geojson(inventory, geojson_path)
    
    # Print summary
    print("\n" + "="*50)
    print(f"✅ Video processing complete: {output_path}")
    print(f"✅ Unique tracked defects:   {len(inventory)}")
    print(f"✅ GeoJSON exported to:      {geojson_path}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="Model weights")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    args = parser.parse_args()

    process_video(args.video, args.weights, args.output)
