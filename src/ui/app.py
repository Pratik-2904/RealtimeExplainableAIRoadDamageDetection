import streamlit as st
import ast
import urllib.parse
import os
import sys
import json
import pandas as pd
import cv2
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.custom_model import build_dual_path_model
from src.utils.tscm import CDKFTracker, irc_severity
from src.utils.analytics import backproject_to_gps
from src.utils.geojson_exporter import assets_to_geojson

# ─── Config ───
st.set_page_config(page_title="DPR-RIA Dashboard", layout="wide", page_icon="🛣️")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

# ─── Sidebar ───
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/2/21/Pothole.jpg", width=250)
    st.title("DPR-RIA System")
    st.write("**Dual-Path Road Infrastructure Analytics**")
    st.write("---")
    st.markdown("""
        **Modules Active:**
        - ✅ YOLO SPD-Conv Backbone
        - ✅ BMS-SPPF Pooling
        - ✅ CDKF Tracking (TSCM)
        - ✅ IRC:82-2015 Severity
    """)
    st.write("---")
    
    st.write("### 1. Upload Video")
    uploaded_video = st.file_uploader("Upload Raw MP4 Dashcam Video", type=['mp4'])
    video_path = None
    if uploaded_video:
        video_path = os.path.join(RESULTS_DIR, "input_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
        st.success("Video Uploaded!")

    elif os.path.exists(os.path.join(RESULTS_DIR, "sample_dashcam.mp4")):
        # Fallback to test video
        video_path = os.path.join(RESULTS_DIR, "sample_dashcam.mp4")
        st.info("Using sample dashcam video.")

    st.write("### 2. Live Processing")
    run_live = False
    if video_path:
        run_live = st.button("▶️ Run Live Video Pipeline", type="primary", use_container_width=True)

# ─── Header ───
st.title("🛣️ Live Road Defect Auditor")
st.write("Watch the Dual-Path model detect, track, and grade road hazards in real-time.")

# ─── Live Video Processing Logic ───
if run_live and video_path:
    st.markdown("---")
    col_vid, col_met = st.columns([2, 1])
    
    with col_vid:
        st.subheader("📹 Real-Time Safety Feed")
        video_placeholder = st.empty()
        
    with col_met:
        st.subheader("📊 Live Metrics")
        frames_metric = st.empty()
        assets_metric = st.empty()
        status_text = st.empty()
        st.markdown("**Active Detections Stream**")
        live_table = st.empty()
        
    status_text.info("Loading Custom Dual-Path Model (SPD-Conv + BMS-SPPF)...")
    
    # Load model
    model = build_dual_path_model(weights="best.pt")
    
    # Initialize Tracker
    tracker = CDKFTracker(match_threshold_meters=3.0, max_frames_unmatched=40)
    
    # Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Cannot open video format for {video_path}")
        st.stop()
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    out_video_path = os.path.join(RESULTS_DIR, "output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    
    # Mock Start GPS
    base_lat, base_lon = 37.7749, -122.4194
    speed_mps = 15.0  # 54 km/h

    status_text.success("Pipeline running! Extracting topology...")
    
    progress_bar = st.progress(0)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_lat = base_lat + (frame_idx * speed_mps / fps) / 111320.0
        current_lon = base_lon

        # 1. Inference
        # Force a higher resolution (imgsz=1280) instead of the default 640.
        # This gives the SPD-Conv and P2 head the raw pixels they need to find potholes at 40m.
        # Lowered conf to 0.15 because distant objects naturally have lower confidence in early epochs.
        # Increased iou=0.6 and max_det=300 so large bounding boxes aren't aggressively suppressed by NMS
        # Added augment=True (Test-Time Augmentation) to drastically improve detection on difficult objects (like wet/large potholes)
        results = model.predict(frame, conf=0.15, imgsz=1280, iou=0.6, augment=True, max_det=300, verbose=False)
        frame_detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = CLASS_NAMES.get(cls, f"Class_{cls}")

                center_y = (y1 + y2) / 2
                distance_m = max(3.0, 40.0 * (1.0 - (center_y / height)))
                box_sq_px = (x2 - x1) * (y2 - y1)
                area_cm2 = (box_sq_px / 1000.0) * (distance_m * 0.5)
                depth_mm = min(80.0, area_cm2 * 0.4) 

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
                    "cls_id": cls,
                })

        # 1b. Merge overlapping boxes of the same class into one big box
        # This prevents a large pothole from being fragmented into many small detections
        def _iou(b1, b2):
            ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
            ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
            a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
            return inter / (a1 + a2 - inter + 1e-6)
        
        def _contained(small, big):
            """Check if small box is mostly inside big box (>50% overlap)"""
            ix1 = max(small[0], big[0]); iy1 = max(small[1], big[1])
            ix2 = min(small[2], big[2]); iy2 = min(small[3], big[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            small_area = (small[2]-small[0]) * (small[3]-small[1]) + 1e-6
            return (inter / small_area) > 0.5

        # Group by class
        from collections import defaultdict
        class_groups = defaultdict(list)
        for det in frame_detections:
            class_groups[det["cls_id"]].append(det)
        
        merged_detections = []
        for cls_id, dets in class_groups.items():
            # Sort by area descending so biggest box comes first
            dets.sort(key=lambda d: (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1]), reverse=True)
            used = [False] * len(dets)
            
            for i in range(len(dets)):
                if used[i]:
                    continue
                # Start with this box as the anchor
                bx1, by1, bx2, by2 = dets[i]["bbox"]
                best_conf = dets[i]["confidence"]
                
                # Absorb any smaller boxes that overlap with it
                for j in range(i + 1, len(dets)):
                    if used[j]:
                        continue
                    if _iou(dets[i]["bbox"], dets[j]["bbox"]) > 0.15 or _contained(dets[j]["bbox"], (bx1, by1, bx2, by2)):
                        # Expand the union bounding box
                        jb = dets[j]["bbox"]
                        bx1 = min(bx1, jb[0])
                        by1 = min(by1, jb[1])
                        bx2 = max(bx2, jb[2])
                        by2 = max(by2, jb[3])
                        best_conf = max(best_conf, dets[j]["confidence"])
                        used[j] = True
                
                # Recalculate area/depth/distance for the merged box
                m_center_y = (by1 + by2) / 2
                m_distance = max(3.0, 40.0 * (1.0 - (m_center_y / height)))
                m_box_sq_px = (bx2 - bx1) * (by2 - by1)
                m_area_cm2 = (m_box_sq_px / 1000.0) * (m_distance * 0.5)
                m_depth_mm = min(80.0, m_area_cm2 * 0.4)
                
                merged_det = dict(dets[i])  # copy all fields
                merged_det["bbox"] = (bx1, by1, bx2, by2)
                merged_det["confidence"] = best_conf
                merged_det["area"] = m_area_cm2
                merged_det["depth"] = m_depth_mm
                merged_det["distance"] = m_distance
                merged_detections.append(merged_det)
        
        frame_detections = merged_detections

        # 2. Tracking
        tracker.update(frame_detections)

        # 3. Render
        for det in frame_detections:
            x1, y1, x2, y2 = det["bbox"]
            
            best_id = "?"
            best_sev = "Unknown"
            min_d = float('inf')
            for track in tracker.assets:
                d = CDKFTracker._haversine(det["gps"], (track.state[0], track.state[1]))
                if d < min_d:
                    min_d = d
                    best_id = str(track.asset_id)
                    best_sev, _ = irc_severity(track.state[3])

            # Determine color by severity (BGR format for OpenCV)
            if best_sev == "High":
                color = (0, 0, 255)      # Red
            elif best_sev == "Medium":
                color = (0, 165, 255)    # Orange
            elif best_sev == "Low":
                color = (0, 255, 0)      # Green
            else:
                color = (255, 255, 255)  # White

            # If it's a pothole, shade the entire bounding box area
            # If it's a crack, we only want the outline
            is_pothole = "Pothole" in det["class_name"]
            
            if is_pothole:
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1) # -1 means filled
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

            # Always draw the outline box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{best_id} | {det['distance']:.1f}m | {det['area']:.0f}cm2 | {best_sev}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Save these back to det for the table
            det["best_id"] = best_id
            det["best_sev"] = best_sev

        cv2.putText(frame, f"DPR-RIA Live | Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        out.write(frame)
        
        # UI Updates (Convert BGR to RGB for Streamlit)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Live Table Data (Persistent across all frames)
        full_inventory = tracker.get_inventory()
        if len(full_inventory) > 0:
            tracked_data = []
            for t_data in full_inventory:
                c_name = t_data["class"].replace("D00_", "").replace("D10_", "").replace("D20_", "").replace("D40_", "")
                
                tracked_data.append({
                    "ID": str(t_data["id"]),
                    "Type": c_name,
                    "Dist (m)": f"{t_data.get('last_distance', 0.0):.1f}",    # distance might need to be passed through to_dict
                    "Area (cm²)": f"{t_data['area_cm2']:.1f}",  # area
                    "Depth (mm)": f"{t_data['depth_mm']:.1f}",  # depth
                    "Severity": t_data["severity_irc82"],
                    "Frames Seen": t_data["total_detections"]
                })
            
            live_df = pd.DataFrame(tracked_data)
            # Sort by ID descending so newest are at the top
            live_df = live_df.sort_values(by="ID", ascending=False)
            live_table.dataframe(live_df, use_container_width=True, hide_index=True)
        else:
            live_table.info("No hazards tracked yet.")
        
        frame_idx += 1

    cap.release()
    out.release()
    
    # 4. Export Final Inventory
    status_text.info("Video finished! Exporting Auditor GeoJSON Inventory...")
    inventory = tracker.get_inventory()
    out_geojson_path = os.path.join(RESULTS_DIR, "output_assets.geojson")
    assets_to_geojson(inventory, out_geojson_path)
    
    status_text.success("Pipeline Full Execution Complete! See final report below.")
    st.session_state["run_live_complete"] = True

# ─── Display Final Tables ───
geojson_file = os.path.join(RESULTS_DIR, "output_assets.geojson")
if os.path.exists(geojson_file) and (not run_live or st.session_state.get("run_live_complete")):
    st.markdown("---")
    st.subheader("🏁 Final Auditor Feature Inventory (IRC:82-2015)")
    
    with open(geojson_file, 'r') as f:
        geo_data = json.load(f)

    data_rows = []
    for feat in geo_data.get("features", []):
        props = feat["properties"]
        coords = feat.get("geometry", {}).get("coordinates", [0, 0])
        props["latitude"] = coords[1]
        props["longitude"] = coords[0]
        data_rows.append(props)

    df = pd.DataFrame(data_rows)

    if df.empty:
        st.info("No defects found in the processed report.")
    else:
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        high_sev = len(df[df["severity_irc82"] == "High"])
        med_sev = len(df[df["severity_irc82"] == "Medium"])

        col1.metric("Total Unique Defects", len(df), f"Tracked via CDKF")
        col2.metric("Critical Hazards", high_sev, "Require Emergency Repair", delta_color="inverse")
        col3.metric("Medium Severity", med_sev, "Patching recommended")
        col4.metric("Average Depth", f"{df['depth_mm'].mean():.1f} mm")

        # Tables
        table_df = df[["id", "class", "severity_irc82", "depth_mm", "area_cm2", "frames_tracked", "latitude", "longitude", "maintenance_action"]]
        st.dataframe(table_df, use_container_width=True)

        col_csv, col_json = st.columns(2)
        with col_csv:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV Report", csv, "road_defects.csv", "text/csv")
        with col_json:
            with open(geojson_file, "rb") as f:
                st.download_button("Download Raw GeoJSON (GIS Ready)", f, "road_defects.geojson", "application/geo+json")

            