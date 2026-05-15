"""
DPR-RIA — Live Road Defect Auditor Web Dashboard
=================================================
Flask + SocketIO backend that serves the Stitch-designed
Cyber-Tactical UI and streams live video inference results
via WebSockets.

Usage:
    python src/ui/app.py
    # Then open http://localhost:5000
"""

import os
import sys
import json
import time
import base64
import threading
import cv2
import numpy as np

# ─── Path setup ───
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from google import genai
from google.genai import types

from src.models.custom_model import build_dual_path_model
from src.utils.tscm import CDKFTracker, irc_severity
from src.utils.analytics import backproject_to_gps
from src.utils.geojson_exporter import assets_to_geojson

# ─── App Setup ───
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['SECRET_KEY'] = 'dpr-ria-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_NAMES = {
    0: "D00_Longitudinal_Crack",
    1: "D10_Transverse_Crack",
    2: "D20_Alligator_Crack",
    3: "D40_Pothole"
}

# ─── Global State ───
pipeline_state = {
    "running": False,
    "model": None,
    "tracker": None,
    "frame_idx": 0,
    "total_frames": 0,
}


# ─── Routes ───
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload via drag-and-drop or file picker."""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    video_path = os.path.join(RESULTS_DIR, "input_video.mp4")
    file.save(video_path)

    # Get video metadata
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return jsonify({
        "status": "ok",
        "filename": file.filename,
        "frames": total_frames,
        "fps": fps,
        "resolution": f"{width}x{height}",
    })


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(RESULTS_DIR, filename, as_attachment=True)


# ─── WebSocket Events ───
@socketio.on('start_pipeline')
def handle_start_pipeline():
    """Starts the full video inference pipeline in a background thread."""
    if pipeline_state["running"]:
        emit('pipeline_status', {"status": "already_running"})
        return

    video_path = os.path.join(RESULTS_DIR, "input_video.mp4")
    if not os.path.exists(video_path):
        # Try sample
        sample_path = os.path.join(RESULTS_DIR, "sample_dashcam.mp4")
        if os.path.exists(sample_path):
            video_path = sample_path
        else:
            emit('pipeline_status', {"status": "no_video"})
            return

    pipeline_state["running"] = True
    emit('pipeline_status', {"status": "loading_model"})

    # Run in background thread
    thread = threading.Thread(target=run_pipeline, args=(video_path,))
    thread.daemon = True
    thread.start()


@socketio.on('start_live_camera')
def handle_start_live_camera():
    """Starts the full video inference pipeline using the live webcam."""
    if pipeline_state["running"]:
        emit('pipeline_status', {"status": "already_running"})
        return

    pipeline_state["running"] = True
    emit('pipeline_status', {"status": "loading_model"})

    # Run in background thread using source 0 (default camera)
    thread = threading.Thread(target=run_pipeline, args=(0,))
    thread.daemon = True
    thread.start()


@socketio.on('stop_pipeline')
def handle_stop_pipeline():
    pipeline_state["running"] = False
    emit('pipeline_status', {"status": "stopped"})


def run_pipeline(video_path_or_cam_id):
    """Main pipeline loop — runs inference frame-by-frame and emits results via WebSocket."""
    try:
        # 1. Load Model
        socketio.emit('pipeline_status', {"status": "loading_model"})
        weights_path = os.path.join(os.path.dirname(__file__), '..', '..', 'best.pt')
        if not os.path.exists(weights_path):
            weights_path = os.path.join(os.path.dirname(__file__), '..', '..', 'yolo11n.pt')
        model = build_dual_path_model(weights=weights_path)

        # 2. Init Tracker
        tracker = CDKFTracker(match_threshold_meters=3.0, max_frames_unmatched=40)
        pipeline_state["tracker"] = tracker

        # 3. Open Video or Camera
        is_live_camera = isinstance(video_path_or_cam_id, int)
        cap = cv2.VideoCapture(video_path_or_cam_id)
        if not cap.isOpened():
            socketio.emit('pipeline_status', {"status": "error", "message": "Cannot open video source"})
            pipeline_state["running"] = False
            return

        # Camera-specific optimizations to reduce lag
        if is_live_camera:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer to get latest frame
            cap.set(cv2.CAP_PROP_FPS, 30)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pipeline_state["total_frames"] = total_frames

        # Output video (save at native camera resolution for future use)
        out_video_path = os.path.join(RESULTS_DIR, "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        # Mock GPS
        base_lat, base_lon = 37.7749, -122.4194
        speed_mps = 15.0

        socketio.emit('pipeline_status', {"status": "running"})

        frame_idx = 0
        inference_times = []
        last_emit_time = 0
        EMIT_INTERVAL = 1.0 / 15  # Cap streaming at ~15fps to reduce client lag

        while cap.isOpened() and pipeline_state["running"]:
            ret, frame = cap.read()
            if not ret:
                break

            # For live camera: drain buffer to always get latest frame
            if is_live_camera:
                for _ in range(2):  # Grab & discard 2 buffered frames
                    cap.grab()

            t_start = time.time()

            current_lat = base_lat + (frame_idx * speed_mps / fps) / 111320.0
            current_lon = base_lon

            # Inference
            results = model.predict(frame, conf=0.15, imgsz=1280, iou=0.6,
                                    augment=True, max_det=300, verbose=False)

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
                        u=(x1 + x2) / 2, v=center_y, z=distance_m,
                        K_inv=np.eye(3), camera_height_m=1.5,
                        vehicle_gps=(current_lat, current_lon), heading_rad=0.0
                    )

                    sev, action = irc_severity(depth_mm)
                    frame_detections.append({
                        "gps": (det_lat, det_lon),
                        "area": area_cm2,
                        "depth": depth_mm,
                        "confidence": conf,
                        "distance": distance_m,
                        "class_name": class_name,
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "cls_id": cls,
                        "severity": sev,
                    })

            # Tracking
            tracker.update(frame_detections)

            # Render overlays on frame
            for det in frame_detections:
                x1, y1, x2, y2 = det["bbox"]
                sev = det["severity"]

                if sev == "High":
                    color = (0, 0, 255)
                elif sev == "Medium":
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 0)

                is_pothole = "Pothole" in det["class_name"]
                if is_pothole:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{det['class_name'].split('_')[-1]} {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame)

            t_end = time.time()
            latency_ms = (t_end - t_start) * 1000
            inference_times.append(latency_ms)

            # Encode frame as JPEG for streaming
            jpeg_quality = 60 if is_live_camera else 70
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')

            # Build inventory for metrics
            inventory = tracker.get_inventory()
            total_detections = len(inventory)
            critical_count = sum(1 for a in inventory if a["severity_irc82"] == "High")
            avg_depth = (sum(a["depth_mm"] for a in inventory) / total_detections) if total_detections > 0 else 0
            active_tracks = len(tracker.assets)

            # Sort inventory by severity priority: High > Medium > Low
            SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}
            sorted_inventory = sorted(inventory, key=lambda x: SEVERITY_ORDER.get(x["severity_irc82"], 3))

            # Build detection rows for the table from the sorted inventory (top 20)
            detection_rows = []
            for item in sorted_inventory[:20]:
                c_name = item["class"].replace("D00_", "").replace("D10_", "").replace("D20_", "").replace("D40_", "")
                detection_rows.append({
                    "id": f"DF-{item['id']:04d}",
                    "type": c_name,
                    "distance": "-",
                    "area": f"{item['area_cm2']:.0f}",
                    "severity": item["severity_irc82"],
                })

            # Throttle emission to prevent overwhelming the client
            now = time.time()
            if now - last_emit_time >= EMIT_INTERVAL:
                last_emit_time = now
                socketio.emit('frame_update', {
                    "frame": frame_b64,
                    "frame_idx": frame_idx,
                    "total_frames": total_frames,
                    "progress": round((frame_idx / max(total_frames, 1)) * 100, 1),
                    "latency_ms": round(latency_ms, 1),
                    "metrics": {
                        "total_detections": total_detections,
                        "critical_count": critical_count,
                        "avg_depth": round(avg_depth, 1),
                        "active_tracks": active_tracks,
                    },
                    "detections": detection_rows,
                })

            frame_idx += 1
            pipeline_state["frame_idx"] = frame_idx

        cap.release()
        out.release()

        # Export GeoJSON
        inventory = tracker.get_inventory()
        out_geojson_path = os.path.join(RESULTS_DIR, "output_assets.geojson")
        assets_to_geojson(inventory, out_geojson_path)

        socketio.emit('pipeline_status', {
            "status": "complete",
            "total_assets": len(inventory),
            "geojson_path": "output_assets.geojson",
            "video_path": "output.mp4",
        })

    except Exception as e:
        socketio.emit('pipeline_status', {"status": "error", "message": str(e)})
        import traceback
        traceback.print_exc()
    finally:
        pipeline_state["running"] = False


# ─── Main ───
@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json or {}
    api_key = data.get('api_key') or os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        return jsonify({"error": "No API Key provided. Please provide a Gemini API Key."}), 400

    geojson_path = os.path.join(RESULTS_DIR, "output_assets.geojson")
    if not os.path.exists(geojson_path):
        return jsonify({"error": "No data available yet. Please run the pipeline first."}), 404

    try:
        with open(geojson_path, 'r') as f:
            geo_data = json.load(f)

        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        You are an expert Civil Engineering Auditor. Analyze this GeoJSON export of road defects detected by our system.
        Provide a concise, professional "End-of-Shift Maintenance Report".
        Include:
        1. Total defects found.
        2. Breakdown by severity (High, Medium, Low).
        3. A paragraph summarizing the overall condition and highlighting any 'High' severity emergency patches needed immediately.
        4. Estimated patching materials required based on the area_cm2 and depth_mm metrics.
        
        GeoJSON Data:
        {json.dumps(geo_data)[:30000]} # Trimmed to avoid exceeding context if huge
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return jsonify({"report": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json or {}
    api_key = data.get('api_key') or os.environ.get('GEMINI_API_KEY')
    message = data.get('message', '')
    history = data.get('history', [])
    
    if not api_key:
        return jsonify({"error": "No API Key provided."}), 400

    geojson_path = os.path.join(RESULTS_DIR, "output_assets.geojson")
    geo_data_str = "No data yet."
    if os.path.exists(geojson_path):
        with open(geojson_path, 'r') as f:
            geo_data_str = json.dumps(json.load(f))[:20000]

    try:
        client = genai.Client(api_key=api_key)
        
        # Convert simple history to GenAI history format
        contents = [
            types.Content(role="user", parts=[
                types.Part.from_text(text=f"System Context: You are a helpful Civil Engineering Assistant answering questions about this road survey data: {geo_data_str}")
            ]),
            types.Content(role="model", parts=[types.Part.from_text(text="Understood. How can I help you analyze the road survey data?")])
        ]
        
        for msg in history:
            role = "user" if msg['role'] == 'user' else "model"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg['content'])]))
            
        # Add current message
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
        )
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  DPR-RIA — Live Road Defect Auditor")
    print("  Open http://localhost:5001 in your browser")
    print("=" * 60 + "\n")
    socketio.run(app, host='0.0.0.0', port=5001, debug=False, allow_unsafe_werkzeug=True)