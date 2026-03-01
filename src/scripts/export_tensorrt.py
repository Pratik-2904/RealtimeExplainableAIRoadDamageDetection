import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.custom_model import build_yolo_spd_p2

def export_model_to_tensorrt(weights_path, img_size=640, int8=False, data_yaml=None):
    """
    Exports the custom YOLO architecture to NVIDIA TensorRT.
    By designing the detection head to output direct bounding boxes (NMS-free config), 
    we reduce post-processing latency for the safety branch.
    """
    print(f"Loading custom model from {weights_path}...")
    model = build_yolo_spd_p2(weights=weights_path)
    
    export_kwargs = {
        'format': 'engine',
        'half': not int8,
        'int8': int8,
        'imgsz': img_size,
        'simplify': True, 
        'workspace': 4,
    }
    
    if int8:
        if data_yaml is None:
            print("WARNING: INT8 calibration requires a subset of data. No data.yaml provided.")
            print("Will attempt export, but calibration may fail or be inaccurate.")
        else:
            export_kwargs['data'] = data_yaml
            
    print(f"Starting TensorRT export with {export_kwargs}...")
    try:
        engine_path = model.export(**export_kwargs)
        print(f"Success! Model successfully exported to: {engine_path}")
    except Exception as e:
        print(f"Export failed. Please ensure 'tensorrt' python package and system libraries are installed. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Shared Backbone to TensorRT")
    parser.add_argument("--weights", type=str, help="Path to trained .pt weights", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--int8", action="store_true", help="Enable INT8 Quantization")
    parser.add_argument("--data", type=str, default=None, help="data.yaml path for calibration")
    
    args = parser.parse_args()
    
    # We create a dummy weights file or download one if none exists just to show the pipeline works
    if not os.path.exists(args.weights) and args.weights == "yolo11n.pt":
        from ultralytics.utils.downloads import download
        download("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt")
        
    export_model_to_tensorrt(args.weights, args.imgsz, args.int8, args.data)
