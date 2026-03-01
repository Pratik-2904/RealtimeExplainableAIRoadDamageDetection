import torch
import cv2
import numpy as np
from transformers import pipeline
from PIL import Image

class MetricDepthEstimator:
    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Small-hf", device="cpu"):
        """
        Initializes the DepthAnything V2 model for generating metric depth maps.
        """
        print(f"Loading Depth Estimation model: {model_id} on {device}...")
        try:
            # We use depth-estimation pipeline. Depending on actual HF repo, the name might vary.
            self.pipe = pipeline(task="depth-estimation", model=model_id, device=0 if device == "cuda" else -1)
        except Exception as e:
            print(f"Failed to load specific V2 model, falling back to v1. Error: {e}")
            self.pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device=0 if device == "cuda" else -1)

    def estimate_depth(self, image_path_or_array):
        """
        Estimates the depth map of an image.
        Returns the depth map as a numpy array.
        """
        if isinstance(image_path_or_array, str):
            image = Image.open(image_path_or_array).convert('RGB')
        elif isinstance(image_path_or_array, np.ndarray):
            # assume BGR from cv2, convert to RGB PIL
            image_rgb = cv2.cvtColor(image_path_or_array, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)
        else:
            image = image_path_or_array
            
        # The pipeline returns a dict with 'depth' (PIL Image) and 'predicted_depth' (tensor)
        result = self.pipe(image)
        
        # 'predicted_depth' contains the raw tensor output (unscaled relative depth)
        # To get metric depth, we would typically multiply by a scaling factor or use calibration
        # For this prototype, we return the raw predicted depth map scaled to image size
        depth_tensor = result['predicted_depth'][0]
        depth_map = depth_tensor.cpu().numpy()
        
        # Resize to original image size
        original_width, original_height = image.size
        depth_map_resized = cv2.resize(depth_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        
        return depth_map_resized

if __name__ == "__main__":
    import numpy as np
    estimator = MetricDepthEstimator()
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = estimator.estimate_depth(dummy_img)
    print(f"Estimated depth map shape: {depth.shape}")
