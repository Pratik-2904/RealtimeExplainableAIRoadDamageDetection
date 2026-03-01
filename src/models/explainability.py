import torch
import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.custom_model import build_yolo_spd_p2

# Try to import from grad-cam wrapper or provide a fallback
try:
    from pytorch_grad_cam import GradCAM, LayerCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("pytorch-grad-cam not found. Run: pip install grad-cam")

class YOLOExplainability:
    def __init__(self, weights_path="yolo11n.pt"):
        """
        Initializes the Explainability module using LayerCAM / GradCAM.
        """
        print("Initializing Explainability Module...")
        self.yolo = build_yolo_spd_p2(weights=weights_path)
        self.model = self.yolo.model # Extract PyTorch model
        self.model.eval()
        
        # In YOLOv11 with our SPD & P2 modifications, 
        # the best target layers for CAM are usually the output of the final C3k2 layers or Detect layer
        # By default we can try aiming for the last backbone layer or the P5 detection head features
        try:
            self.target_layers = [self.model.model[10]] # Example: targeting C2PSA block at end of backbone
        except:
            self.target_layers = [self.model.model[-2]] # fallback

    def generate_heatmap(self, rgb_img_array, method='layercam'):
        """
        Generates a heatmap overlay for the given image.
        """
        # Prepare input tensor
        img_tensor = torch.from_numpy(rgb_img_array).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0) # Add batch dimension
        
        # Choose CAM method
        CAMClass = LayerCAM if method.lower() == 'layercam' else GradCAM
        
        # Pytorch-grad-cam expects a model that outputs a tensor, but YOLO outputs a tuple/list.
        # We need a wrapper class to return just the logits
        class YOLOWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                # YOLO returns [preds, [activations]]
                return self.model(x)[0]
                
        wrapped_model = YOLOWrapper(self.model)
        
        with CAMClass(model=wrapped_model, target_layers=self.target_layers) as cam:
            # We don't pass targets to get the CAM for the highest scoring class
            grayscale_cam = cam(input_tensor=img_tensor)
            grayscale_cam = grayscale_cam[0, :]
            
            # Create overlay
            normalized_img = rgb_img_array.astype(np.float32) / 255.0
            visualization = show_cam_on_image(normalized_img, grayscale_cam, use_rgb=True)
            return visualization

if __name__ == "__main__":
    if os.path.exists("yolo11n.pt"):
        explainer = YOLOExplainability("yolo11n.pt")
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        heatmap = explainer.generate_heatmap(dummy_img, method='layercam')
        print(f"Generated heatmap shape: {heatmap.shape}")
