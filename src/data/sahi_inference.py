"""
SAHI Inference Pipeline — Auditor Analytics Branch
====================================================
Handles variable-resolution images (512×512 → 3650×2044 from RDD2022).

Instead of resize-and-lose, we:
  1. Slice the high-res image into overlapping 640×640 tiles.
  2. Run YOLO inference on each tile.
  3. Stitch results back with NMS to deduplicate cross-tile boxes.

This prevents the "vanishing pothole" problem at 30-40 m range.
"""

import os
import sys
import cv2
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ─── Class map matching our YAML ──────────────────────────────────────────────
CLASSES = {
    0: "D00_Longitudinal_Crack",
    1: "D10_Transverse_Crack",
    2: "D20_Alligator_Crack",
    3: "D40_Pothole",
}


class SAHIAuditorPipeline:
    """
    SAHI-powered inference pipeline for the Expert Auditor Analytics Branch.
    Designed for high-res images where standard resizing destroys distant details.
    """

    def __init__(
        self,
        weights_path: str = "yolo11n.pt",
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_ratio: float = 0.20,
        confidence_threshold: float = 0.25,
        device: str = "cpu",
    ):
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = overlap_ratio
        self.confidence_threshold = confidence_threshold

        print(f"[SAHI] Loading detection model from {weights_path} on {device}...")
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=weights_path,
            confidence_threshold=confidence_threshold,
            device=device,
        )

    def predict_image(self, image_path: str) -> dict:
        """
        Run SAHI sliced inference on a single image.

        Returns
        -------
        dict with keys:
          image_path, detections: list of {class, confidence, bbox_xyxy}
        """
        result = get_sliced_prediction(
            image=image_path,
            detection_model=self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            verbose=0,
        )

        detections = []
        for obj in result.object_prediction_list:
            bbox = obj.bbox.to_xyxy()  # [x1, y1, x2, y2]
            center_u = int((bbox[0] + bbox[2]) / 2)
            center_v = int((bbox[1] + bbox[3]) / 2)
            detections.append({
                "class_id": obj.category.id,
                "class_name": CLASSES.get(obj.category.id, f"cls_{obj.category.id}"),
                "confidence": float(obj.score.value),
                "bbox_xyxy": [float(x) for x in bbox],
                "center_uv": [center_u, center_v],
            })

        return {"image_path": image_path, "detections": detections}

    def predict_batch(self, image_dir: str, output_json: str = "sahi_results.json"):
        """
        Run inference over a directory of images and save results as JSON.
        """
        supported_exts = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}
        images = [
            f for f in os.listdir(image_dir) if os.path.splitext(f)[1] in supported_exts
        ]

        all_results = []
        for i, fname in enumerate(images):
            image_path = os.path.join(image_dir, fname)
            print(f"[SAHI] Processing {i+1}/{len(images)}: {fname}")
            res = self.predict_image(image_path)
            all_results.append(res)

        with open(output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[SAHI] Saved {len(all_results)} results → {output_json}")
        return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAHI Auditor Inference Pipeline")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--image", default=None, help="Single image path")
    parser.add_argument("--image_dir", default=None, help="Directory of images")
    parser.add_argument("--output_json", default="sahi_results.json")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--slice_size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.20)
    args = parser.parse_args()

    pipeline = SAHIAuditorPipeline(
        weights_path=args.weights,
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_ratio=args.overlap,
        confidence_threshold=args.conf,
    )

    if args.image:
        result = pipeline.predict_image(args.image)
        print(json.dumps(result, indent=2))
    elif args.image_dir:
        pipeline.predict_batch(args.image_dir, args.output_json)
    else:
        # Quick smoke test with dummy image
        dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cv2.imwrite("/tmp/test_road.jpg", dummy)
        result = pipeline.predict_image("/tmp/test_road.jpg")
        print(f"Smoke test detections: {len(result['detections'])}")
        print("SAHI pipeline OK ✓")
