import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class BMS_SPPF(nn.Module):
    """
    Bidirectional Multi-Scale Spatial Pyramid Pooling Fast (BMS-SPPF).

    Motivation
    ----------
    Standard SPPF uses a single kernel size (k=5) and misses:
      - Elongated crack patterns (need wider receptive fields).
      - Deep pothole centroids (need sharp, localised pooling).

    Architecture
    ------------
    1. cv1  1×1 conv  c1 → c_ (compressed).
    2. Three cascaded max-pool branches  at k=5, 9, 13 (standard SPPF).
    3. Bidirectional branch: avg-pool → max-pool (crack gradient smoothing).
    4. All branches concatenated (5 × c_) → cv2 1×1 to c2.

    Signature intentionally mirrors SPPF(c1, c2, k=5) so parse_model works.
    """

    def __init__(self, c1: int, c2: int, k: int = 5) -> None:
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 5, c2, 1, 1)   # 5 feature maps concatenated
        self.m   = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.avg = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.m2  = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        y_bidi = self.m2(self.avg(x))        # crack-aware bidirectional branch
        return self.cv2(torch.cat([x, y1, y2, y3, y_bidi], dim=1))


if __name__ == "__main__":
    layer = BMS_SPPF(1024, 1024, k=5)
    dummy = torch.randn(1, 1024, 20, 20)
    out   = layer(dummy)
    print(f"BMS-SPPF  {tuple(dummy.shape)}  →  {tuple(out.shape)}")
    assert out.shape == dummy.shape, "BMS-SPPF shape mismatch!"
    print("BMS-SPPF shape check ✓")

