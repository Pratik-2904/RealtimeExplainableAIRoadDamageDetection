import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class SPDConv(nn.Module):
    """
    Space-to-Depth Convolution for YOLO v11.
    Replaces strided convolutions to preserve fine-grained feature details for small objects.
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.scale = 2
        # SPD layer scales channel dimension by scale^2
        c_spd = c1 * (self.scale ** 2)
        # We always use stride=1 in the actual conv to avoid information loss
        self.conv = Conv(c_spd, c2, k, s=1, p=p, g=g, d=d, act=act)

    def forward(self, x):
        B, C, H, W = x.shape
        s = self.scale
        # Pad if necessary so dimensions are divisible by the scale (2)
        if H % s != 0 or W % s != 0:
            pad_h = (s - H % s) % s
            pad_w = (s - W % s) % s
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
            
        sub_tensors = []
        for i in range(s):
            for j in range(s):
                sub_tensors.append(x[:, :, i::s, j::s])
                
        x_spd = torch.cat(sub_tensors, dim=1)
        return self.conv(x_spd)
