import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import numpy as np
import cv2
import base64
from PIL import Image

def generate_random_unc():
    log_var = torch.randn(1, 1, 352, 352) * 5 # Simulate raw conv output, high variance
    original_size = (800, 600)
    res_unc = torch.nn.functional.interpolate(
        log_var,
        size=(original_size[1], original_size[0]),
        mode="bilinear",
        align_corners=True,
    )
    res_unc = torch.clamp(res_unc, min=-20, max=20)
    unc_map = torch.exp(res_unc).squeeze().cpu().numpy()
    unc_map = np.nan_to_num(unc_map, nan=0.0, posinf=0.0, neginf=0.0)

    min_v, max_v = unc_map.min(), unc_map.max()
    print(f"min_v: {min_v}, max_v: {max_v}")
    if max_v > min_v:
        unc_map = (unc_map - min_v) / (max_v - min_v)
    else:
        unc_map = np.zeros_like(unc_map)

    unc_map = (unc_map * 255).astype(np.uint8)
    print("max uint8 val:", unc_map.max())
    unc_map_color = cv2.applyColorMap(unc_map, cv2.COLORMAP_JET)
    _, buffer_unc = cv2.imencode('.png', unc_map_color)
    unc_b64 = base64.b64encode(buffer_unc).decode('utf-8')
    print("Base64 string length:", len(unc_b64))

generate_random_unc()
