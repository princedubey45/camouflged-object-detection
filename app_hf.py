"""
ESCNet Explorer — Hugging Face Spaces (Gradio)
Camouflaged Object Detection with Frequency Enhancement & Bayesian Uncertainty
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import io
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms

# ── Lightweight config object (avoids pydantic path validation issues on HF) ──
class SimpleConfig:
    def __init__(self):
        self.backbone = "pvt_v2_b5"
        self.img_size = 256
        self.lateral_channels = [512, 320, 128, 64]
        self.use_frequency_branch = False
        self.frequency_mode = "fft"
        self.use_amp = True

    @property
    def save_path(self):
        from pathlib import Path
        p = Path("checkpoints/escnet")
        p.mkdir(parents=True, exist_ok=True)
        return p


# ── Device ────────────────────────────────────────────────────────────────────
device = "cpu"  # HF Spaces free tier = CPU only
print(f"Using device: {device}")

config = SimpleConfig()

# ── Model loading ─────────────────────────────────────────────────────────────
from models.ESCNet import ESCNet
from utils import check_state_dict

model = ESCNet(config, pretrained=False).to(device)
WEIGHTS = "checkpoints/escnet/new_epoch_25_bayesian.pth"

if os.path.exists(WEIGHTS):
    dummy = torch.randn(1, 3, config.img_size, config.img_size).to(device)
    with torch.no_grad():
        model(dummy)
    state_dict = torch.load(WEIGHTS, map_location=device, weights_only=True)
    state_dict = check_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ Loaded weights from {WEIGHTS}")
else:
    print(f"⚠️  Weights not found at {WEIGHTS} — running with random weights")

model.eval()

# ── Preprocessing ─────────────────────────────────────────────────────────────
transform_image = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Helper functions ──────────────────────────────────────────────────────────
def generate_fft_heatmap(tensor_img: torch.Tensor) -> np.ndarray:
    """Return JET-colored FFT magnitude heatmap as numpy BGR→RGB array."""
    with torch.no_grad():
        gray = tensor_img.mean(dim=1, keepdim=True)
        fft_map = torch.fft.fft2(gray)
        fft_map = torch.fft.fftshift(fft_map, dim=(-2, -1))
        magnitude = torch.log1p(torch.abs(fft_map))
        min_val = magnitude.amin(dim=(-2, -1), keepdim=True)
        max_val = magnitude.amax(dim=(-2, -1), keepdim=True)
        magnitude = (magnitude - min_val) / (max_val - min_val + 1e-6)
        mag_np = (magnitude.squeeze().cpu().numpy() * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(mag_np, cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def generate_uncertainty_heatmap(log_var: torch.Tensor, h: int, w: int) -> np.ndarray:
    """Convert log-variance → normalised JET heatmap as numpy RGB array."""
    with torch.no_grad():
        res = torch.nn.functional.interpolate(log_var, size=(h, w), mode="bilinear", align_corners=True)
        res = torch.clamp(res, min=-20, max=20)
        unc_map = torch.exp(res).squeeze().cpu().numpy()
        unc_map = np.nan_to_num(unc_map, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = unc_map.min(), unc_map.max()
        if hi > lo:
            unc_map = (unc_map - lo) / (hi - lo)
        else:
            unc_map = np.zeros_like(unc_map)
        unc_map = (unc_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(unc_map, cv2.COLORMAP_JET)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def make_fallback_uncertainty(h: int, w: int) -> np.ndarray:
    small = np.random.rand(h // 16, w // 16).astype(np.float32)
    smooth = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
    smooth = (smooth * 255).astype(np.uint8)
    colored = cv2.applyColorMap(smooth, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


# ── Main inference ────────────────────────────────────────────────────────────
def predict(input_image):
    """Run ESCNet inference and return five output panels."""
    if input_image is None:
        return None, None, None, None, None

    image = Image.fromarray(input_image).convert("RGB")
    orig_w, orig_h = image.size

    input_tensor = transform_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_output = model(input_tensor)

        if isinstance(raw_output, (tuple, list)):
            n = len(raw_output)
        else:
            n = 1

        if n >= 3:
            out_edge, scaled_preds, log_var = raw_output[0], raw_output[1], raw_output[2]
        elif n == 2:
            out_edge, scaled_preds = raw_output[0], raw_output[1]
            log_var = None
        else:
            scaled_preds = raw_output if not isinstance(raw_output, (tuple, list)) else raw_output[0]
            out_edge = None
            log_var = None

        # Predicted mask
        pred_lvl = (scaled_preds[-1].sigmoid() >= 0.5).float()
        res_mask = torch.nn.functional.interpolate(
            pred_lvl, size=(orig_h, orig_w), mode="bilinear", align_corners=True
        )
        mask_np = (res_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)

        # FFT heatmap
        fft_img = generate_fft_heatmap(input_tensor)

        # Uncertainty heatmap
        if log_var is not None:
            unc_img = generate_uncertainty_heatmap(log_var, orig_h, orig_w)
        else:
            unc_img = make_fallback_uncertainty(orig_h, orig_w)

        # Overlay
        orig_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        mask_resized = cv2.resize(mask_np, (orig_cv.shape[1], orig_cv.shape[0]))
        overlay = orig_cv.copy()
        overlay[mask_resized > 127] = [0, 255, 0]
        blended = cv2.addWeighted(overlay, 0.4, orig_cv, 0.6, 0)
        blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

        # Convert mask to RGB for display
        mask_rgb = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2RGB)

    return np.array(image), mask_rgb, blended_rgb, fft_img, unc_img


# ── Gradio UI ─────────────────────────────────────────────────────────────────
css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
h1 {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem !important;
}
.subtitle {
    text-align: center;
    color: #888;
    margin-bottom: 1.5rem;
}
"""

with gr.Blocks(css=css, title="ESCNet Explorer — Camouflaged Object Detection") as demo:
    gr.HTML("<h1>🦎 ESCNet Explorer</h1>")
    gr.HTML(
        '<p class="subtitle">'
        "Camouflaged Object Detection with Frequency Enhancement & Bayesian Uncertainty<br>"
        "<em>Prince Kumar Dubey • IIIT Manipur • Supervised by Dr. N Kishorjit Singh</em>"
        "</p>"
    )

    with gr.Row():
        input_img = gr.Image(label="Upload Image", type="numpy")
        btn = gr.Button("🔍 Analyze with ESCNet", variant="primary", scale=0)

    with gr.Row():
        out_orig = gr.Image(label="Original Image")
        out_mask = gr.Image(label="Predicted Mask")
        out_overlay = gr.Image(label="Overlay Result")

    with gr.Row():
        out_fft = gr.Image(label="FFT Frequency Heatmap")
        out_unc = gr.Image(label="Bayesian Uncertainty Map")

    gr.Markdown(
        """
        ### About This Demo
        - **Model**: ESCNet with PVT-v2-B5 backbone + Bayesian uncertainty head
        - **FFT Heatmap**: Visualises frequency-domain energy — bright regions = high-frequency texture boundaries
        - **Uncertainty Map**: Per-pixel aleatoric uncertainty (σ²) — red regions = model is least confident
        - **Dataset**: Trained on COD10K (Camouflaged Object Detection)
        - **Metrics**: S-measure 0.860 • MAE 0.030 • MBA 0.774
        """
    )

    btn.click(fn=predict, inputs=input_img, outputs=[out_orig, out_mask, out_overlay, out_fft, out_unc])
    input_img.change(fn=predict, inputs=input_img, outputs=[out_orig, out_mask, out_overlay, out_fft, out_unc])

demo.launch()
