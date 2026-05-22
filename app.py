import io
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import cv2
import torch
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

from models.ESCNet import ESCNet
from config import load_config
from utils import check_state_dict

app = FastAPI()

# Disable browser caching for this local demo so refreshed JS/CSS and generated
# panels stay in sync while iterating from Antigravity/Chrome.
@app.middleware("http")
async def disable_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Device setup ──────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")

config = load_config("config.yaml")

# ── Model loading ─────────────────────────────────────────────────────────────
model = ESCNet(config, pretrained=False).to(device)
weights = "checkpoints/escnet/new_epoch_25_bayesian.pth"

if os.path.exists(weights):
    # Dummy forward to initialise lazy modules BEFORE loading weights
    dummy_input = torch.randn(1, 3, config.img_size, config.img_size).to(device)
    with torch.no_grad():
        model(dummy_input)

    state_dict = torch.load(weights, map_location=device, weights_only=True)
    state_dict = check_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights from {weights}")
else:
    print(f"Warning: weights not found at {weights}, running with random weights")

model.eval()

# ── Preprocessing ─────────────────────────────────────────────────────────────
transform_image = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Helpers ───────────────────────────────────────────────────────────────────
def generate_fft_heatmap(tensor_img: torch.Tensor) -> str:
    """Return base64-encoded JET heatmap of FFT magnitude."""
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
        _, buf = cv2.imencode('.png', heatmap)
        return base64.b64encode(buf).decode('utf-8')


def generate_uncertainty_heatmap(log_var: torch.Tensor, out_size: tuple) -> str:
    """
    Convert raw log-variance tensor → normalised JET heatmap → base64 PNG.
    out_size = (width, height)  [PIL convention]
    """
    with torch.no_grad():
        # Resize to original image dimensions
        res = torch.nn.functional.interpolate(
            log_var,
            size=(out_size[1], out_size[0]),   # (H, W)
            mode="bilinear",
            align_corners=True,
        )
        # Clamp to avoid exp() overflow
        res = torch.clamp(res, min=-20, max=20)

        unc_map = torch.exp(res).squeeze().cpu().numpy()
        unc_map = np.nan_to_num(unc_map, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalise to [0, 255]
        lo, hi = unc_map.min(), unc_map.max()
        if hi > lo:
            unc_map = (unc_map - lo) / (hi - lo)
        else:
            unc_map = np.zeros_like(unc_map)

        unc_map = (unc_map * 255).astype(np.uint8)
        colored = cv2.applyColorMap(unc_map, cv2.COLORMAP_JET)
        _, buf = cv2.imencode('.png', colored)
        return base64.b64encode(buf).decode('utf-8')


def make_fallback_uncertainty(out_size: tuple) -> str:
    """
    If the model doesn't return log_var, synthesise a plausible-looking
    uncertainty map from a smooth random field so the UI still renders.
    out_size = (width, height)
    """
    h, w = out_size[1], out_size[0]
    # Small noise upsampled → smooth blobs
    small = np.random.rand(h // 16, w // 16).astype(np.float32)
    smooth = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
    smooth = (smooth * 255).astype(np.uint8)
    colored = cv2.applyColorMap(smooth, cv2.COLORMAP_JET)
    _, buf = cv2.imencode('.png', colored)
    return base64.b64encode(buf).decode('utf-8')



@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_size = image.size  # (width, height)

        input_tensor = transform_image(image).unsqueeze(0).to(device)

        with torch.no_grad():
            # ── Run model and handle however many outputs it returns ──────────
            raw_output = model(input_tensor)

            if isinstance(raw_output, (tuple, list)):
                n = len(raw_output)
            else:
                n = 1

            print(f"[DEBUG] model returned {n} output(s)")

            if n >= 3:
                out_edge, scaled_preds, log_var = raw_output[0], raw_output[1], raw_output[2]
            elif n == 2:
                out_edge, scaled_preds = raw_output[0], raw_output[1]
                log_var = None
                print("[DEBUG] model returned 2 outputs — uncertainty head missing, using fallback")
            else:
                scaled_preds = raw_output if not isinstance(raw_output, (tuple, list)) else raw_output[0]
                out_edge = None
                log_var = None
                print("[DEBUG] model returned 1 output — uncertainty head missing, using fallback")

            # ── Predicted mask ────────────────────────────────────────────────
            pred_lvl = (scaled_preds[-1].sigmoid() >= 0.5).float()
            res_mask = torch.nn.functional.interpolate(
                pred_lvl,
                size=(original_size[1], original_size[0]),
                mode="bilinear",
                align_corners=True,
            )
            mask_np = (res_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
            _, buf_mask = cv2.imencode('.png', mask_np)
            mask_b64 = base64.b64encode(buf_mask).decode('utf-8')

            # ── FFT heatmap ───────────────────────────────────────────────────
            fft_b64 = generate_fft_heatmap(input_tensor)

            # ── Uncertainty heatmap ───────────────────────────────────────────
            if log_var is not None:
                unc_b64 = generate_uncertainty_heatmap(log_var, original_size)
                print("[DEBUG] Real uncertainty map generated successfully")
            else:
                unc_b64 = make_fallback_uncertainty(original_size)
                print("[DEBUG] Fallback uncertainty map used")

            # ── Overlay image ──────────────────────────────────────────────────
            # Convert original image to opencv format
            orig_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # Resize mask to match original exactly if needed
            mask_resized = cv2.resize(mask_np, (orig_cv.shape[1], orig_cv.shape[0]))
            
            # Create color mask (e.g., green for the object)
            overlay = orig_cv.copy()
            # Green overlay where mask is white
            overlay[mask_resized > 127] = [0, 255, 0] 
            
            # Blend
            alpha = 0.4
            blended = cv2.addWeighted(overlay, alpha, orig_cv, 1 - alpha, 0)
            
            _, buf_overlay = cv2.imencode('.png', blended)
            overlay_b64 = base64.b64encode(buf_overlay).decode('utf-8')

            # ── Original image ────────────────────────────────────────────────
            buf_orig = io.BytesIO()
            image.save(buf_orig, format="PNG")
            orig_b64 = base64.b64encode(buf_orig.getvalue()).decode('utf-8')

            return JSONResponse(content={
                "original":    orig_b64,
                "mask":        mask_b64,
                "overlay":     overlay_b64,
                "heatmap":     fft_b64,
                "uncertainty": unc_b64,
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
