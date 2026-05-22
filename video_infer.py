import argparse
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

from config import load_config
from models.ESCNet import ESCNet
from utils import check_state_dict


def build_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def resolve_checkpoint(config, ckpt_arg: str | None) -> str:
    if ckpt_arg:
        ckpt_path = Path(ckpt_arg)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(__file__).resolve().parent / ckpt_path
        return str(ckpt_path)

    ckpt_dir = Path(config.save_model_dir) / config.name
    ckpt_files = sorted(ckpt_dir.glob("*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return str(ckpt_files[-1])


def load_model(config, checkpoint_path: str, device: torch.device):
    model = ESCNet(config, pretrained=False).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    state_dict = check_state_dict(state_dict)
    
    # Dummy forward pass to initialize lazy modules
    dummy_input = torch.randn(1, 3, config.img_size, config.img_size).to(device)
    with torch.no_grad():
        model(dummy_input)
        
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def predict_mask(model, frame_bgr, transform, device):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        _, scaled_preds, _ = model(input_tensor)
        pred = scaled_preds[-1].sigmoid()

    pred = pred.squeeze().detach().cpu().numpy()
    pred = cv2.resize(pred, (frame_bgr.shape[1], frame_bgr.shape[0]))
    pred = (pred * 255).clip(0, 255).astype("uint8")
    return pred


def write_video(model, input_path, output_path, transform, device, overlay_alpha):
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        mask = predict_mask(model, frame, transform, device)
        mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        result = cv2.addWeighted(frame, 1.0 - overlay_alpha, mask_color, overlay_alpha, 0)
        writer.write(result)
        frame_count += 1

        if frame_count % 25 == 0:
            print(f"Processed {frame_count} frames")

    capture.release()
    writer.release()
    print(f"Saved output video to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ESCNet video inference.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default="outputs/video_prediction.mp4")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--overlay-alpha", type=float, default=0.4)
    args = parser.parse_args()

    config = load_config(args.config)
    device = build_device()
    print(f"Using device: {device}")

    checkpoint_path = resolve_checkpoint(config, args.ckpt)
    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_model(config, checkpoint_path, device)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transform = build_transform(config.img_size)
    write_video(
        model=model,
        input_path=args.input,
        output_path=str(output_path),
        transform=transform,
        device=device,
        overlay_alpha=args.overlay_alpha,
    )


if __name__ == "__main__":
    main()
