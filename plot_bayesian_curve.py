import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from models.ESCNet import ESCNet
from config import load_config
from utils import get_image

def calculate_bayesian_curve(model, config, device, num_samples=30):
    # Prepare data
    train_img_dir = os.path.join(config.train_dir, "Image")
    train_gt_dir = os.path.join(config.train_dir, "GT_Object")
    
    img_list = sorted([f for f in os.listdir(train_img_dir) if f.endswith(".jpg")])[:num_samples]
    
    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    gt_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
    ])

    all_uncertainties = []
    all_errors = []

    model.eval()
    with torch.no_grad():
        for img_name in tqdm(img_list, desc="Analyzing Uncertainty"):
            img_path = os.path.join(train_img_dir, img_name)
            gt_path = os.path.join(train_gt_dir, img_name.replace(".jpg", ".png"))
            
            if not os.path.exists(gt_path):
                continue
                
            img = Image.open(img_path).convert("RGB")
            gt = Image.open(gt_path).convert("L")
            
            img_tensor = transform(img).unsqueeze(0).to(device)
            # Match sizes
            gt_tensor = gt_transform(gt).to(device)
            
            # Forward pass
            _, scaled_preds, log_var = model(img_tensor)
            
            pred = scaled_preds[-1].sigmoid() # (1, 1, H, W)
            # Exp log var to get variance
            uncertainty = torch.exp(log_var) # (1, 1, H, W)
            
            # Interpolate pred to GT size or vice versa
            pred_resized = F.interpolate(pred, size=gt_tensor.shape[1:], mode="bilinear", align_corners=False)
            uncertainty_resized = F.interpolate(uncertainty, size=gt_tensor.shape[1:], mode="bilinear", align_corners=False)
            
            # Calculate pixel-wise error (MAE)
            error = torch.abs(pred_resized - gt_tensor)
            
            all_uncertainties.append(uncertainty_resized.cpu().numpy().flatten())
            all_errors.append(error.cpu().numpy().flatten())

    uncertainties = np.concatenate(all_uncertainties)
    errors = np.concatenate(all_errors)

    # Binning
    num_bins = 15
    bin_edges = np.linspace(uncertainties.min(), uncertainties.max(), num_bins + 1)
    
    bin_centers = []
    bin_errors = []
    bin_stds = []

    for i in range(len(bin_edges) - 1):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i+1])
        if np.sum(mask) > 50:
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
            bin_errors.append(np.mean(errors[mask]))
            bin_stds.append(np.std(errors[mask]) / np.sqrt(np.sum(mask)))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, bin_errors, 'o-', linewidth=2, markersize=8, label='Empirical Error')
    plt.fill_between(bin_centers, np.array(bin_errors) - np.array(bin_stds), np.array(bin_errors) + np.array(bin_stds), alpha=0.2)
    
    plt.title("Bayesian Reliability Curve (Uncertainty vs. Error)")
    plt.xlabel("Predictive Uncertainty ($\sigma^2$)")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    save_path = "bayesian_curve.png"
    plt.savefig(save_path)
    print(f"Bayesian curve saved to {save_path}")
    
    return bin_centers, bin_errors

if __name__ == "__main__":
    config = load_config("config.yaml")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ESCNet(config, pretrained=False).to(device)
    checkpoint_path = "checkpoints/escnet/new_epoch_25_bayesian.pth"
    
    if os.path.exists(checkpoint_path):
        from utils import check_state_dict
        state_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = check_state_dict(state_dict)
        try:
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {checkpoint_path}")
        except Exception as e:
            print(f"Warning during loading: {e}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        exit()

    calculate_bayesian_curve(model, config, device)
