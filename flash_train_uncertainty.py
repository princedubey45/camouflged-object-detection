import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.ESCNet import ESCNet
from config import load_config
from dataset import MyData
from loss import UncertaintyStructureLoss, StructureLoss
from utils import check_state_dict

def flash_train():
    config = load_config("config.yaml")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = ESCNet(config, pretrained=False).to(device)
    weights_path = "checkpoints/escnet/new_epoch_25.pth"
    
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        state_dict = check_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded base weights from {weights_path}")
    else:
        print(f"Error: {weights_path} not found.")
        return

    # Freeze all except uncertainty head
    for name, param in model.named_parameters():
        if "uncertainty_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Dataloader - just 100 samples for flash training
    dataset = MyData(config, dataset_dir=config.train_dir, image_size=config.img_size, is_train=True)
    # Subset to 100 samples
    indices = torch.randperm(len(dataset))[:100]
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=4, shuffle=True)

    # Losses
    uncertainty_loss_fn = UncertaintyStructureLoss()
    
    # Optimizer - only for uncertainty head
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    model.train()
    print("Starting Flash Training of Bayesian head (approx. 2-3 mins)...")
    
    num_epochs = 2
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for inputs, gts, edges in pbar:
            inputs, gts = inputs.to(device), gts.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            _, scaled_preds, log_var = model(inputs)
            
            # Last level prediction for uncertainty
            pred = scaled_preds[-1]
            pred_resized = nn.functional.interpolate(pred, size=gts.shape[2:], mode="bilinear", align_corners=False)
            
            # Calculate uncertainty loss
            loss = uncertainty_loss_fn(pred_resized, gts, log_var)
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Save fixed weights
    save_path = "checkpoints/escnet/new_epoch_25_bayesian.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nFlash training complete! Saved Bayesian weights to {save_path}")
    print("You can now use this checkpoint for both accurate masks and Bayesian heatmaps.")

if __name__ == "__main__":
    flash_train()
