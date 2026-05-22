import torch
from config import load_config
from models.ESCNet import ESCNet
from utils import check_state_dict

config = load_config("config.yaml")
model = ESCNet(config, pretrained=False)
dummy = torch.randn(1, 3, config.img_size, config.img_size)
model(dummy)

# Get list of uncertainty head layers
unc_layers = [name for name, _ in model.named_parameters() if "uncertainty_head" in name]
print(f"Uncertainty head layers: {unc_layers}")

import glob
# Load state dict
for weights_path in glob.glob("**/*.pth", recursive=True):
    print(f"\nChecking {weights_path}:")
    try:
        state_dict = torch.load(weights_path, map_location="cpu")
        state_dict = check_state_dict(state_dict)

        # Check if uncertainty head layers are in state_dict
        # Try different naming conventions
        found_layers = []
        for layer in unc_layers:
            if any(layer in key for key in state_dict.keys()):
                found_layers.append(layer)
                
        print(f"Found {len(found_layers)} out of {len(unc_layers)} uncertainty head layers.")
        
        if len(found_layers) > 0:
            print(f"SUCCESS: Uncertainty head weights are present in {weights_path}!")
            if len(found_layers) < len(unc_layers):
                print(f"Missing: {[l for l in unc_layers if l not in found_layers]}")
    except Exception as e:
        print(f"Error loading {weights_path}: {e}")
