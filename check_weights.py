import torch
from config import load_config
from models.ESCNet import ESCNet
import argparse

config = load_config("config.yaml")
model = ESCNet(config, pretrained=False)
dummy = torch.randn(1, 3, config.img_size, config.img_size)
model(dummy)

# Get weights before
w_before = model.decoder.ipt_blk5.conv1.weight.clone()

# Load state dict
state_dict = torch.load("checkpoints/escnet/epoch_30.pth", map_location="cpu")
from utils import check_state_dict
state_dict = check_state_dict(state_dict)

res = model.load_state_dict(state_dict, strict=True)
print("load res:", res)

w_after = model.decoder.ipt_blk5.conv1.weight.clone()

print("Weights equal:", torch.equal(w_before, w_after))
