from models.modules.Decoder import Decoder
from models.modules.AETP import AETP
from models.modules.Encoder import Encoder
from models.modules.FrequencyBranch import FrequencyBranch
import torch.nn as nn


# @torch.compile()
class ESCNet(nn.Module):
    def __init__(self, config, pretrained=True):
        super(ESCNet, self).__init__()
        self.channels = config.lateral_channels
        inter_channel = 128

        self.encoder = Encoder(config, pretrained)
        self.decoder = Decoder(config,inter_channel)
        self.enhanced = AETP(inter_channel)
        self.use_frequency_branch = config.use_frequency_branch
        self.frequency_branch = (
            FrequencyBranch(inter_channel, mode=config.frequency_mode)
            if self.use_frequency_branch
            else None
        )

        self.asa4 = nn.Sequential(
            nn.Conv2d(self.channels[0], inter_channel, 1, 1, 0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
        )
        self.asa3 = nn.Sequential(
            nn.Conv2d(self.channels[1], inter_channel, 1, 1, 0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
        )
        self.asa2 = nn.Sequential(
            nn.Conv2d(self.channels[2], inter_channel, 1, 1, 0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
        )
        self.asa1 = nn.Sequential(
            nn.Conv2d(self.channels[3], inter_channel, 1, 1, 0),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        ########## Encoder ##########
        (x1, x2, x3, x4) = self.encoder(x)  # x1,x2,x3,x1+x2+x3+x4
        x4 = self.asa4(x4)
        x3 = self.asa3(x3)
        x2 = self.asa2(x2)
        x1 = self.asa1(x1)

        if self.frequency_branch is not None:
            f1, f2, f3, f4 = self.frequency_branch(
                x, [x1.shape[2:], x2.shape[2:], x3.shape[2:], x4.shape[2:]]
            )
            x1 = x1 + f1
            x2 = x2 + f2
            x3 = x3 + f3
            x4 = x4 + f4

        features = [x, x1, x2, x3, x4]

        ########## Decoder ##########
        out_edge = self.enhanced(features)  # logits

        out_mask, log_var = self.decoder(features, out_edge.sigmoid())
        return out_edge, out_mask, log_var
