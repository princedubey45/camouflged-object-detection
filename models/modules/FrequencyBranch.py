import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FrequencyBranch(nn.Module):
    def __init__(self, out_channels: int, mode: str = "fft"):
        super().__init__()
        self.mode = mode.lower()
        if self.mode != "fft":
            raise ValueError(f"Unsupported frequency mode: {mode}")

        self.stem = ConvBNReLU(1, 32)
        self.stage1 = ConvBNReLU(32, 64, stride=2)
        self.stage2 = ConvBNReLU(64, 96, stride=2)
        self.stage3 = ConvBNReLU(96, 128, stride=2)
        self.stage4 = ConvBNReLU(128, 128, stride=2)

        self.proj1 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.proj2 = nn.Conv2d(96, out_channels, kernel_size=1)
        self.proj3 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.proj4 = nn.Conv2d(128, out_channels, kernel_size=1)

    def _fft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        fft_map = torch.fft.fft2(gray)
        fft_map = torch.fft.fftshift(fft_map, dim=(-2, -1))
        magnitude = torch.log1p(torch.abs(fft_map))

        min_val = magnitude.amin(dim=(-2, -1), keepdim=True)
        max_val = magnitude.amax(dim=(-2, -1), keepdim=True)
        magnitude = (magnitude - min_val) / (max_val - min_val + 1e-6)
        return magnitude

    def forward(self, x: torch.Tensor, target_sizes):
        freq = self._fft_magnitude(x)

        x0 = self.stem(freq)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        f1 = F.interpolate(self.proj1(x1), size=target_sizes[0], mode="bilinear", align_corners=False)
        f2 = F.interpolate(self.proj2(x2), size=target_sizes[1], mode="bilinear", align_corners=False)
        f3 = F.interpolate(self.proj3(x3), size=target_sizes[2], mode="bilinear", align_corners=False)
        f4 = F.interpolate(self.proj4(x4), size=target_sizes[3], mode="bilinear", align_corners=False)

        return f1, f2, f3, f4
