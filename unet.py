"""面向遥感影像语义分割的 UNet 网络架构。

相比原始版本的改进：
1. 前向自动把输入补齐到 16 的倍数，输出再裁回原尺寸 —— 真正支持任意尺寸输入，
   不再因为 H/W 不能被 16 整除而在 torch.cat 处崩溃；
2. 可选 base_width 控制模型容量；
3. 提供 model_summary / count_parameters 便于快速核对规模。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 4 次 MaxPool2d(2)，因此输入边长必须是 2**4 的倍数
DOWNSAMPLE_FACTOR = 16


class DoubleConv(nn.Module):
    """(卷积 -> 批归一化 -> ReLU) * 2 基础卷积块"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    """面向遥感影像语义分割的完整 UNet 网络架构"""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_width: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        w = base_width

        self.inc = DoubleConv(in_channels, w)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(w, w * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(w * 2, w * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(w * 4, w * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(w * 8, w * 16))

        self.up1 = nn.ConvTranspose2d(w * 16, w * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(w * 16, w * 8)

        self.up2 = nn.ConvTranspose2d(w * 8, w * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(w * 8, w * 4)

        self.up3 = nn.ConvTranspose2d(w * 4, w * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(w * 4, w * 2)

        self.up4 = nn.ConvTranspose2d(w * 2, w, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(w * 2, w)

        self.outc = nn.Conv2d(w, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        # 补齐到 16 的倍数，保证上采样与跳跃连接形状一致
        pad_h = (-h) % DOWNSAMPLE_FACTOR
        pad_w = (-w) % DOWNSAMPLE_FACTOR
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)

        logits = self.outc(x)
        return logits[..., :h, :w]

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"参数量: {total / 1e6:.2f} M (可训练 {trainable / 1e6:.2f} M)"
