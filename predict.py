"""批量推理：输入任意尺寸遥感影像，输出二值建筑物掩膜（白色=建筑物）。

相比原始版本的改进：
1. 支持批量推理（一次 forward 处理多张），吞吐显著提升；
2. 输入尺寸自动 resize 到训练尺寸，网络内部再补齐到 16 的倍数，任意尺寸都不会崩；
3. 可同时输出叠加可视化图，便于人工核验；
4. 权重缺失、目录为空等情况给出明确提示。
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch

from dataset import IMG_EXTS, _read_image
from unet import UNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UNet 建筑物提取批量推理")
    p.add_argument("--input", default="data/test_images", help="输入影像目录或单张影像")
    p.add_argument("--weights", default="runs/unet/best.pt")
    p.add_argument("--output", default="preds/test", help="掩膜输出目录")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--overlay", action="store_true", help="同时输出叠加可视化图")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_model(weights: str, device: torch.device) -> UNet:
    if not os.path.isfile(weights):
        raise SystemExit(f"[错误] 权重文件不存在: {os.path.abspath(weights)}，请先运行 train.py")
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(state)
    return model.to(device).to(memory_format=torch.channels_last).eval()


@torch.no_grad()
def infer_batch(model: UNet, batch: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div_(255.0)
    tensor = tensor.to(device, memory_format=torch.channels_last)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(tensor)
    return torch.sigmoid(logits.float()).cpu().numpy()[:, 0]


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[警告] 未检测到 CUDA，推理将退回 CPU。")
        args.device = "cpu"
    device = torch.device(args.device)

    if os.path.isdir(args.input):
        files = sorted(f for f in os.listdir(args.input) if f.lower().endswith(IMG_EXTS))
        files = [os.path.join(args.input, f) for f in files]
    else:
        files = [args.input]
    if not files:
        raise SystemExit(f"[错误] 输入目录中没有影像: {os.path.abspath(args.input)}")

    model = load_model(args.weights, device)
    os.makedirs(args.output, exist_ok=True)
    print(f"设备 {device} | 权重 {args.weights} | 待处理 {len(files)} 张 | 输出 {os.path.abspath(args.output)}")

    done = 0
    for start in range(0, len(files), args.batch_size):
        chunk = files[start:start + args.batch_size]
        imgs, originals = [], []
        for path in chunk:
            rgb = _read_image(path)
            originals.append(rgb)
            imgs.append(cv2.resize(rgb, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR))
        probs = infer_batch(model, np.stack(imgs), device)

        for path, prob, rgb in zip(chunk, probs, originals):
            mask = (prob > args.threshold).astype(np.uint8) * 255
            if mask.shape != rgb.shape[:2]:
                mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            stem = os.path.splitext(os.path.basename(path))[0]
            cv2.imwrite(os.path.join(args.output, f"{stem}.png"), mask)
            if args.overlay:
                over = rgb.copy()
                over[mask > 0] = (0.45 * over[mask > 0] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
                cv2.imwrite(os.path.join(args.output, f"{stem}_overlay.png"), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
            done += 1

    print(f"[完成] 共导出 {done} 张掩膜到 {os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
