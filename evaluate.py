"""在 test 集上定量评估模型，并输出可视化对比图。

用法:
    python evaluate.py --weights runs/unet/best.pt --split test --out-dir runs/unet/eval
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch

from dataset import RemoteSensingDataset
from unet import UNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UNet 建筑物提取评估")
    p.add_argument("--data-root", default="data")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--weights", default="runs/unet/best.pt")
    p.add_argument("--out-dir", default="runs/unet/eval")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--vis", type=int, default=8, help="导出多少张可视化对比图")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_model(weights: str, device: torch.device) -> UNet:
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(state)
    model.to(device).to(memory_format=torch.channels_last).eval()
    return model


@torch.no_grad()
def predict_probs(model: UNet, images: torch.Tensor, device: torch.device) -> np.ndarray:
    images = images.to(device, memory_format=torch.channels_last)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(images)
    return torch.sigmoid(logits.float()).cpu().numpy()[:, 0]


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[警告] 未检测到 CUDA，评估将退回 CPU。")
        args.device = "cpu"
    device = torch.device(args.device)

    model = load_model(args.weights, device)
    ds = RemoteSensingDataset(
        os.path.join(args.data_root, f"{args.split}_images"),
        os.path.join(args.data_root, f"{args.split}_masks"),
        image_size=args.image_size,
        train=False,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda",
    )
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"评估 {args.split} 集: {len(ds)} 张 | 权重 {args.weights} | 设备 {device}")

    probs_all, masks_all = [], []
    for images, masks in loader:
        probs_all.append(predict_probs(model, images, device))
        masks_all.append(masks.numpy()[:, 0] > 0.5)
    probs = np.concatenate(probs_all)          # (N,H,W)
    gts = np.concatenate(masks_all)            # (N,H,W)

    # ---------- 阈值扫描 ----------
    sweep = []
    for thr in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        tp = float(((probs > thr) & gts).sum())
        fp = float(((probs > thr) & ~gts).sum())
        fn = float(((probs <= thr) & gts).sum())
        tn = float(((probs <= thr) & ~gts).sum())
        eps = 1e-7
        sweep.append({
            "threshold": thr,
            "iou": tp / (tp + fp + fn + eps),
            "dice": 2 * tp / (2 * tp + fp + fn + eps),
            "precision": tp / (tp + fp + eps),
            "recall": tp / (tp + fn + eps),
            "pixel_acc": (tp + tn) / (tp + tn + fp + fn + eps),
        })

    best = max(sweep, key=lambda r: r["dice"])
    chosen = next((r for r in sweep if abs(r["threshold"] - args.threshold) < 1e-9), best)

    # ---------- 逐图指标 ----------
    rows = []
    eps = 1e-7
    for i in range(len(ds)):
        pred = probs[i] > args.threshold
        gt = gts[i]
        tp = float((pred & gt).sum())
        fp = float((pred & ~gt).sum())
        fn = float((~pred & gt).sum())
        rows.append({
            "file": ds.filenames[i],
            "iou": tp / (tp + fp + fn + eps),
            "dice": 2 * tp / (2 * tp + fp + fn + eps),
            "precision": tp / (tp + fp + eps),
            "recall": tp / (tp + fn + eps),
            "gt_ratio": float(gt.mean()),
            "pred_ratio": float(pred.mean()),
        })

    with open(os.path.join(args.out_dir, "per_image.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ious = np.array([r["iou"] for r in rows])
    summary = {
        "split": args.split,
        "num_images": len(ds),
        "threshold": args.threshold,
        "metrics_at_threshold": chosen,
        "best_threshold_by_dice": best,
        "threshold_sweep": sweep,
        "per_image_iou": {
            "mean": float(ious.mean()),
            "median": float(np.median(ious)),
            "min": float(ious.min()),
            "max": float(ious.max()),
            "std": float(ious.std()),
            "images_over_0.5": int((ious > 0.5).sum()),
        },
        "weights": os.path.abspath(args.weights),
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("-" * 70)
    print(f"IoU={chosen['iou']:.4f} | Dice/F1={chosen['dice']:.4f} | "
          f"Precision={chosen['precision']:.4f} | Recall={chosen['recall']:.4f} | "
          f"PixelAcc={chosen['pixel_acc']:.4f}")
    print(f"逐图 IoU: mean={ious.mean():.4f} median={np.median(ious):.4f} "
          f"min={ious.min():.4f} max={ious.max():.4f}")
    print(f"最佳阈值(按 Dice): {best['threshold']} -> Dice={best['dice']:.4f}, IoU={best['iou']:.4f}")
    print(f"结果已写入 {os.path.abspath(args.out_dir)}")

    # ---------- 可视化 ----------
    if args.vis > 0:
        n = min(args.vis, len(ds))
        idxs = np.linspace(0, len(ds) - 1, n).astype(int)
        tiles = []
        for i in idxs:
            img = (cv2.imread(os.path.join(ds.images_dir, ds.filenames[i]))[:, :, ::-1]).copy()
            img = cv2.resize(img, (args.image_size, args.image_size))
            gt = (gts[i].astype(np.uint8) * 255)
            pr = ((probs[i] > args.threshold).astype(np.uint8) * 255)
            # 覆盖层：绿=GT，红=预测
            overlay = img.copy()
            overlay[gt > 0] = (0.4 * overlay[gt > 0] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
            overlay[pr > 0] = (0.4 * overlay[pr > 0] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
            tiles.append(np.concatenate([img, cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB),
                                         cv2.cvtColor(pr, cv2.COLOR_GRAY2RGB), overlay], axis=1))
        grid = np.concatenate(tiles, axis=0)
        vis_path = os.path.join(args.out_dir, "visual_grid.png")
        cv2.imwrite(vis_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"可视化对比图: {os.path.abspath(vis_path)}（每行: 原图 | 真值 | 预测 | 叠加）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
