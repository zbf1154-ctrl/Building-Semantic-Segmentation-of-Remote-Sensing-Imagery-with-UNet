"""UNet 遥感建筑物提取训练脚本（GPU 版）。

相比原始版本的改进：
1. 强制使用 GPU 训练（CUDA 不可用时直接报错退出，而非静默退回 CPU）；
2. 混合精度 AMP（优先 bf16，不支持时退 fp16+GradScaler）+ channels_last；
3. AdamW + 余弦退火 + warmup，替代固定的 Adam(1e-4)；
4. 以验证集 IoU 保存 best.pt 并支持早停，不再只保存固定轮次；
5. 指标补齐 Dice/F1 与像素准确率，逐轮写入 history.csv 与日志文件；
6. 全流程随机种子固定，结果可复现；
7. 每个 epoch 打印耗时与剩余时间（ETA），并做梯度裁剪。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import RemoteSensingDataset
from unet import UNet, model_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UNet 遥感建筑物提取训练 (GPU)")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default="runs/unet")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.6, help="BCE 权重，1-alpha 为 Dice 权重")
    p.add_argument("--patience", type=int, default=15, help="验证 IoU 无提升的早停轮数")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--cache", action="store_true",
                   help="把解码后的影像常驻内存（单进程加载时约 550MB，省掉每轮 TIFF 解码）")
    p.add_argument("--limit-train", type=int, default=0, help="调试用：只用前 N 张训练图")
    p.add_argument("--limit-val", type=int, default=0, help="调试用：只用前 N 张验证图")
    return p.parse_args()


def _worker_init(_worker_id: int) -> None:
    """DataLoader 子进程内限制 cv2 线程数，避免多进程互相争抢。"""
    import cv2

    cv2.setNumThreads(1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class ComboLoss(nn.Module):
    """UNet + 混合损失函数 (BCE + Dice)，输入为 logits。"""

    def __init__(self, alpha: float = 0.6, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        union = pred_sigmoid.sum() + target.sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss


class ConfusionMeter:
    """像素级混淆矩阵累计器。"""

    def __init__(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0.0

    def update(self, logits: torch.Tensor, target: torch.Tensor, threshold: float) -> None:
        pred = torch.sigmoid(logits) > threshold
        tgt = target > 0.5
        self.tp += (pred & tgt).sum().item()
        self.fp += (pred & ~tgt).sum().item()
        self.fn += (~pred & tgt).sum().item()
        self.tn += (~pred & ~tgt).sum().item()

    def compute(self) -> dict:
        eps = 1e-7
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        return {
            "iou": iou,
            "dice": dice,
            "f1": dice,
            "precision": precision,
            "recall": recall,
            "pixel_acc": (tp + tn) / (tp + tn + fp + fn + eps),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }


def build_loaders(args: argparse.Namespace):
    train_ds = RemoteSensingDataset(
        os.path.join(args.data_root, "train_images"),
        os.path.join(args.data_root, "train_masks"),
        image_size=args.image_size,
        train=True,
        cache=args.cache,
    )
    val_ds = RemoteSensingDataset(
        os.path.join(args.data_root, "val_images"),
        os.path.join(args.data_root, "val_masks"),
        image_size=args.image_size,
        train=False,
        cache=args.cache,
    )
    if args.limit_train:
        train_ds = Subset(train_ds, range(min(args.limit_train, len(train_ds))))
    if args.limit_val:
        val_ds = Subset(val_ds, range(min(args.limit_val, len(val_ds))))

    common = dict(
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
        worker_init_fn=_worker_init if args.workers > 0 else None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=len(train_ds) > args.batch_size, **common,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)
    return train_loader, val_loader, len(train_ds), len(val_ds)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device, threshold: float, amp_dtype, amp: bool) -> dict:
    model.eval()
    meter = ConfusionMeter()
    total_loss, n_samples = 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp):
            logits = model(images)
            loss = criterion(logits, masks)
        bs = images.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
        meter.update(logits.float(), masks, threshold)
    metrics = meter.compute()
    metrics["loss"] = total_loss / max(n_samples, 1)
    return metrics


def main() -> int:
    args = parse_args()

    # ---------------- 设备检查：必须 GPU ----------------
    if not torch.cuda.is_available():
        print("[错误] 当前环境检测不到 CUDA GPU，本脚本按要求必须使用 GPU 训练。", file=sys.stderr)
        print(f"       当前 torch={torch.__version__}，请检查显卡驱动与 PyTorch CUDA 版本是否匹配。",
              file=sys.stderr)
        return 2
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    total_vram = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train.log")
    log_fp = open(log_path, "w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_fp.write(msg + "\n")
        log_fp.flush()

    amp = not args.no_amp
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    log("=" * 78)
    log(f"设备: {gpu_name} | 显存 {total_vram:.1f} GB | torch {torch.__version__} | cuda {torch.version.cuda}")
    log(f"混合精度: {'开启' if amp else '关闭'} ({str(amp_dtype).split('.')[-1]}) | channels_last: 开启")
    log(f"配置: {json.dumps(vars(args), ensure_ascii=False)}")
    log("=" * 78)

    train_loader, val_loader, n_train, n_val = build_loaders(args)
    log(f"训练样本 {n_train} | 验证样本 {n_val} | 每轮迭代 {len(train_loader)} 步")

    model = UNet(in_channels=3, out_channels=1).to(device).to(memory_format=torch.channels_last)
    log(model_summary(model))

    criterion = ComboLoss(alpha=args.alpha).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup = max(1, args.warmup_epochs)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return max(1e-3, (epoch + 1) / warmup)
        progress = (epoch - warmup) / max(1, args.epochs - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return max(args.min_lr / args.lr, cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history: list[dict] = []
    best_iou, best_epoch, stale = -1.0, 0, 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss, n_seen = 0.0, 0

        for step, (images, masks) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp):
                logits = model(images)
                loss = criterion(logits, masks)

            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            bs = images.size(0)
            running_loss += loss.item() * bs
            n_seen += bs
            if step % 20 == 0 or step == len(train_loader):
                log(f"  epoch {epoch:02d} [{step:3d}/{len(train_loader)}] loss={running_loss / n_seen:.4f}")

        train_loss = running_loss / max(n_seen, 1)
        val_metrics = evaluate(model, val_loader, criterion, device, args.threshold, amp_dtype, amp)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        eta = (time.time() - start) / epoch * (args.epochs - epoch)
        vram = torch.cuda.max_memory_allocated() / 1024 ** 3
        log(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | IoU={val_metrics['iou']:.4f} | "
            f"Dice={val_metrics['dice']:.4f} | P={val_metrics['precision']:.4f} | "
            f"R={val_metrics['recall']:.4f} | Acc={val_metrics['pixel_acc']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {epoch_time:.1f}s | "
            f"peakVRAM={vram:.2f}GB | ETA {eta / 60:.1f}min")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": round(epoch_time, 2),
            "peak_vram_gb": round(vram, 3),
        })

        if val_metrics["iou"] > best_iou:
            best_iou, best_epoch, stale = val_metrics["iou"], epoch, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": val_metrics,
                    "args": vars(args),
                    "in_channels": 3,
                    "out_channels": 1,
                },
                os.path.join(args.out_dir, "best.pt"),
            )
            log(f"  --> 新最佳: epoch {epoch}, val IoU={best_iou:.4f}，已保存 best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                log(f"  --> 验证 IoU 连续 {args.patience} 轮未提升，提前停止")
                break

    total_minutes = (time.time() - start) / 60
    log("=" * 78)
    log(f"训练结束 | 最佳 epoch={best_epoch} | 最佳 val IoU={best_iou:.4f} | 总耗时 {total_minutes:.1f} min")

    with open(os.path.join(args.out_dir, "history.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_iou": best_iou,
                "best_val_metrics": max(history, key=lambda r: r["val_iou"]),
                "epochs_run": len(history),
                "total_minutes": round(total_minutes, 2),
                "gpu": gpu_name,
                "args": vars(args),
            },
            f, ensure_ascii=False, indent=2,
        )
    log_fp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
