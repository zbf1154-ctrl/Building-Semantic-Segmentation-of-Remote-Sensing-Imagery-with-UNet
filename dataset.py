"""遥感影像建筑物提取数据集。

相比原始版本的改进：
1. 统一 resize 到 image_size（默认 512），消除尺寸不一致导致的 stack 失败；
2. 兼容 uint8 / uint16 / 单波段 / 4 波段影像，按位深正确归一化；
3. 训练集提供在线数据增强（翻转、旋转、缩放裁剪、亮度对比度、噪声）；
4. 掩膜统一为 {0,1} 的 float32，避免 0/255 阈值残留；
5. Windows 下多进程 DataLoader 的 cv2 线程配置。
"""
from __future__ import annotations

import os
import random
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def _read_image(path: str) -> np.ndarray:
    """读取影像并统一成 HxWx3 的 uint8 RGB。"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取影像文件: {path}")

    # 位深归一化：16 位 / 浮点遥感影像缩放到 8 位动态范围
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        img = (img * 255.0 if img.max() <= 1.0 else img).clip(0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # 通道归一化：单波段 / 4 波段（含近红外）统一成 3 通道
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(img)


def _read_mask(path: str) -> np.ndarray:
    """读取标签并二值化为 {0,1} 的 uint8。"""
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"无法读取对应的标签文件: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype != np.uint8:
        mask = mask.astype(np.float32)
        mask = (mask / max(float(mask.max()), 1e-6) * 255.0).round().astype(np.uint8)
    # 兼容 0/1 与 0/255 两种标注方式
    thr = 1 if mask.max() <= 1 else 127
    return (mask > thr).astype(np.uint8)


class RemoteSensingDataset(Dataset):
    """遥感影像与建筑物标签标准读取器（带增强）。"""

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_size: int = 512,
        train: bool = False,
        augment: Optional[bool] = None,
        cache: bool = False,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.train = train
        self.augment = train if augment is None else augment
        # 内存缓存：单进程加载时缓存已解码的图，避免每个 epoch 重复解码 TIFF
        self.cache = cache
        self._cache: dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"影像目录不存在: {images_dir}")
        if not os.path.isdir(masks_dir):
            raise FileNotFoundError(f"标签目录不存在: {masks_dir}")

        self.filenames = sorted(
            f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS)
        )
        if not self.filenames:
            raise RuntimeError(f"影像目录中没有可用影像: {images_dir}")

        # 只保留成对存在的样本，并提示缺失情况
        missing = {f for f in self.filenames if not os.path.isfile(os.path.join(masks_dir, f))}
        if missing:
            print(f"[警告] {images_dir} 中有 {len(missing)} 张影像缺少同名掩膜，已跳过，例如 {sorted(missing)[:3]}")
            self.filenames = [f for f in self.filenames if f not in missing]

    def __len__(self) -> int:
        return len(self.filenames)

    # ---------- 增强 ----------
    @staticmethod
    def _augment(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]

        # 几何：水平/垂直翻转
        if random.random() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        if random.random() < 0.5:
            img, mask = img[::-1, :], mask[::-1, :]

        # 几何：90/180/270 旋转（建筑物无方向性）
        k = random.randint(0, 3)
        if k:
            img = np.ascontiguousarray(np.rot90(img, k))
            mask = np.ascontiguousarray(np.rot90(mask, k))

        # 几何：随机缩放 + 回裁到原尺寸，模拟不同空间分辨率
        if random.random() < 0.5:
            scale = random.uniform(0.85, 1.2)
            nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
            if nh >= h and nw >= w:
                top = random.randint(0, nh - h)
                left = random.randint(0, nw - w)
                img, mask = img[top:top + h, left:left + w], mask[top:top + h, left:left + w]
            else:
                pad_h, pad_w = max(0, h - nh), max(0, w - nw)
                img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
                mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        # 光度：亮度 / 对比度抖动
        if random.random() < 0.6:
            alpha = random.uniform(0.9, 1.12)
            beta = random.uniform(-12, 12)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # 光度：轻微高斯噪声
        if random.random() < 0.2:
            noise = np.random.normal(0, 5.0, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(img), np.ascontiguousarray(mask)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = self.filenames[idx]
        img_path = os.path.join(self.images_dir, filename)
        mask_path = os.path.join(self.masks_dir, filename)

        cached = self._cache.get(idx)
        if cached is not None:
            image, mask = cached
        else:
            image = _read_image(img_path)
            mask = _read_mask(mask_path)

            # 统一尺寸（网络要求边长可被 16 整除；512 满足）
            if image.shape[:2] != (self.image_size, self.image_size):
                image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

            if self.cache:
                image = np.ascontiguousarray(image)
                mask = np.ascontiguousarray(mask)
                self._cache[idx] = (image, mask)

        if self.augment:
            image, mask = self._augment(image, mask)

        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).contiguous()
        return image_tensor, mask_tensor


def build_dataloaders(
    data_root: str,
    image_size: int = 512,
    batch_size: int = 8,
    workers: int = 4,
    pin_memory: bool = True,
    cache: bool = False,
):
    """构建 train/val/test 三个 DataLoader。"""
    loaders, datasets = {}, {}
    for split in ("train", "val", "test"):
        ds = RemoteSensingDataset(
            images_dir=os.path.join(data_root, f"{split}_images"),
            masks_dir=os.path.join(data_root, f"{split}_masks"),
            image_size=image_size,
            train=(split == "train"),
            cache=cache,
        )
        datasets[split] = ds
        loaders[split] = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
            persistent_workers=workers > 0,
            prefetch_factor=2 if workers > 0 else None,
        )
    return loaders, datasets
