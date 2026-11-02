"""性能剖析：分别测数据读取/增强耗时与 GPU 单步耗时，用于确定训练配置。"""
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

sys.argv = ["bench"]

from dataset import RemoteSensingDataset, build_dataloaders
from unet import UNet

N = 64
ds = RemoteSensingDataset("data/train_images", "data/train_masks", image_size=512, train=True)
t0 = time.time()
for i in range(N):
    ds[i]
dt = time.time() - t0
print(f"数据集读取+增强: {dt / N * 1000:.1f} ms/样本 -> 700 张约 {dt / N * 700:.1f} s")

device = torch.device("cuda")
model = UNet().to(device).to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
crit = torch.nn.BCEWithLogitsLoss()

for bs in (8, 12):
    x = torch.randn(bs, 3, 512, 512, device=device).to(memory_format=torch.channels_last)
    y = (torch.rand(bs, 1, 512, 512, device=device) > 0.9).float()
    # warmup
    for _ in range(2):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    steps = 5
    for _ in range(steps):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    mem = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"batch={bs}: {dt * 1000:.0f} ms/step ({bs / dt:.1f} img/s), peak VRAM {mem:.2f} GB")
