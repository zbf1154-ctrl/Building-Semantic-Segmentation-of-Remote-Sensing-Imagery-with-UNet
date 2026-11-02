"""数据体检：统计尺寸/通道/位深/掩膜取值，供后续预处理决策使用。"""
import os
import sys
from collections import Counter

import cv2
import numpy as np

cv2.setNumThreads(0)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SPLITS = ["train", "val", "test"]


def probe(split):
    img_dir = os.path.join(DATA, f"{split}_images")
    msk_dir = os.path.join(DATA, f"{split}_masks")
    imgs = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".tif"))
    msks = set(f for f in os.listdir(msk_dir) if f.lower().endswith(".tif"))
    shapes, dtypes, chans, vals = Counter(), Counter(), Counter(), Counter()
    no_mask, mismatched = [], []
    for f in imgs:
        im = cv2.imread(os.path.join(img_dir, f), cv2.IMREAD_UNCHANGED)
        if im is None:
            no_mask.append(f)
            continue
        shapes[im.shape] += 1
        dtypes[str(im.dtype)] += 1
        chans[im.shape[2] if im.ndim == 3 else 1] += 1
        m = cv2.imread(os.path.join(msk_dir, f), cv2.IMREAD_UNCHANGED)
        if m is None:
            no_mask.append(f)
            continue
        if m.shape[:2] != im.shape[:2]:
            mismatched.append((f, im.shape[:2], m.shape[:2]))
        sample = np.unique(m)
        if len(sample) <= 8:
            vals[tuple(int(x) for x in sample)] += 1
        else:
            vals[f"many({len(sample)})"] += 1
    print(f"\n===== {split} =====")
    print(f"影像数={len(imgs)}  掩膜数={len(msks)}  缺掩膜={len(no_mask)} {no_mask[:5]}")
    print(f"尺寸分布: {dict(shapes)}")
    print(f"dtype分布: {dict(dtypes)}")
    print(f"通道分布: {dict(chans)}")
    print(f"掩膜取值分布(top5): {vals.most_common(5)}")
    print(f"尺寸不一致: {len(mismatched)} {mismatched[:5]}")
    return set(imgs), msks


if __name__ == "__main__":
    all_imgs, all_msks = set(), set()
    for s in SPLITS:
        i, m = probe(s)
        all_imgs |= {f"{s}/{x}" for x in i}
        all_msks |= {f"{s}/{x}" for x in m}
    print("\n===== 汇总 =====")
    print(f"总影像 {len(all_imgs)} / 总掩膜 {len(all_msks)}")
    dup = [x for x in all_msks if x.split("/")[1] in {y.split("/")[1] for y in all_imgs}]
    print(f"掩膜文件名在影像集合中出现过的数量: {len(dup)}")
    sys.stdout.flush()
