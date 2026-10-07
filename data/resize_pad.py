"""
Resize chest X-rays to the square frame used everywhere in this repository:
the long side is resized to --size with INTER_AREA and the short side is
zero-padded at the centre (the same geometry as prepare_padchest_gr.py and as
the box mapping in kafground/eval/datasets.py).

    python data/resize_pad.py --src <mimic-cxr-jpg>/files --dst data/mimic-cxr-jpg --size 518

The directory structure below --src (p1x/p1xxxxxxx/sxxxxxxxx/<dicom_id>.jpg) is
kept, so the same output serves training (MIMIC-CXR) and MS-CXR evaluation.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def resize_pad(img, s):
    h, w = img.shape[:2]
    if h >= w:
        nh, nw = s, int(w * s / h)
    else:
        nw, nh = s, int(h * s / w)
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    pt, pl = (s - nh) // 2, (s - nw) // 2
    return np.pad(out, [(pt, s - nh - pt), (pl, s - nw - pl)], mode="constant")


def one(args):
    src, dst, size = args
    if dst.exists():
        return
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), resize_pad(img, size), [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    jobs = [(p, dst / p.relative_to(src), a.size) for p in src.rglob("*.jpg")]
    print(f"{len(jobs)} images")
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(one, jobs, chunksize=64))


if __name__ == "__main__":
    main()
