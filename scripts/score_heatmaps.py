"""
Score a heatmap dump with and without Dynamic Heatmap Sharpening.

    python scripts/score_heatmaps.py --dataset ms-cxr \
        --heatmaps runs/heatmaps/kaf-ground_ms-cxr_rqp-on.npy \
        --out results/ms-cxr/persample/kaf-ground_rqp-on.csv

Input: the dict {pair_id: heatmap} written by dump_heatmaps.py, heatmaps at the
ground-truth resolution (518 x 518).

For every pair the margin is applied first (pixels outside the central crop
become NaN and are ignored), then

    DHS off : IoU / Dice averaged over thresholds 0.1..0.5, CNR
    DHS on  : the same on the sharpened map
    P@1     : arg-max inside the box, always from the unsharpened map

Output: one row per (pair, DHS setting) with columns
    idx rank pair_id category dhs q point iou dice cnr
`rank` is the position of the pair in sorted order of (relative image path +
phrase); the bootstrap in summarize_results.py resamples pairs in that order.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kafground.eval import datasets as D                       # noqa: E402
from kafground.eval.metrics import (compute_cnr, compute_dice,   # noqa: E402
                                    compute_iou, pointing_game, threshold_mean)
from kafground.inference.dhs import sharpen                      # noqa: E402
from kafground.inference.engine import set_margin                # noqa: E402


def sort_keys(data, dataset):
    """Relative image path + phrase, the order the bootstrap resamples in."""
    if dataset == "ms-cxr":
        rel = [str(Path(*Path(p).parts[-4:])) for p in data["path"]]   # p1x/p1x.../s.../x.jpg
    else:
        rel = [Path(p).name for p in data["path"]]
    return [r + t for r, t in zip(rel, data["label_text"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(D.DATASETS))
    ap.add_argument("--heatmaps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-size", type=int, default=518)
    ap.add_argument("--no-margin", action="store_true")
    ap.add_argument("--gt", default="boxes", choices=["boxes", "extra_boxes"],
                    help="PadChest-GR only: which annotation is the ground truth")
    a = ap.parse_args()

    D.set_eval_img_size(a.gt_size)
    data = D.load_data(a.dataset, gt=a.gt)
    hm = np.load(a.heatmaps, allow_pickle=True).item()

    skeys = sort_keys(data, a.dataset)
    rank = {k: i for i, k in enumerate(sorted(set(skeys)))}

    rows, missing = [], 0
    for i in range(len(data["path"])):
        pid = D.pair_id(data["path"][i], data["label_text"][i])
        if pid not in hm:
            missing += 1
            continue
        h = np.array(hm[pid], dtype=np.float32, copy=True)
        gt = np.asarray(data["gtmasks"][i])
        assert h.shape == gt.shape, f"heatmap {h.shape} vs ground truth {gt.shape}"
        if not a.no_margin:
            h = set_margin(h)
        nan = np.isnan(h)

        h_off = D.norm_heatmap(h, nan, mode=0)
        point = pointing_game(gt, h_off, nan)
        sharp, q = sharpen(h, nan)
        h_on = D.norm_heatmap(sharp, nan, mode=0)

        base = dict(idx=i, rank=rank[skeys[i]], pair_id=pid,
                    category=data["category"][i], q=q, point=point)
        for dhs, hh in (("off", h_off), ("on", h_on)):
            rows.append(dict(base, dhs=dhs,
                             iou=threshold_mean(compute_iou, gt, hh, nan),
                             dice=threshold_mean(compute_dice, gt, hh, nan),
                             cnr=float(compute_cnr(gt, hh, nan))))

    if missing:
        print(f"[score] warning: {missing} pairs have no heatmap")
    df = pd.DataFrame(rows)[["idx", "rank", "pair_id", "category", "dhs",
                             "q", "point", "iou", "dice", "cnr"]]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)

    for dhs in ("off", "on"):
        s = df[df.dhs == dhs]
        cw = lambda c: float(np.nanmean(s.groupby("category")[c].mean()))
        print(f"DHS {dhs:<3}  mIoU {cw('iou'):.4f}  Dice {cw('dice'):.4f}  "
              f"CNR {cw('cnr'):.3f}  P@1 {cw('point'):.4f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
