"""
Compute one heatmap per image-phrase pair and save them as {pair_id: heatmap}.

    python scripts/dump_heatmaps.py --model kaf-ground --ckpt checkpoints/kaf_ground.ckpt \
        --dataset ms-cxr --rqp on --out runs/heatmaps/kaf-ground_ms-cxr_rqp-on.npy

--model
    kaf-ground  this work (word-max read-out on the shallow head)
    afloc       released AFLoc checkpoint (report matching); needs the official
                AFLoc code as `afloc` on PYTHONPATH, see baselines/README.md
    gloria, medklip   see baselines/README.md
--rqp on|off   RadGraph Query Pruning on the query side
               (DHS is applied later, by score_heatmaps.py)

All heatmaps are written at the ground-truth size (518); models that work at a
lower resolution are resized onto it, so every model is scored against the same
rasterised boxes.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines"))
from kafground.eval import datasets as D   # noqa: E402


def build_engine(a):
    rqp = a.rqp == "on"
    if a.model == "kaf-ground":
        from kafground.inference.engine import KAFGroundEngine
        return KAFGroundEngine(a.ckpt, rqp=rqp, head=a.head, query_mode=a.query_mode,
                               views=a.views, tokens=a.tokens, role_lexicon=a.role_lexicon)
    if a.model == "afloc":
        from kafground.inference.engine import AFLocEngine
        return AFLocEngine(a.ckpt, rqp=rqp)
    if a.model == "gloria":
        from gloria_engine import GloriaEngine
        return GloriaEngine(a.ckpt, rqp=rqp)
    if a.model == "medklip":
        if rqp:
            sys.exit("MedKLIP does not read the phrase, so RQP is undefined for it.")
        from medklip_engine import MedKLIPEngine
        return MedKLIPEngine(a.ckpt, dataset=a.dataset)
    raise ValueError(a.model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["kaf-ground", "afloc", "gloria", "medklip"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True, choices=list(D.DATASETS))
    ap.add_argument("--rqp", default="off", choices=["off", "on"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-size", type=int, default=518)
    # ablation options (kaf-ground only); see results/ablations/README.md
    ap.add_argument("--head", default="l", choices=["l", "l2", "lf"])
    ap.add_argument("--query-mode", default=None,
                    choices=["orig", "core", "ens", "rg-core", "rg-ens"],
                    help="overrides --rqp")
    ap.add_argument("--views", default="id@518", help='e.g. "id@518,hflip@518"')
    ap.add_argument("--tokens", default="all", choices=["all", "rg-mask", "rg-mean", "rg-anat"])
    ap.add_argument("--role-lexicon", default=None,
                    help="json from build_rqp_lexicon.py --roles (needed for --tokens)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pairs")
    a = ap.parse_args()

    D.set_eval_img_size(a.gt_size)
    data = D.load_data(a.dataset)
    n = len(data["path"]) if not a.limit else min(a.limit, len(data["path"]))
    engine = build_engine(a)

    out = {}
    for i in tqdm(range(n)):
        path, phrase = data["path"][i], data["label_text"][i]
        h = engine.heatmap(path, phrase, category=data["category"][i]) \
            if a.model == "medklip" else engine.heatmap(path, phrase)
        gt_shape = np.asarray(data["gtmasks"][i]).shape
        if h.shape != gt_shape:
            h = cv2.resize(h, (gt_shape[1], gt_shape[0]))
        out[D.pair_id(path, phrase)] = h

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, out)
    print(f"wrote {len(out)} heatmaps to {a.out}")


if __name__ == "__main__":
    main()
