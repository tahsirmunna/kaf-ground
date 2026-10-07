"""
Evaluation data: MS-CXR and PadChest-GR as phrase / ground-truth-mask pairs.

Both loaders return the same columns:

    path  label_text  gtmasks  boxes  category

with every GT mask rasterised at EVAL_IMG_SIZE (518 for all models in the paper;
models that run at a lower resolution have their maps resized onto this GT).

Locations are taken from the environment (see README):

    KAF_MSCXR_JSON       MS_CXR_Local_Alignment_v1.1.0.json
    KAF_MSCXR_IMG_DIR    root holding p10/p10xxxxxx/s5xxxxxxx/<dicom_id>.jpg
    KAF_PADCHEST_DIR     output of data/padchest/prepare_padchest_gr.py
    KAF_PADCHEST_BENCH   csv written by data/padchest/build_benchmark.py

MS-CXR images are the MIMIC-CXR JPGs resized so the long side is 518 and
centre-padded to a square, the same geometry used in training (see
data/mimic/README.md).
"""
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_IMG_SIZE = 518


def set_eval_img_size(size: int):
    """GT masks are rasterised at this size; it must match the heatmaps."""
    global EVAL_IMG_SIZE
    EVAL_IMG_SIZE = int(size)


def pair_id(path, phrase: str) -> str:
    """
    Stable, shareable id for an image-phrase pair: a hash of the image file
    name and the phrase. Released per-sample results are keyed on this, so they
    contain no dataset text or local paths but can be joined back to a local
    copy of the data.
    """
    return hashlib.sha1(f"{Path(str(path)).name}|{phrase}".encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# geometry helpers                                                             #
# --------------------------------------------------------------------------- #
def box_mapping(box, w, h, scale):
    """Map an [x, y, w, h] box from the original w x h image into the
    aspect-preserving resized + centre-padded scale x scale frame."""
    size    = (h, w)
    max_dim = max(size)
    max_ind = size.index(max_dim)
    box     = np.array(box)

    if max_ind == 0:
        wpercent        = scale / float(size[0])
        hsize           = int(float(size[1]) * float(wpercent))
        desireable_size = (scale, hsize)
        box             = box * wpercent
    else:
        hpercent        = scale / float(size[1])
        wsize           = int(float(size[0]) * float(hpercent))
        desireable_size = (wsize, scale)
        box             = box * hpercent

    if max_ind == 0:
        pad_size = scale - desireable_size[1]
        left, top = int(np.floor(pad_size / 2)), 0
    else:
        pad_size = scale - desireable_size[0]
        left, top = 0, int(np.floor(pad_size / 2))

    box[0] = int(np.floor(box[0] + left))
    box[1] = int(np.floor(box[1] + top))
    box[2] = int(np.floor(box[2]))
    box[3] = int(np.floor(box[3]))
    return box.astype(np.int32)


def box2mask(box, w, h):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = 1
    return mask


def norm_heatmap(heatmap_, nan, mode=0):
    """Min-max normalise the valid pixels to [-1, 1] (mode 0) or [0, 1] (mode 1)."""
    heatmap        = copy.deepcopy(heatmap_)
    heatmap_wo_nan = heatmap[~nan]
    if heatmap_wo_nan.max() - heatmap_wo_nan.min() == 0:
        return heatmap_wo_nan
    heatmap_wo_nan = ((heatmap_wo_nan - heatmap_wo_nan.min())
                      / (heatmap_wo_nan.max() - heatmap_wo_nan.min()))
    if mode == 0:
        heatmap_wo_nan = heatmap_wo_nan * 2 - 1
    heatmap[~nan] = heatmap_wo_nan
    return heatmap


# --------------------------------------------------------------------------- #
# MS-CXR                                                                       #
# --------------------------------------------------------------------------- #
def _merge_annotation(path_to_json, scale):
    """One entry per (image, phrase); boxes of the same phrase are merged."""
    from pycocotools.coco import COCO

    coco   = COCO(annotation_file=str(path_to_json))
    cats   = coco.cats
    merged = {"path": [], "gtmasks": [], "label_text": [], "boxes": [], "category": []}

    for img_id, anns in coco.imgToAnns.items():
        path = coco.loadImgs(img_id)[0]["path"]
        mask_dct, bbox_dct, cats_dct = {}, {}, {}
        for ann in anns:
            tbox       = box_mapping(ann["bbox"], ann["width"], ann["height"], scale)
            mask       = box2mask(tbox, scale, scale)
            category   = cats[ann["category_id"]]["name"]
            label_text = ann["label_text"].lower()
            if label_text not in mask_dct:
                mask_dct[label_text] = mask
                bbox_dct[label_text] = [tbox]
            else:
                mask_dct[label_text] += mask
                bbox_dct[label_text].append(tbox)
            cats_dct[label_text] = category

        for k, v in mask_dct.items():
            merged["path"].append(path)
            merged["gtmasks"].append(v)
            merged["label_text"].append(k)
            merged["boxes"].append(bbox_dct[k])
            merged["category"].append(cats_dct[k])
    return merged


def load_ms_cxr(json_path=None, img_dir=None, **kwargs):
    """All 1,162 MS-CXR phrase-region pairs (the benchmark has no usable
    validation split; every number in the paper uses the full set)."""
    json_path = Path(json_path or os.environ.get(
        "KAF_MSCXR_JSON", "data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.json"))
    img_dir = Path(img_dir or os.environ.get("KAF_MSCXR_IMG_DIR", "data/ms-cxr/images"))
    data = _merge_annotation(json_path, scale=EVAL_IMG_SIZE)
    data["path"] = [img_dir / p.replace("files/", "") for p in data["path"]]
    return data


# --------------------------------------------------------------------------- #
# PadChest-GR                                                                  #
# --------------------------------------------------------------------------- #
# Images were resized and padded to a square frame by prepare_padchest_gr.py
# and the boxes stored normalised in that frame, so a box is valid at every
# eval scale: the GT mask is simply box * size.

def _mask_from_norm_boxes(boxes, size):
    m = np.zeros((size, size), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        a, b = int(np.floor(x1 * size)), int(np.floor(y1 * size))
        c, d = int(np.ceil(x2 * size)), int(np.ceil(y2 * size))
        a, b, c, d = max(a, 0), max(b, 0), min(c, size), min(d, size)
        if c > a and d > b:
            m[b:d, a:c] = 1
    return m


def _px_boxes(boxes, size):
    out = []
    for x1, y1, x2, y2 in boxes:
        a, b = int(np.floor(x1 * size)), int(np.floor(y1 * size))
        c, d = int(np.ceil(x2 * size)), int(np.ceil(y2 * size))
        out.append(np.array([a, b, max(c - a, 1), max(d - b, 1)], dtype=np.int32))
    return out


def load_padchest_gr(bench_csv=None, img_dir=None, gt="boxes", **kwargs):
    """
    The 530 PadChest-GR test-split pairs in the eight evaluated categories.

    gt: "boxes" (primary annotation, used in the paper) or "extra_boxes"
        (second annotation, for the sensitivity check).
    """
    root = Path(os.environ.get("KAF_PADCHEST_DIR", "data/padchest/padchest_gr_prepared"))
    bench_csv = Path(bench_csv or os.environ.get(
        "KAF_PADCHEST_BENCH", "data/padchest/bench_testsplit.csv"))
    img_dir = Path(img_dir or root / "images_518")
    if not bench_csv.exists():
        raise FileNotFoundError(f"{bench_csv}: run data/padchest/build_benchmark.py first")

    df = pd.read_csv(bench_csv)
    size = EVAL_IMG_SIZE
    paths, texts, masks, boxes_l, cats = [], [], [], [], []
    for r in df.itertuples():
        bx = json.loads(getattr(r, gt))
        if not bx:
            continue
        m = _mask_from_norm_boxes(bx, size)
        if m.sum() == 0:
            continue
        paths.append(img_dir / r.image_id)
        texts.append(r.sentence)
        masks.append(m)
        boxes_l.append(_px_boxes(bx, size))
        cats.append(r.category)

    keys = [str(p) + t for p, t in zip(paths, texts)]
    assert len(set(keys)) == len(keys), "duplicate (image, phrase) pair"
    return pd.DataFrame({"path": paths, "label_text": texts, "gtmasks": masks,
                         "boxes": boxes_l, "category": cats})


DATASETS = {"ms-cxr": load_ms_cxr, "padchest-gr": load_padchest_gr}


def load_data(dataset, **kwargs):
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset '{dataset}' (choose from {list(DATASETS)})")
    return DATASETS[dataset](**kwargs)
