"""
Build the PadChest-GR evaluation set used in the paper.

    python data/padchest/build_benchmark.py \
        --prepared data/padchest/padchest_gr_prepared \
        --out data/padchest/bench_testsplit.csv

Input is the output of prepare_padchest_gr.py (grounded_reports_518.json with
boxes already mapped into the padded 518 frame, plus master_table.csv).

The evaluation set is PadChest-GR's own test split, restricted to eight finding
types: the three shared with MS-CXR (cardiomegaly, pleural effusion,
atelectasis), two near-analogues (alveolar and interstitial pattern) and three
that MS-CXR does not contain (nodule, pleural thickening, aortic elongation).
That gives 530 image-phrase pairs on 359 images. Findings on the same image
with an identical sentence are merged into one pair with the union of their
boxes.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

CATEGORIES = ["cardiomegaly", "pleural effusion", "atelectasis",
              "alveolar pattern", "interstitial pattern",
              "nodule", "pleural thickening", "aortic elongation"]


def load_pairs(prepared: Path):
    studies = json.load(open(prepared / "grounded_reports_518.json"))
    master = pd.read_csv(prepared / "master_table.csv")

    split, group = {}, {}
    for r in master.itertuples():
        iid = r.ImageID.replace(".png", ".jpg")
        split[iid] = r.split
        if isinstance(r.sentence_en, str):
            group.setdefault((iid, r.sentence_en.strip()), r.label_group)

    merged = {}
    for st in studies:
        iid = st["ImageID"]
        for f in st["findings"]:
            sent = (f.get("sentence_en") or "").strip()
            if not f.get("boxes") or not sent:
                continue
            k = (iid, sent)
            if k in merged:
                merged[k]["boxes"].extend(f["boxes"])
                merged[k]["extra_boxes"].extend(f.get("extra_boxes") or [])
            else:
                merged[k] = dict(boxes=list(f["boxes"]),
                                 extra_boxes=list(f.get("extra_boxes") or []))

    rows = []
    for (iid, sent), v in merged.items():
        rows.append(dict(
            image_id=iid, sentence=sent,
            category=group.get((iid, sent), "UNMAPPED"),
            split=split.get(iid, "unknown"),
            n_boxes=len(v["boxes"]),
            boxes=json.dumps([[round(c, 6) for c in b] for b in v["boxes"]]),
            extra_boxes=json.dumps([[round(c, 6) for c in b] for b in v["extra_boxes"]]),
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default="data/padchest/padchest_gr_prepared")
    ap.add_argument("--out", default="data/padchest/bench_testsplit.csv")
    a = ap.parse_args()

    df = load_pairs(Path(a.prepared))
    test = df[(df.split == "test") & (df.category.isin(CATEGORIES))]
    test = test.sort_values(["category", "image_id", "sentence"])
    test.to_csv(a.out, index=False)
    print(f"{len(test)} pairs on {test.image_id.nunique()} images -> {a.out}")
    print(test.category.value_counts().reindex(CATEGORIES).to_string())


if __name__ == "__main__":
    main()
