"""
Prepare PadChest-GR for evaluation.

Converts the raw 16-bit PNGs into 8-bit square JPEGs with the same geometry as
the MIMIC-CXR / MS-CXR images (long side resized with INTER_AREA, short side
zero-padded at the centre) and maps the normalised grounding boxes into the
padded frame.

  python data/padchest/prepare_padchest_gr.py \
      --src  <PadChest-GR>/Padchest_GR_files \
      --json <PadChest-GR>/grounded_reports_20240819.json \
      --csv  <PadChest-GR>/master_table.csv \
      --out  data/padchest/padchest_gr_prepared \
      --sizes 518

The paper uses the 518 px output without --fix-inversion.
"""
import argparse, json, os, sys, hashlib, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import cv2

cv2.setNumThreads(1)  # we parallelise with processes


# ---------------------------------------------------------------- geometry
def resize_pad_params(h, w, s):
    """Mirror _resize_img: fit long side to s, centre-pad the short side."""
    if h >= w:
        nh, nw = s, int(w * s / h)
        pad_left = (s - nw) // 2
        pad_top = 0
    else:
        nw, nh = s, int(h * s / w)
        pad_top = (s - nh) // 2
        pad_left = 0
    return nh, nw, pad_top, pad_left


def resize_pad(img, s):
    h, w = img.shape[:2]
    nh, nw, pt, pl = resize_pad_params(h, w, s)
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.pad(out, [(pt, s - nh - pt), (pl, s - nw - pl)], mode="constant")
    return out


def remap_box(box, h, w, s):
    """Normalised [x1,y1,x2,y2] in the ORIGINAL frame -> padded s*s frame."""
    nh, nw, pt, pl = resize_pad_params(h, w, s)
    x1, y1, x2, y2 = box
    return [
        (x1 * nw + pl) / s,
        (y1 * nh + pt) / s,
        (x2 * nw + pl) / s,
        (y2 * nh + pt) / s,
    ]


# ------------------------------------------------------------ intensities
def to_uint8(img, lo_p=0.5, hi_p=99.5):
    """16-bit (or any) -> 8-bit via percentile window. PadChest PNGs rarely use
    the full 16-bit range, so naive /257 yields near-black images."""
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [lo_p, hi_p])
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            return np.zeros(x.shape, np.uint8)
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    return (x * 255.0).round().astype(np.uint8)


def looks_inverted(u8):
    """MONOCHROME1 heuristic: in a normal CXR the image corners (air/collimation)
    are dark relative to the body. Returns True if they are bright instead."""
    h, w = u8.shape
    ch, cw = max(h // 10, 1), max(w // 10, 1)
    corners = np.concatenate([
        u8[:ch, :cw].ravel(), u8[:ch, -cw:].ravel(),
        u8[-ch:, :cw].ravel(), u8[-ch:, -cw:].ravel(),
    ])
    return float(corners.mean()) > float(np.median(u8)) + 20.0


# ---------------------------------------------------------------- worker
def process_one(args):
    image_id, src, out_root, sizes, quality, fix_inversion = args
    rec = {"ImageID": image_id}
    try:
        path = os.path.join(src, image_id)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return {**rec, "status": "unreadable"}
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = img.shape[:2]
        rec.update(orig_h=h, orig_w=w, dtype=str(img.dtype),
                   raw_bytes=os.path.getsize(path))

        u8 = to_uint8(img)
        inv = looks_inverted(u8)
        rec["suspected_inverted"] = bool(inv)
        if inv and fix_inversion:
            u8 = 255 - u8
            rec["inverted_applied"] = True

        for s in sizes:
            d = os.path.join(out_root, f"images_{s}")
            os.makedirs(d, exist_ok=True)
            dst = os.path.join(d, image_id.replace(".png", ".jpg"))
            q = quality if s <= 600 else min(98, quality + 3)
            ok = cv2.imwrite(dst, resize_pad(u8, s),
                             [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                return {**rec, "status": "write_failed"}
            rec[f"bytes_{s}"] = os.path.getsize(dst)

        return {**rec, "status": "ok"}
    except Exception:
        return {**rec, "status": "error", "traceback": traceback.format_exc()[-800:]}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir holding the raw *.png")
    ap.add_argument("--json", required=True, help="grounded_reports_*.json")
    ap.add_argument("--csv", default=None, help="master_table.csv (copied through)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[518],
                    help="518 = working tier (matches MIMIC); add 1024 as archive")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--fix-inversion", action="store_true",
                    help="invert images flagged MONOCHROME1-like (inspect QC first)")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on N images")
    a = ap.parse_args()

    studies = json.load(open(a.json))
    if a.limit:
        studies = studies[: a.limit]
    os.makedirs(a.out, exist_ok=True)

    jobs = [(s["ImageID"], a.src, a.out, a.sizes, a.quality, a.fix_inversion)
            for s in studies]
    print(f"[i] {len(jobs)} images -> sizes {a.sizes}, {a.workers} workers", flush=True)

    recs = {}
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for fut in as_completed([ex.submit(process_one, j) for j in jobs]):
            r = fut.result()
            recs[r["ImageID"]] = r
            done += 1
            if done % 200 == 0 or done == len(jobs):
                print(f"    {done}/{len(jobs)}", flush=True)

    ok = {k: v for k, v in recs.items() if v["status"] == "ok"}
    bad = {k: v for k, v in recs.items() if v["status"] != "ok"}
    inverted = [k for k, v in ok.items() if v.get("suspected_inverted")]

    # ---- rewrite annotations into every padded frame -------------------
    for s in a.sizes:
        out_studies, rows = [], []
        for st in studies:
            m = ok.get(st["ImageID"])
            if m is None:
                continue
            h, w = m["orig_h"], m["orig_w"]
            new = dict(st)
            new["ImageID"] = st["ImageID"].replace(".png", ".jpg")
            new["orig_height"], new["orig_width"], new["frame"] = h, w, s
            nf = []
            for f in st["findings"]:
                f2 = dict(f)
                for key in ("boxes", "extra_boxes"):
                    if f.get(key):
                        f2[key] = [remap_box(b, h, w, s) for b in f[key]]
                nf.append(f2)
                for key in ("boxes", "extra_boxes"):
                    for b in f2.get(key) or []:
                        rows.append({
                            "image_id": new["ImageID"], "study_id": st["StudyID"],
                            "sentence": f.get("sentence_en", ""),
                            "label": "|".join(f.get("labels") or []),
                            "box_type": key, "abnormal": f.get("abnormal"),
                            "x1": round(b[0], 6), "y1": round(b[1], 6),
                            "x2": round(b[2], 6), "y2": round(b[3], 6),
                            "frame": s,
                        })
            new["findings"] = nf
            out_studies.append(new)

        json.dump(out_studies, open(f"{a.out}/grounded_reports_{s}.json", "w"),
                  ensure_ascii=False)
        import csv as _csv
        with open(f"{a.out}/boxes_{s}.csv", "w", newline="") as fh:
            wtr = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wtr.writeheader(); wtr.writerows(rows)
        print(f"[i] size {s}: {len(out_studies)} studies, {len(rows)} boxes")

    if a.csv and os.path.exists(a.csv):
        import shutil; shutil.copy(a.csv, os.path.join(a.out, "master_table.csv"))

    json.dump({"records": recs, "failed": list(bad),
               "suspected_inverted": inverted},
              open(f"{a.out}/manifest.json", "w"), indent=1)

    tot = {s: sum(v[f"bytes_{s}"] for v in ok.values()) for s in a.sizes}
    raw = sum(v["raw_bytes"] for v in ok.values())
    print(f"\n[✓] ok={len(ok)}  failed={len(bad)}  suspected-inverted={len(inverted)}")
    print(f"    raw {raw/2**30:.1f} GB")
    for s in a.sizes:
        print(f"    {s:>5}px -> {tot[s]/2**20:.0f} MB  ({raw/max(tot[s],1):.0f}x smaller)")
    if bad:
        print(f"    !! see {a.out}/manifest.json for the {len(bad)} failures")


if __name__ == "__main__":
    main()
