"""
Build every results table from the per-sample scores.

    python scripts/summarize_results.py                 # both datasets
    python scripts/summarize_results.py --dataset ms-cxr

Reads   results/<dataset>/persample/<model>_rqp-<off|on>.csv
Writes  results/<dataset>/tables/
            overall.csv        every model x stage: mIoU, Dice, CNR, P@1 with 95% CIs
            contrasts.csv      paired differences (delta, 95% CI, two-sided p)
            per_category.csv   every model x stage x finding type
            complete_misses.csv  IoU = 0 failures, AFLoc vs KAF-Ground under BEP
        results/<dataset>/summary.md   the main tables in markdown

Conventions (identical to the paper):
  * Stages: BEP (neither rule), +RQP, +DHS, +RQP+DHS.
  * Headline numbers are category-weighted: the mean over finding types of the
    per-type mean, so every finding type counts equally.
  * Confidence intervals: 1,000 bootstrap replicates (seed 0) that resample the
    unique image-phrase pairs with replacement. Every model and stage uses the
    same replicates, which is what makes a difference of two cells a paired
    statistic. The category-weighted mean is recomputed inside each replicate.
"""
import argparse
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N_BOOT, SEED = 1000, 0

MODELS = {   # key: (display name, reads the phrase -> RQP defined)
    "medklip":         ("MedKLIP", False),
    "gloria":          ("GLoRIA", True),
    "afloc":           ("AFLoc", True),
    "raddino-control": ("RAD-DINO control (no L_SG)", True),
    "kaf-ground":      ("KAF-Ground", True),
}
STAGES = {"BEP": ("off", "off"), "+RQP": ("on", "off"),
          "+DHS": ("off", "on"), "+RQP+DHS": ("on", "on")}
METRICS = [("iou", "mIoU"), ("dice", "Dice"), ("cnr", "CNR"), ("point", "P@1")]


def stages_of(model):
    return list(STAGES) if MODELS[model][1] else ["BEP", "+DHS"]


def best_stage(model):
    return "+RQP+DHS" if MODELS[model][1] else "+DHS"


def load(dataset):
    frames = []
    for model in MODELS:
        for st in stages_of(model):
            rqp, dhs = STAGES[st]
            f = ROOT / "results" / dataset / "persample" / f"{model}_rqp-{rqp}.csv"
            if not f.exists():
                print(f"  missing {f.relative_to(ROOT)}; skipping {model}")
                break
            d = pd.read_csv(f).sort_values(["idx", "dhs"])
            d = d[d.dhs == dhs].copy()
            d["model"], d["stage"] = model, st
            d["cell"] = f"{model}|{st}"
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def catmean(d, col):
    return float(np.nanmean(d.groupby("category")[col].mean()))


def bootstrap(df):
    """{(cell, metric): replicate means} with common random numbers."""
    common = sorted(set.intersection(*[set(g["rank"]) for _, g in df.groupby("cell")]))
    df = df[df["rank"].isin(common)]
    ranks = pd.Index(common)
    # one category per pair (two MS-CXR pairs carry two labels; the first is used)
    cat = df.drop_duplicates("rank").set_index("rank").loc[ranks, "category"].to_numpy()
    cats = sorted(set(cat))
    onehot = np.stack([cat == c for c in cats])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ranks), size=(N_BOOT, len(ranks)))

    boots = {}
    for col, _ in METRICS:
        piv = df.pivot_table(index="rank", columns="cell", values=col).loc[ranks]
        for cell in piv.columns:
            v = piv[cell].to_numpy()
            out = np.empty(N_BOOT)
            for r in range(N_BOOT):
                vv, oo = v[idx[r]], onehot[:, idx[r]]
                with np.errstate(invalid="ignore"):
                    out[r] = np.nanmean([np.nanmean(vv[o]) if o.any() else np.nan for o in oo])
            boots[(cell, col)] = out
    return boots, len(ranks)


def mcnemar_exact(b, c):
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) * 0.5 ** n)


def summarize(dataset):
    df = load(dataset)
    out = ROOT / "results" / dataset / "tables"
    out.mkdir(parents=True, exist_ok=True)
    boots, n_pairs = bootstrap(df)
    n_rows = int((df.cell == df.cell.iloc[0]).sum())
    print(f"{dataset}: {n_rows} pairs ({n_pairs} unique), {df.cell.nunique()} cells")

    # ---------------------------------------------------------------- levels
    level, rows = {}, []
    for cell, g in df.groupby("cell", sort=False):
        model, st = cell.split("|")
        row = dict(model=MODELS[model][0], stage=st)
        for col, name in METRICS:
            level[(cell, col)] = catmean(g, col)
            lo, hi = np.percentile(boots[(cell, col)], [2.5, 97.5])
            row.update({name: level[(cell, col)], f"{name}_lo": lo, f"{name}_hi": hi})
        rows.append(row)
    overall = pd.DataFrame(rows)
    overall.to_csv(out / "overall.csv", index=False)

    # ------------------------------------------------------------- contrasts
    pairs = []
    for m in MODELS:
        for st in stages_of(m)[1:]:
            pairs.append((f"{MODELS[m][0]}: {st} vs BEP", f"{m}|{st}", f"{m}|BEP"))
        if MODELS[m][1]:
            pairs.append((f"{MODELS[m][0]}: +RQP+DHS vs +RQP", f"{m}|+RQP+DHS", f"{m}|+RQP"))
    for m in MODELS:
        if m == "kaf-ground":
            continue
        for st in stages_of(m):
            pairs.append((f"KAF-Ground vs {MODELS[m][0]} @ {st}", f"kaf-ground|{st}", f"{m}|{st}"))
        pairs.append((f"KAF-Ground vs {MODELS[m][0]} @ best vs best",
                      "kaf-ground|+RQP+DHS", f"{m}|{best_stage(m)}"))
    for st in STAGES:
        pairs.append((f"RAD-DINO control vs AFLoc @ {st}", f"raddino-control|{st}", f"afloc|{st}"))

    crow = []
    for label, a, b in pairs:
        for col, name in METRICS:
            if (a, col) not in boots or (b, col) not in boots:
                continue
            d = boots[(a, col)] - boots[(b, col)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = min(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / N_BOOT), 1.0)
            crow.append(dict(contrast=label, metric=name,
                             delta=level[(a, col)] - level[(b, col)],
                             ci_lo=lo, ci_hi=hi, p=p,
                             significant=bool(lo > 0 or hi < 0)))
    # interaction of the two rules for KAF-Ground (is RQP + DHS additive?)
    if ("kaf-ground|+RQP+DHS", "iou") in boots:
        k = lambda s: boots[(f"kaf-ground|{s}", "iou")]
        L = lambda s: level[(f"kaf-ground|{s}", "iou")]
        d = k("+RQP+DHS") - k("+RQP") - k("+DHS") + k("BEP")
        lo, hi = np.percentile(d, [2.5, 97.5])
        crow.append(dict(contrast="KAF-Ground: RQP x DHS interaction", metric="mIoU",
                         delta=L("+RQP+DHS") - L("+RQP") - L("+DHS") + L("BEP"),
                         ci_lo=lo, ci_hi=hi,
                         p=min(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / N_BOOT), 1.0),
                         significant=bool(lo > 0 or hi < 0)))
    contrasts = pd.DataFrame(crow)
    contrasts.to_csv(out / "contrasts.csv", index=False)

    # ---------------------------------------------------------- per category
    prow = []
    for cell, g in df.groupby("cell", sort=False):
        model, st = cell.split("|")
        for c, s in g.groupby("category"):
            prow.append(dict(model=MODELS[model][0], stage=st, category=c, n=len(s),
                             IoU=s.iou.mean(), Dice=s.dice.mean(),
                             CNR=s.cnr.mean(), **{"P@1": s.point.mean()}))
    percat = pd.DataFrame(prow)
    percat.to_csv(out / "per_category.csv", index=False)

    # ------------------------------------------------------- complete misses
    a = df[df.cell == "afloc|BEP"].set_index(["idx"])
    k = df[df.cell == "kaf-ground|BEP"].set_index(["idx"])
    m = a[["category", "iou"]].join(k[["iou"]], rsuffix="_kaf")
    az, kz = (m.iou == 0), (m.iou_kaf == 0)
    mrow = []
    for c, s in m.groupby("category"):
        sa, sk = (s.iou == 0), (s.iou_kaf == 0)
        mrow.append(dict(category=c, n=len(s), afloc_misses=int(sa.sum()),
                         kaf_ground_misses=int(sk.sum()), recovered=int((sa & ~sk).sum()),
                         lost=int((~sa & sk).sum())))
    rec, lost = int((az & ~kz).sum()), int((~az & kz).sum())
    mrow.append(dict(category="all", n=len(m), afloc_misses=int(az.sum()),
                     kaf_ground_misses=int(kz.sum()), recovered=rec, lost=lost,
                     mcnemar_p=mcnemar_exact(rec, lost)))
    misses = pd.DataFrame(mrow)
    misses.to_csv(out / "complete_misses.csv", index=False)

    write_markdown(dataset, overall, contrasts, percat, misses, n_rows)


def _fmt(v, lo, hi, dp):
    return f"{v:.{dp}f} [{lo:.{dp}f}, {hi:.{dp}f}]"


def write_markdown(dataset, overall, contrasts, percat, misses, n):
    title = {"ms-cxr": "MS-CXR", "padchest-gr": "PadChest-GR (test split)"}[dataset]
    L = [f"# {title}: {n} image-phrase pairs", "",
         "Category-weighted means with 95% bootstrap intervals. "
         "BEP = no inference rule; RQP = RadGraph Query Pruning; "
         "DHS = Dynamic Heatmap Sharpening. MedKLIP does not read the phrase, "
         "so RQP is undefined for it.", "",
         "| model | stage | mIoU | Dice | CNR | P@1 |", "|---|---|---|---|---|---|"]
    order = [MODELS[m][0] for m in MODELS]
    for _, r in overall.assign(o=overall.model.map(order.index)).sort_values(["o"]).iterrows():
        L.append(f"| {r.model} | {r.stage} | " + " | ".join(
            _fmt(r[k], r[f'{k}_lo'], r[f'{k}_hi'], 3 if k == "CNR" else 4)
            for _, k in METRICS) + " |")
    L += ["", "## Paired differences in mIoU", "",
          "| contrast | delta | 95% CI | p |", "|---|---|---|---|"]
    for _, r in contrasts[contrasts.metric == "mIoU"].iterrows():
        L.append(f"| {r.contrast} | {r.delta:+.4f} | [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | {r.p:.3f} |")
    L += ["", "## IoU per finding type (BEP and full protocol)", ""]
    for st in ("BEP", "+RQP+DHS"):
        sub = percat[(percat.stage == st) | ((percat.model == "MedKLIP") & (percat.stage == ("+DHS" if st != "BEP" else "BEP")))]
        piv = sub.pivot_table(index="category", columns="model", values="IoU")
        piv = piv[[m for m in order if m in piv.columns]]
        n_of = sub.drop_duplicates("category").set_index("category").n
        L += [f"**{st}**" + (" (MedKLIP: +DHS)" if st != "BEP" else ""), "",
              "| finding | n | " + " | ".join(piv.columns) + " |",
              "|---|---|" + "---|" * len(piv.columns)]
        for c, r in piv.iterrows():
            L.append(f"| {c} | {int(n_of[c])} | " + " | ".join(f"{v:.4f}" for v in r) + " |")
        L.append("")
    tot = misses[misses.category == "all"].iloc[0]
    L += ["## Complete failures (IoU = 0) under BEP", "",
          f"AFLoc misses {tot.afloc_misses} pairs, KAF-Ground {tot.kaf_ground_misses}; "
          f"{tot.recovered} recovered, {tot.lost} lost (exact McNemar p = {tot.mcnemar_p:.2g}).", "",
          "| finding | n | AFLoc | KAF-Ground | recovered | lost |", "|---|---|---|---|---|---|"]
    for _, r in misses[misses.category != "all"].iterrows():
        L.append(f"| {r.category} | {r.n} | {r.afloc_misses} | {r.kaf_ground_misses} | "
                 f"{r.recovered} | {r.lost} |")
    (ROOT / "results" / dataset / "summary.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ms-cxr", "padchest-gr"], default=None)
    a = ap.parse_args()
    for d in ([a.dataset] if a.dataset else ["ms-cxr", "padchest-gr"]):
        summarize(d)
