"""
Check the numbers stated in the paper against the regenerated tables.

    python scripts/summarize_results.py
    python scripts/check_paper_numbers.py

Each check prints the stated value next to the value recomputed from
results/*/tables; the script exits with an error if any check fails.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = {d: ROOT / "results" / d / "tables" for d in ("ms-cxr", "padchest-gr")}
ROWS = []


def check(claim, stated, actual, dp=4):
    ok = abs(round(float(actual), dp) - stated) < 10 ** (-dp) / 2 + 1e-12
    ROWS.append((ok, claim, stated, float(actual)))


def truth(claim, ok):
    ROWS.append((bool(ok), claim, None, None))


ov = {d: pd.read_csv(t / "overall.csv") for d, t in T.items()}
co = {d: pd.read_csv(t / "contrasts.csv") for d, t in T.items()}
pc = {d: pd.read_csv(t / "per_category.csv") for d, t in T.items()}
mi = {d: pd.read_csv(t / "complete_misses.csv") for d, t in T.items()}


def lvl(d, model, stage, metric="mIoU", part=""):
    r = ov[d][(ov[d].model == model) & (ov[d].stage == stage)]
    assert len(r) == 1, (d, model, stage)
    return float(r[metric + part].iloc[0])


def con(d, label, metric="mIoU"):
    r = co[d][(co[d].contrast == label) & (co[d].metric == metric)]
    assert len(r) == 1, (d, label)
    r = r.iloc[0]
    return r.delta, r.ci_lo, r.ci_hi


def cat(d, model, stage, category, col="IoU"):
    r = pc[d][(pc[d].model == model) & (pc[d].stage == stage) & (pc[d].category == category)]
    return float(r[col].iloc[0])


KAF, AF, GL, MK = "KAF-Ground", "AFLoc", "GLoRIA", "MedKLIP"
MS, PG = "ms-cxr", "padchest-gr"

# ---------------------------------------------------- Table 1 (BEP, with CIs)
TABLE1 = {  # (value, lo, hi) for mIoU, Dice, CNR, P@1
    (MS, MK):  [(0.1871, 0.179, 0.194), (0.2935, 0.283, 0.303), (0.952, 0.919, 0.986), (0.4121, 0.378, 0.444)],
    (MS, GL):  [(0.2424, 0.232, 0.252), (0.3638, 0.351, 0.376), (1.280, 1.225, 1.341), (0.6561, 0.623, 0.687)],
    (MS, AF):  [(0.3232, 0.314, 0.332), (0.4610, 0.450, 0.472), (1.631, 1.578, 1.682), (0.7843, 0.758, 0.806)],
    (MS, KAF): [(0.3379, 0.329, 0.347), (0.4817, 0.472, 0.492), (1.672, 1.628, 1.714), (0.8194, 0.793, 0.842)],
    (PG, MK):  [(0.1847, 0.175, 0.195), (0.2763, 0.263, 0.290), (0.828, 0.772, 0.888), (0.3921, 0.356, 0.429)],
    (PG, GL):  [(0.2080, 0.196, 0.222), (0.3108, 0.295, 0.330), (1.220, 1.133, 1.322), (0.5556, 0.515, 0.597)],
    (PG, AF):  [(0.2738, 0.260, 0.288), (0.3888, 0.372, 0.405), (1.412, 1.335, 1.491), (0.6241, 0.586, 0.659)],
    (PG, KAF): [(0.2894, 0.276, 0.303), (0.4181, 0.403, 0.435), (1.572, 1.514, 1.634), (0.7661, 0.733, 0.803)],
}
for (d, m), vals in TABLE1.items():
    for metric, (v, lo, hi) in zip(("mIoU", "Dice", "CNR", "P@1"), vals):
        dp = 3 if metric == "CNR" else 4
        check(f"Table 1 {d} {m} {metric}", v, lvl(d, m, "BEP", metric), dp)
        check(f"  lower bound", lo, lvl(d, m, "BEP", metric, "_lo"), 3)
        check(f"  upper bound", hi, lvl(d, m, "BEP", metric, "_hi"), 3)

# ------------------------------------------------------ comparison under BEP
for d, other, stated, lo, hi in [(MS, AF, 0.0147, 0.0072, 0.0231), (PG, AF, 0.0155, 0.0049, 0.0273)]:
    dd, l, h = con(d, f"KAF-Ground vs {other} @ BEP")
    check(f"{d}: KAF-Ground - AFLoc at BEP", stated, dd)
    check("  ci lo", lo, l); check("  ci hi", hi, h)
check("MS-CXR: KAF-Ground - GLoRIA at BEP", 0.0955, con(MS, "KAF-Ground vs GLoRIA @ BEP")[0])
check("MS-CXR: KAF-Ground - MedKLIP at BEP", 0.1508, con(MS, "KAF-Ground vs MedKLIP @ BEP")[0])

# ------------------------------------------------------------ inference rules
d, l, h = con(MS, "KAF-Ground: +RQP vs BEP")
check("MS-CXR: RQP step for KAF-Ground", 0.0056, d)
check("  ci lo", 0.0033, l); check("  ci hi", 0.0083, h)
check("MS-CXR: RQP step, P@1 after", 0.838, lvl(MS, KAF, "+RQP", "P@1"), 3)
check("MS-CXR: RQP step, CNR after", 1.703, lvl(MS, KAF, "+RQP", "CNR"), 3)
check("MS-CXR: RQP step for AFLoc", 0.0012, con(MS, "AFLoc: +RQP vs BEP")[0])
check("MS-CXR: RQP step for GLoRIA", 0.0001, con(MS, "GLoRIA: +RQP vs BEP")[0])
check("PadChest-GR: RQP step for KAF-Ground", 0.0014, con(PG, "KAF-Ground: +RQP vs BEP")[0])
check("PadChest-GR: P@1 after RQP", 0.780, lvl(PG, KAF, "+RQP", "P@1"), 3)

for d, stated, lo, hi in [(MS, 0.0258, 0.0177, 0.0333), (PG, 0.0140, 0.0054, 0.0227)]:
    dd, l, h = con(d, "KAF-Ground: +DHS vs BEP")
    check(f"{d}: DHS step for KAF-Ground", stated, dd)
    check("  ci lo", lo, l); check("  ci hi", hi, h)
check("MS-CXR: Dice after DHS", 0.510, lvl(MS, KAF, "+DHS", "Dice"), 3)
for m, stated in [(GL, 0.0475), (AF, 0.0177), (MK, 0.0140)]:
    check(f"MS-CXR: DHS step for {m}", stated, con(MS, f"{m}: +DHS vs BEP")[0])

check("MS-CXR: both rules", 0.0293, con(MS, "KAF-Ground: +RQP+DHS vs BEP")[0])
check("MS-CXR: interaction", -0.0021, con(MS, "KAF-Ground: RQP x DHS interaction")[0])
check("PadChest-GR: both rules", 0.0188, con(PG, "KAF-Ground: +RQP+DHS vs BEP")[0])
check("MS-CXR: full protocol mIoU", 0.3672, lvl(MS, KAF, "+RQP+DHS"))
check("MS-CXR: full protocol Dice", 0.514, lvl(MS, KAF, "+RQP+DHS", "Dice"), 3)
check("PadChest-GR: full protocol mIoU", 0.3082, lvl(PG, KAF, "+RQP+DHS"))
check("MS-CXR: relative gain over AFLoc BEP (%)", 13.6,
      100 * (lvl(MS, KAF, "+RQP+DHS") / lvl(MS, AF, "BEP") - 1), 1)
check("PadChest-GR: relative gain over AFLoc BEP (%)", 12.5,
      100 * (lvl(PG, KAF, "+RQP+DHS") / lvl(PG, AF, "BEP") - 1), 1)
for d, stated, lo, hi in [(MS, 0.0255, 0.0194, 0.0321), (PG, 0.0163, 0.0071, 0.0263)]:
    dd, l, h = con(d, "KAF-Ground vs AFLoc @ best vs best")
    check(f"{d}: margin over AFLoc with both rules", stated, dd)
    check("  ci lo", lo, l); check("  ci hi", hi, h)

# --------------------------------------------------------- complete failures
for d, a, k, rec, lost in [(MS, 63, 5, 59, 1), (PG, 30, 5, 28, 3)]:
    r = mi[d][mi[d].category == "all"].iloc[0]
    truth(f"{d}: AFLoc misses {r.afloc_misses}, KAF-Ground {r.kaf_ground_misses}, "
          f"recovered {r.recovered}, lost {r.lost} (stated {a}/{k}/{rec}/{lost})",
          (r.afloc_misses, r.kaf_ground_misses, r.recovered, r.lost) == (a, k, rec, lost))
truth("MS-CXR: McNemar p < 1e-15", mi[MS][mi[MS].category == "all"].mcnemar_p.iloc[0] < 1e-15)
check("PadChest-GR: McNemar p (x1e-6)", 4.6,
      mi[PG][mi[PG].category == "all"].mcnemar_p.iloc[0] * 1e6, 1)
truth("MS-CXR: 57 of the recoveries are pneumothoraces",
      int(mi[MS].set_index("category").loc["Pneumothorax", "recovered"]) == 57)

# ------------------------------------------------------------ per finding type
for name, d, stage, stated in [("MS-CXR BEP", MS, "BEP", 6), ("MS-CXR full", MS, "+RQP+DHS", 7)]:
    cats = sorted(pc[d].category.unique())
    won = 0
    for c in cats:
        mine = cat(d, KAF, stage, c)
        others = [cat(d, m, stage if (m != MK or stage == "BEP") else "+DHS", c) for m in (AF, GL, MK)]
        won += all(mine > o for o in others)
    truth(f"{name}: KAF-Ground beats every competitor on {won}/8 (stated {stated})", won == stated)
for c, stated in [("Pneumothorax", 0.0545), ("Edema", 0.0310), ("Consolidation", 0.0308)]:
    check(f"MS-CXR BEP gain over AFLoc, {c}", stated, cat(MS, KAF, "BEP", c) - cat(MS, AF, "BEP", c))
check("PadChest-GR pleural thickening, AFLoc", 0.060, cat(PG, AF, "BEP", "pleural thickening"), 3)
check("PadChest-GR pleural thickening, KAF-Ground", 0.219, cat(PG, KAF, "BEP", "pleural thickening"), 3)
truth("MS-CXR: AFLoc keeps cardiomegaly under BEP",
      cat(MS, AF, "BEP", "Cardiomegaly") > cat(MS, KAF, "BEP", "Cardiomegaly"))

# ----------------------------------------------------------------- report
bad = [r for r in ROWS if not r[0]]
for ok, claim, stated, actual in ROWS:
    tag = "ok  " if ok else "FAIL"
    if stated is None:
        print(f"{tag}  {claim}")
    else:
        print(f"{tag}  {claim:<52} stated {stated:<9} recomputed {actual:.6f}")
print(f"\n{len(ROWS) - len(bad)}/{len(ROWS)} checks pass")
if bad:
    raise SystemExit(f"{len(bad)} checks failed")
