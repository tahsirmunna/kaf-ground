"""
The paper's quantitative figures, both benchmarks in every one of them.

    fig_overall      : every system across the whole evaluation protocol --
                       AEP, +RQP, +DHS alone, +RQP+DHS -- on all four metrics
                       (mIoU, Dice, CNR, P@1), MS-CXR above and PadChest-GR
                       below, so the transfer result is read off the same axes
                       as the main one.
    fig_findings     : THE SECOND FIGURE THE PAPER PRINTS -- (a) per-finding
                       IoU beside (b) complete misses (IoU = 0), each block
                       holding both datasets, one above the rule and one below.
    fig_results      : all three blocks side by side, for a wider slot than a
                       12.2cm column -- a rotated page, or a talk.
    fig_percategory, fig_recovery_counts : the two blocks of fig_findings on
                       their own, kept for reference and for slides.

DATA
====
Everything is read from results/<dataset>/tables (overall.csv,
per_category.csv) and, for the complete misses, from the per-sample rows in
results/<dataset>/persample, all written by summarize_results.py. "AEP" in the
figures is the stage called BEP in the tables (no inference rule).

AXIS NAMES
==========
Findings are set at 45 degrees under the bars, short form, with a key to the
full names in the figure's own footer: a full name at 45 degrees is longer
than the gap between two bar groups, and set vertically it costs a band under
every row that is deeper than the key.

SIZE
====
Each figure is built at the width it is printed at, so
\\includegraphics[width=\\textwidth] scales it by 1.0 and the point sizes below
are the point sizes on the page.

    python scripts/make_figures.py  -> figures/regenerated/fig_overall.{pdf,png}
                                       figures/regenerated/fig_findings.{pdf,png}
                                       figures/regenerated/fig_results.{pdf,png}
                                       figures/regenerated/fig_percategory.{pdf,png}
                                       figures/regenerated/fig_recovery_counts.{pdf,png}
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "regenerated"
OUT.mkdir(parents=True, exist_ok=True)

PAGE_W = 4.80                      # LNCS \textwidth, inches
INK, MUT, GRID, SURFACE = "#141414", "#6f6f68", "#e3e2de", "#fcfcfb"

# ---- the two benchmarks, in the order the paper reports them ----------------
DATASETS = [
    ("MS-CXR", ROOT / "results" / "ms-cxr" / "tables", 1162,
     ROOT / "results" / "ms-cxr" / "persample"),
    ("PadChest-GR", ROOT / "results" / "padchest-gr" / "tables", 530,
     ROOT / "results" / "padchest-gr" / "persample"),
]

# ---- registry name -> paper name, and the colour fixed per system -----------
# Order is the paper's table order: weakest first, ours last.
PAPER = [("MedKLIP",    "MedKLIP",           "#9151c9"),
         ("GLoRIA",     "GLoRIA",            "#3d9970"),
         ("AFLoc",      "AFLoc",             "#2a78d6"),
         ("KAF-Ground", "KAF-Ground (ours)", "#eb6834")]
NAME = {r: p for r, p, _ in PAPER}
COLOR = {p: c for _, p, c in PAPER}
OURS, RUNNER = "KAF-Ground (ours)", "AFLoc"

# ---- the protocol, read cumulatively ----------------------------------------
# "+DHS" is the control cell: DHS without RQP. It is not a rung of the ladder,
# which is why it is drawn between them -- it says how much of the last column
# is DHS alone, for every system including ours.
# the tick labels are staggered onto two lines: at this panel width four
# stage names in a row touch, and rotating them costs more room than it saves
STAGES = [("RQP-off DHS-off", "AEP"),
          ("RQP-on DHS-off",  "\n+RQP"),
          ("RQP-off DHS-on",  "+DHS"),
          ("RQP-on DHS-on",   "\n+RQP+DHS")]
AEP_STAGE = "RQP-off DHS-off"     # the AFLoc Evaluation Protocol
FULL = "RQP-on DHS-on"            # the full protocol
FULL_NOKG = "RQP-off DHS-on"      # ... for a system with no phrase input

def shades(base, n=4):
    """One hue per system, one shade per protocol condition: light at AEP,
    full strength at the end, so a bar says both at once."""
    import matplotlib.colors as mc
    rgb = np.array(mc.to_rgb(base))
    return [tuple(1 - (1 - rgb) * a) for a in np.linspace(0.40, 1.0, n)]


METRICS = [("mIoU", "mIoU"), ("Dice", "Dice"), ("CNR", "CNR"),
           ("Point", "P@1")]

# ---- finding names: the full name on the axis, the short one where a panel
# ---- is too narrow to carry it --------------------------------------------
def full_name(c):
    """The registries spell MS-CXR findings in title case and PadChest-GR ones
    in lower case. Sentence case for both, so the two panels read alike."""
    return c[0].upper() + c[1:].lower()



ABBR = {"Atelectasis": "Atel.", "Cardiomegaly": "Card.",
        "Consolidation": "Cons.", "Edema": "Edema", "Lung Opacity": "L.Op.",
        "Pleural Effusion": "Pl.Eff.", "Pneumonia": "Pneum.",
        "Pneumothorax": "Ptx.",
        "alveolar pattern": "Alv.pat.", "aortic elongation": "Ao.elong.",
        "atelectasis": "Atel.", "cardiomegaly": "Card.",
        "interstitial pattern": "Int.pat.", "nodule": "Nodule",
        "pleural effusion": "Pl.Eff.", "pleural thickening": "Pl.Thick."}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 6,
        "axes.edgecolor": GRID, "axes.labelcolor": MUT,
        "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "pdf.fonttype": 42,
    })


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


STAGE_KEY = {"BEP": "RQP-off DHS-off", "+RQP": "RQP-on DHS-off",
             "+DHS": "RQP-off DHS-on", "+RQP+DHS": "RQP-on DHS-on"}


def _tables(path):
    d = pd.read_csv(path)
    d = d[d.model.isin(NAME)].copy()
    d["model"] = d.model.map(NAME)
    d["stage"] = d.stage.map(STAGE_KEY)
    return d.rename(columns={"P@1": "Point"})


def overall(tab):
    return _tables(tab / "overall.csv")


def cell(ov, model, stage, metric):
    r = ov[(ov.model == model) & (ov.stage == stage)]
    return float(r[metric].iloc[0]) if len(r) else np.nan


def final_stage(ov, model):
    """A system with no phrase input never defines an RQP cell, so its full
    protocol is DHS alone -- that is a missing lever, not a missing result."""
    return FULL if len(ov[(ov.model == model) & (ov.stage == FULL)]) else FULL_NOKG


# ------------------------------------------------------------- the blocks ---
# Each block draws into a GridSpec the caller has already placed, so the same
# code makes the standalone figure and one panel of a combined one. Only the
# assembly below decides how many blocks share an image.

def draw_overall(fig, slots):
    """Every system across the protocol, four metrics x two datasets."""
    for r, ((dset, tab, n, _), row) in enumerate(zip(DATASETS, slots)):
        ov = overall(tab)
        for c, ((metric, title), slot) in enumerate(zip(METRICS, row)):
            ax = fig.add_subplot(slot)
            for model in COLOR:
                xs, ys = [], []
                for xi, (stage, _) in enumerate(STAGES):
                    v = cell(ov, model, stage, metric)
                    if not np.isnan(v):
                        xs.append(xi); ys.append(v)
                ax.plot(xs, ys, "o", color=COLOR[model], ms=2.4,
                        markeredgecolor=SURFACE, markeredgewidth=0.4, zorder=3)
                # a segment spanning a condition the model does not define is
                # dotted: MedKLIP has no phrase input, so a solid line there
                # would read as a measurement that RQP did nothing
                pts = list(zip(xs, ys))
                for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                    ax.plot([x0, x1], [y0, y1], color=COLOR[model], lw=0.9,
                            ls="-" if x1 - x0 == 1 else ":",
                            alpha=1.0 if x1 - x0 == 1 else 0.55, zorder=2)
            # no annotation on the panel: the levels are Table 1 and the
            # margins are in the text, so the axes are framed tight
            ylo, yhi = ax.get_ylim()
            ax.set_ylim(ylo, yhi + 0.06 * (yhi - ylo))
            ax.set_xlim(-0.35, len(STAGES) - 0.65)
            ax.set_xticks(range(len(STAGES)))
            ax.set_xticklabels([s[1] for s in STAGES], fontsize=4.3)
            ax.tick_params(axis="y", labelsize=5.0, pad=1.2)
            ax.tick_params(axis="x", pad=1.2)
            if r == 0:
                ax.set_title(title, fontsize=6.8, color=INK, pad=2.5)
            if c == 0:
                ax.set_ylabel(f"{dset}\ncategory-weighted", fontsize=5.8,
                              color=INK, labelpad=2)
            tidy(ax)


def draw_percat_steps(fig, slots, titles=None, abbr=True, rot=45,
                      ylab_dset=False, ms=1.7, lw=0.55, fs=4.4,
                      label_full=False):
    """Per-finding IoU for every system at every step of the protocol.

    Three things have to show at once -- finding, system, step -- and a bar can
    carry two. So each system gets a thin strip per finding, with a dot for
    each step on it: colour is the system, shade is the step (palest at AEP),
    and the length of the strip is what the protocol is worth on that finding.
    No stars: the point here is the gain, not who wins a single condition.
    """
    for k, ((dset, tab, n, _), slot) in enumerate(zip(DATASETS, slots)):
        pc = _tables(tab / "per_category.csv")
        cats = sorted(dict.fromkeys(pc.category), key=lambda c: ABBR[c].lower())
        ax = fig.add_subplot(slot)
        w = 0.80 / len(COLOR)
        for i, model in enumerate(COLOR):
            tint = shades(COLOR[model])
            for j, c in enumerate(cats):
                x = j + (i - (len(COLOR) - 1) / 2) * w
                pts = []
                for si, (stage, _) in enumerate(STAGES):
                    q = pc[(pc.model == model) & (pc.stage == stage)
                           & (pc.category == c)]
                    if len(q):
                        pts.append((si, float(q.IoU.iloc[0])))
                if not pts:
                    continue
                ys = [v for _, v in pts]
                ax.plot([x, x], [min(ys), max(ys)], color=COLOR[model],
                        lw=lw, alpha=0.5, zorder=2, solid_capstyle="round")
                for si, v in pts:
                    ax.plot(x, v, "o", ms=ms, color=tint[si], zorder=3,
                            markeredgecolor=SURFACE, markeredgewidth=0.15)
                if label_full:
                    # the value the protocol ends on, over the strip
                    ax.annotate(f"{pts[-1][1]:.2f}".lstrip("0"),
                                (x, max(ys)), xytext=(0, 2.2),
                                textcoords="offset points", rotation=90,
                                ha="center", va="bottom", fontsize=3.6,
                                color=COLOR[model])
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([ABBR[c] if abbr else full_name(c) for c in cats],
                           rotation=rot, ha="center" if rot == 90 else "right",
                           fontsize=fs)
        ax.tick_params(axis="x", pad=1.0)
        ax.tick_params(axis="y", labelsize=fs + 0.4, pad=1.2)
        if titles:
            ax.set_title(titles[k], fontsize=fs + 1.5, color=INK, pad=3)
        ax.set_ylabel(f"{dset}\nIoU" if ylab_dset else "IoU",
                      fontsize=fs + 1.2, color=INK, labelpad=2)
        tidy(ax)


def draw_percat(fig, slots, title=None, titles=None, abbr=True, rot=45,
                ylab_dset=False, stage=AEP_STAGE):
    """Per-finding IoU, every system at its own full protocol."""
    for k, ((dset, tab, n, _), slot) in enumerate(zip(DATASETS, slots)):
        pc = _tables(tab / "per_category.csv")
        pc = pc[pc.stage == stage]
        cats = list(dict.fromkeys(pc.category))
        cats.sort(key=lambda c: ABBR[c].lower())
        ax = fig.add_subplot(slot)
        w = 0.82 / len(COLOR)
        got = {m: {c: float(pc[(pc.model == m) & (pc.category == c)].IoU.iloc[0])
                   for c in cats} for m in COLOR}
        for i, model in enumerate(COLOR):
            vals = [got[model][c] for c in cats]
            xs = np.arange(len(cats)) + (i - (len(COLOR) - 1) / 2) * w
            ax.bar(xs, vals, width=w * 0.9, color=COLOR[model], label=model,
                   edgecolor=SURFACE, linewidth=0.25)
        # a star marks a finding ours wins against every competitor. Fix the
        # height before the loop: reading get_ylim() inside it lets each star
        # raise the limit for the next, and they walk up the panel.
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.13)
        # the star is judged in the condition being plotted, never carried
        # over from another one
        for j, c in enumerate(cats):
            if all(got[OURS][c] > got[m][c] for m in COLOR if m != OURS):
                ax.scatter(j, ymax * 1.05, marker="*", s=9, color="#d4a017",
                           zorder=5, linewidths=0)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([ABBR[c] if abbr else full_name(c) for c in cats],
                           rotation=rot, ha="center" if rot == 90 else "right",
                           fontsize=4.4)
        ax.tick_params(axis="x", pad=1.0)
        ax.tick_params(axis="y", labelsize=4.8, pad=1.2)
        head = titles[k] if titles else (
            title.format(dset=dset, n=f"{n:,}") if title else None)
        if head:
            ax.set_title(head, fontsize=5.9, color=INK, pad=3)
        ax.set_ylabel(f"{dset}\nIoU" if ylab_dset else "IoU",
                      fontsize=5.6, color=INK, labelpad=2)
        tidy(ax)


# A complete miss is a phrase whose returned region does not touch the referent
# at all (IoU = 0), counted on the pairs both systems score, both at their full
# protocol, from the same per-sample rows the tables average.

def misses(ps, variant="orig", calib="off"):
    rqp = "on" if variant == "rgens" else "off"
    b = pd.read_csv(ps / f"afloc_rqp-{rqp}.csv")
    o = pd.read_csv(ps / f"kaf-ground_rqp-{rqp}.csv")
    # paired on (pair, finding), the annotation rows the tables average over:
    # two MS-CXR pairs carry two findings each and count inside both
    m = b[b.dhs == calib].merge(o[o.dhs == calib], on=["pair_id", "category"],
                                suffixes=("_b", "_o"))
    bz, oz = (m.iou_b == 0).values, (m.iou_o == 0).values
    per = pd.DataFrame({"cat": m.category.values, "b": bz, "o": oz})
    g = per.groupby("cat")[["b", "o"]].sum()
    g = g[(g.b > 0) | (g.o > 0)]
    return dict(per_cat=g, afloc=int(bz.sum()), ours=int(oz.sum()),
                recovered=int((bz & ~oz).sum()), lost=int((~bz & oz).sum()),
                n=len(m))


def mcnemar_p(b, c):
    """Exact two-sided McNemar: the discordant pairs under p=1/2."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) * 0.5 ** n
    return min(1.0, 2 * tail)


def ptex(p):
    """The exact p, as the text writes it."""
    if p < 1e-15:
        return "$p<10^{-15}$"
    mant, exp = f"{p:.1e}".split("e")
    return f"$p={mant}\\times 10^{{{int(exp)}}}$"


def draw_misses_steps(fig, slots, titles=None):
    """Complete misses for both systems at every step, read like (a)."""
    for k, ((dset, tab, n, ps), slot) in enumerate(zip(DATASETS, slots)):
        ax = fig.add_subplot(slot)
        xs = list(range(len(STAGES)))
        for model, who in [(RUNNER, "afloc"), (OURS, "ours")]:
            ys = [misses(ps, v, c)[who] for v, c in
                  [("orig", "off"), ("rgens", "off"),
                   ("orig", "on"), ("rgens", "on")]]
            ax.plot(xs, ys, "-o", color=COLOR[model], lw=1.1, ms=2.6,
                    markeredgecolor=SURFACE, markeredgewidth=0.4)
            for x, y in zip(xs, ys):
                ax.annotate(str(y), (x, y), xytext=(0, 2.6),
                            textcoords="offset points", ha="center",
                            fontsize=4.0, color=COLOR[model])
        ax.set_ylim(0, ax.get_ylim()[1] * 1.26)
        ax.set_xlim(-0.35, len(STAGES) - 0.65)
        ax.set_xticks(xs)
        ax.set_xticklabels([st[1] for st in STAGES], fontsize=4.2)
        ax.tick_params(axis="y", labelsize=4.8, pad=1.2)
        if titles:
            ax.set_title(titles[k], fontsize=5.9, color=INK, pad=3)
        ax.set_ylabel("complete misses", fontsize=5.6, color=INK, labelpad=2)
        tidy(ax)


def draw_recovery(fig, slots, title=None, titles=None, abbr=True, rot=45):
    """Complete misses per finding and in total, AFLoc against KAF-Ground."""
    for k, ((dset, tab, n, ps), slot) in enumerate(zip(DATASETS, slots)):
        m = misses(ps)
        g = m["per_cat"]
        cats = [ABBR[c] if abbr else full_name(c) for c in g.index] + ["All"]
        vb = list(g.b) + [m["afloc"]]
        vo = list(g.o) + [m["ours"]]
        ax = fig.add_subplot(slot)
        x = np.arange(len(cats))
        for i, (vals, model) in enumerate([(vb, RUNNER), (vo, OURS)]):
            ax.bar(x + (i - 0.5) * 0.40, vals, width=0.36, color=COLOR[model],
                   edgecolor=SURFACE, linewidth=0.25, label=model)
            for xi, v in zip(x + (i - 0.5) * 0.40, vals):
                ax.annotate(str(int(v)), (xi, v), xytext=(0, 1.0),
                            textcoords="offset points", ha="center",
                            va="bottom", fontsize=4.0, color=INK)
        ax.axvline(len(cats) - 1.5, color=GRID, lw=0.7)   # totals sit apart
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(0, yhi * 1.42)
        # two lines: in the narrow block of fig_results one line runs off the
        # panel, and the label has to read the same in every figure
        ax.annotate(f"{m['recovered']} recovered, {m['lost']} lost\n"
                    + "McNemar " + ptex(mcnemar_p(m["recovered"], m["lost"])),
                    xy=(0.5, 0.995), xycoords="axes fraction",
                    ha="center", va="top", fontsize=4.6, color=INK,
                    linespacing=1.3)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=rot,
                           ha="center" if rot == 90 else "right", fontsize=4.4)
        ax.tick_params(axis="x", pad=1.0)
        ax.tick_params(axis="y", labelsize=4.8, pad=1.2)
        head = titles[k] if titles else (
            title.format(dset=dset, n=f"{m['n']:,}") if title else None)
        if head:
            ax.set_title(head, fontsize=5.9, color=INK, pad=3)
        ax.set_ylabel("complete misses", fontsize=5.6, color=INK, labelpad=2)
        tidy(ax)


def row_rule(fig, upper, lower, x0=0.015, x1=0.988, **kw):
    """A rule across the figure between the two dataset rows.

    Placed by measurement, not by guess: the upper row's tick labels are
    rotated, so where they end is only known once they are drawn. The line
    goes midway between their lowest point and the top of the row below.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    low = min(t.get_window_extent(rend).y0
              for ax in upper for t in ax.get_xticklabels())
    low = inv.transform((0, low))[1]
    high = max(ax.get_position().y1 for ax in lower)
    x1 = kw.get('x1', x1)
    fig.add_artist(Line2D([x0, x1], [(low + high) / 2] * 2, lw=0.5,
                          color=MUT, alpha=0.45, transform=fig.transFigure,
                          zorder=0))


def col_rule(fig, left, right, pad=0.04):
    """A rule between two blocks, in the gap their panels leave.

    Measured off the TIGHT boxes, which include tick labels and the y label,
    so the rule lands in real white space and never crosses a number.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def tight(axs, side):
        xs = [inv.transform(a.get_tightbbox(rend).corners())[:, 0] for a in axs]
        return (max(x.max() for x in xs) if side == "r"
                else min(x.min() for x in xs))

    lb = [a.get_position() for a in left]
    rb = [a.get_position() for a in right]
    x = (tight(left, "r") + tight(right, "l")) / 2
    ys = [b.y0 for b in lb + rb], [b.y1 for b in lb + rb]
    y0, y1 = min(ys[0]), max(ys[1])
    h = (y1 - y0) * pad
    fig.add_artist(Line2D([x, x], [y0 - h, y1 + h], lw=0.5, color=MUT,
                          alpha=0.45, transform=fig.transFigure, zorder=0))


def block_label(fig, axs, text, dy=0.15, fs=6.2):
    """A block's label, centred over the panels it covers.

    Every block gets the same call, so (a), (b) and (c) sit on one line in the
    same size and weight whatever shape the block underneath them is.
    """
    boxes = [a.get_position() for a in axs]
    x = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2
    y = max(b.y1 for b in boxes) + dy / fig.get_size_inches()[1]
    fig.text(x, y, text, ha="center", va="bottom", fontsize=fs, color=INK)


# What the bar blocks are scored at. The line panels walk the whole protocol,
# so only the bars need saying: they are one condition, and which one matters.
FULL_NOTE = ("score every system under the AFLoc Evaluation Protocol "
             "(AEP): the raw model, before RQP or DHS")


def protocol_note(fig, y, text, fs=4.6):
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=fs, color=INK)


def _width_in(fig, text, fs):
    """What a string will actually measure, in inches -- asked of the renderer
    rather than guessed from a nominal glyph width, which is wrong for this
    font by a third."""
    t = fig.text(0, 0, text, fontsize=fs)
    fig.canvas.draw()
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def abbr_key(fig, y, fs=4.0, x=0.013):
    """Short name -> full name, flush left, each line filling the width.

    Set at 45 degrees a short name fits between two bar groups where a full
    one does not; this footer is what the short one costs. Entries are packed
    left to right by measured width, so a line starts at the left margin and
    runs to the right edge instead of sitting centred with a ragged tail.
    """
    pairs = sorted({(v, k.lower()) for k, v in ABBR.items()
                    if v.rstrip(".").lower() != k.lower()},
                   key=lambda t: t[0].lower())
    room = fig.get_size_inches()[0] * (1 - 2 * x)
    entries = [f"{a} {full}" for a, full in pairs]
    lines, cur = [], []
    for e in entries:
        if cur and _width_in(fig, "   ".join(cur + [e]), fs) > room:
            lines.append(cur)
            cur = [e]
        else:
            cur.append(e)
    lines.append(cur)
    # a last line holding one entry reads as an orphan; borrow from the line
    # above, which still leaves every line flush left
    if len(lines) > 1 and len(lines[-1]) == 1 and len(lines[-2]) > 2:
        lines[-1].insert(0, lines[-2].pop())
    lines = ["   ".join(l) for l in lines]
    fig.text(x, y, "\n".join(lines), ha="left", va="bottom", fontsize=fs,
             color=MUT, linespacing=1.45)


def legend(fig, y, models=None, star=False, fs=5.4):
    models = models or list(COLOR)
    h = [Patch(facecolor=COLOR[m], label=m) for m in models]
    if star:
        h.append(Line2D([], [], marker="*", ls="", ms=4.5, color="#d4a017",
                        label="KAF-Ground beats every competitor"))
    fig.legend(handles=h, loc="lower center", ncol=len(h), frameon=False,
               fontsize=fs, bbox_to_anchor=(0.5, y), handlelength=1.2,
               columnspacing=1.3, handletextpad=0.4)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=400)
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf  ({w:.2f} x {h:.2f} in)")


# ------------------------------------------------------- one block per image -
def fig_overall():
    H = 2.30
    fig = plt.figure(figsize=(PAGE_W, H))
    # the right margin is what the last stage label needs: it is centred on
    # the final point, which sits near the axes edge. Both rows share the
    # stage axis, so its labels are printed once, under the bottom row --
    # that is what lets the rows sit this close
    gs = fig.add_gridspec(2, 4, left=0.118, right=0.972, top=1 - 0.31 / H,
                          bottom=0.40 / H, wspace=0.30, hspace=0.16)
    fig.text(0.5, 1 - 0.02 / H, "All four metrics across the evaluation protocol, "
             "both benchmarks", ha="center", va="top", fontsize=7.2, color=INK)
    draw_overall(fig, [[gs[r, c] for c in range(4)] for r in range(2)])
    for ax in fig.axes[:4]:
        ax.tick_params(axis="x", labelbottom=False)
    handles = [Line2D([], [], marker="o", ls="-", ms=3, lw=1.1,
                      color=COLOR[m], label=m) for m in COLOR]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=6.0, bbox_to_anchor=(0.5, -0.01), handlelength=1.5,
               columnspacing=1.8, handletextpad=0.4)
    save(fig, "fig_overall")


def fig_percategory():
    """Block (b) of fig_results on its own, enlarged.

    Same strips -- colour the system, shade the step, length what the protocol
    is worth -- but at twice the width, with the full-protocol value written
    over each strip, for reading a finding off the page.
    """
    # bottom band, in inches from the base: key, note, legend, each clear of
    # the next
    W, pan, xlab = 7.60, 1.55, 0.44
    key_y, note_y, leg_y, band = 0.03, 0.30, 0.46, 0.68
    H = 0.20 + pan + xlab + pan + xlab + band + 0.04
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, 1, left=0.072, right=0.995, top=1 - 0.20 / H,
                          bottom=(xlab + band) / H, hspace=xlab / pan)
    n0 = len(fig.axes)
    draw_percat_steps(fig, [gs[0, 0], gs[1, 0]], ylab_dset=True, ms=3.4,
                      lw=1.0, fs=6.0, label_full=True)
    row_rule(fig, [fig.axes[n0]], [fig.axes[n0 + 1]])
    grey = shades("#5b5b58")
    h = [Line2D([], [], marker="o", ls="", ms=4, color=COLOR[m], label=m)
         for m in COLOR]
    h += [Patch(facecolor=grey[i], label=lab.strip())
          for i, (_, lab) in enumerate(STAGES)]
    fig.legend(handles=h, loc="lower center", ncol=8, frameon=False,
               fontsize=6.0, bbox_to_anchor=(0.5, leg_y / H),
               handlelength=1.2, columnspacing=1.3, handletextpad=0.4)
    fig.text(0.5, note_y / H, "each strip is one system on one finding: a dot "
             "per step, palest at AEP; the number is the full protocol",
             ha="center", va="bottom", fontsize=5.0, color=INK)
    abbr_key(fig, key_y / H, fs=4.6)
    save(fig, "fig_percategory")


def fig_recovery():
    fig = plt.figure(figsize=(PAGE_W, 2.20))
    gs = fig.add_gridspec(1, 2, left=0.075, right=0.995, top=0.865,
                          bottom=0.350, wspace=0.17)
    draw_recovery(fig, [gs[0, 0], gs[0, 1]], title="{dset}  ({n} pairs)")
    legend(fig, 0.155, models=[RUNNER, OURS])
    protocol_note(fig, 0.092, "Both systems under AEP: the raw model, "
                  "before RQP or DHS")
    abbr_key(fig, 0.012)
    save(fig, "fig_recovery_counts")


# --------------------------------------------- blocks that share one image --
# Side by side, never stacked: a block keeps its two datasets one above the
# other inside its own column, so the columns are what the reader compares.
# Full finding names on the axis, set vertically -- at this width a rotated
# name collides with its neighbour, and a vertical one never can.

def fig_findings():
    """Per-finding IoU beside complete misses, both datasets in each."""
    pan, xlab, leg, key, note = 1.05, 0.34, 0.26, 0.22, 0.13
    H = 0.17 + pan + xlab + pan + (xlab + leg + key + note) + 0.03
    fig = plt.figure(figsize=(PAGE_W, H))
    # the dataset rides in the row's y label, so nothing has to be written
    # between the two rows and they sit as close as their tick labels allow
    gs = fig.add_gridspec(2, 2, left=0.115, right=0.995, top=1 - 0.17 / H,
                          bottom=(xlab + leg + key + note) / H,
                          wspace=0.24, hspace=xlab / pan)
    n0 = len(fig.axes)
    draw_percat(fig, [gs[0, 0], gs[1, 0]], ylab_dset=True)
    n1 = len(fig.axes)
    draw_recovery(fig, [gs[0, 1], gs[1, 1]])
    block_label(fig, fig.axes[n0:n1], "(a) Per-finding IoU", dy=0.03)
    block_label(fig, fig.axes[n1:], "(b) Complete misses", dy=0.03)
    row_rule(fig, [fig.axes[n0], fig.axes[n1]],
             [fig.axes[n0 + 1], fig.axes[n1 + 1]])
    legend(fig, (key + note + 0.02) / H, star=True, fs=5.2)
    protocol_note(fig, (key + 0.01) / H, "Both blocks " + FULL_NOTE)
    abbr_key(fig, 0.012)
    save(fig, "fig_findings")


def fig_results():
    """Every system across the protocol, three ways, side by side.

    (a) the four metrics, (b) every finding, (c) the complete misses -- all of
    them stepping AEP -> +RQP -> +DHS -> +RQP+DHS, so one figure answers what
    each step is worth to each system. Built for a wider slot than a 12.2cm
    column: a rotated page, or a talk.
    """
    # the bottom band, in inches from the base: key, then the note, then the
    # legend -- each needs its own line or they land on one another
    W, pan, xlab, col_t = 7.60, 1.15, 0.40, 0.14
    key_y, note_y, leg_y, band = 0.03, 0.30, 0.44, 0.62
    H = 0.17 + col_t + pan + xlab + pan + xlab + band + 0.04
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, 6, left=0.062, right=0.982,
                          top=1 - (0.17 + col_t) / H,
                          bottom=(xlab + band) / H,
                          wspace=0.40, hspace=xlab / pan,
                          width_ratios=[1, 1, 1, 1, 3.0, 1.35])
    n0 = len(fig.axes)
    draw_overall(fig, [[gs[r, c] for c in range(4)] for r in range(2)])
    n1 = len(fig.axes)
    draw_percat_steps(fig, [gs[0, 4], gs[1, 4]])
    n2 = len(fig.axes)
    draw_misses_steps(fig, [gs[0, 5], gs[1, 5]])
    block_label(fig, fig.axes[n0:n1], "(a) Four metrics")
    block_label(fig, fig.axes[n1:n2], "(b) Per-finding IoU")
    block_label(fig, fig.axes[n2:], "(c) Complete misses")
    row_rule(fig, fig.axes[n0:n0 + 4] + [fig.axes[n1], fig.axes[n2]],
             fig.axes[n0 + 4:n1] + [fig.axes[n1 + 1], fig.axes[n2 + 1]])
    col_rule(fig, fig.axes[n0:n1], fig.axes[n1:n2])
    col_rule(fig, fig.axes[n1:n2], fig.axes[n2:])
    grey = shades("#5b5b58")
    h = [Line2D([], [], marker="o", ls="-", ms=3, lw=1.1, color=COLOR[m],
                label=m) for m in COLOR]
    h += [Patch(facecolor=grey[i], label=lab.strip())
          for i, (_, lab) in enumerate(STAGES)]
    fig.legend(handles=h, loc="lower center", ncol=8, frameon=False,
               fontsize=5.2, bbox_to_anchor=(0.5, leg_y / H),
               handlelength=1.2, columnspacing=1.1, handletextpad=0.4)
    fig.text(0.5, note_y / H, "every block spans AEP $\\rightarrow$ $+$RQP "
             "$\\rightarrow$ $+$DHS $\\rightarrow$ $+$RQP$+$DHS", ha="center",
             va="bottom", fontsize=4.8, color=INK)
    abbr_key(fig, key_y / H, fs=4.2)
    save(fig, "fig_results")


# ============================================ the numbers the text quotes ====
def report():
    """Print every value the results section states, so the prose can be
    checked against the tables without opening them."""
    for dset, tab, n, _ in DATASETS:
        ov = overall(tab)
        print(f"\n== {dset} (n={n}) ==")
        print(f"{'model':<14} {'stage':<16} " +
              " ".join(f"{t:>7}" for _, t in METRICS))
        for model in COLOR:
            for stage, lab in STAGES:
                vals = [cell(ov, model, stage, m) for m, _ in METRICS]
                if all(np.isnan(v) for v in vals):
                    continue
                print(f"{model:<14} {lab.replace(chr(10), ' '):<16} " +
                      " ".join(f"{v:7.4f}" for v in vals))
        print("-- margin of ours over each system at its full protocol --")
        for model in COLOR:
            if model == OURS:
                continue
            d = [cell(ov, OURS, final_stage(ov, OURS), m) -
                 cell(ov, model, final_stage(ov, model), m) for m, _ in METRICS]
            print(f"{'vs ' + model:<14} {'':<16} " +
                  " ".join(f"{v:+7.4f}" for v in d))


if __name__ == "__main__":
    style()
    fig_overall()
    fig_percategory()
    fig_recovery()
    fig_findings()
    fig_results()
    report()
