"""
Grounding metrics. `nan` marks pixels outside the evaluated region (the
margin), which are ignored everywhere.

IoU and Dice are computed per threshold on the [-1, 1]-normalised heatmap and
averaged over t in {0.1, ..., 0.5}; samples whose thresholded map is empty get
NaN and are skipped by the category means (as in AFLoc's scorer).
"""
import numpy as np

from .datasets import norm_heatmap

THRESHOLDS = list(np.arange(0.1, 0.6, 0.1))


def compute_iou(gtmask_, premask_, nan, only_pos=True):
    gtmask, premask = gtmask_[~nan], premask_[~nan]
    inter = np.logical_and(gtmask, premask).sum()
    union = np.logical_or(gtmask, premask).sum()
    if only_pos:
        if premask.sum() == 0 or gtmask.sum() == 0:
            return np.nan
    elif union == 0:
        return np.nan
    return inter / union


def compute_dice(gtmask_, premask_, nan, only_pos=True):
    gtmask, premask = gtmask_[~nan], premask_[~nan]
    inter = np.logical_and(gtmask, premask).sum()
    if only_pos:
        if premask.sum() == 0 or gtmask.sum() == 0:
            return np.nan
    elif np.logical_or(gtmask, premask).sum() == 0:
        return np.nan
    return 2 * inter / (premask.sum() + gtmask.sum())


def compute_cnr(gtmask_, heatmap_, nan):
    """Contrast-to-noise ratio |mean_in - mean_out| / sqrt(var_in + var_out)."""
    heatmap = norm_heatmap(heatmap_, nan)
    h, g = heatmap[~nan], gtmask_[~nan]
    a_in, a_out = h[g == 1], h[g == 0]
    var = a_in.var() + a_out.var()
    return 0 if var == 0 else (a_in.mean() - a_out.mean()) / var ** 0.5


def pointing_game(gtmask, heatmap, nan):
    """1 if the arg-max pixel of the map lies inside the ground truth (P@1)."""
    h = np.where(nan, -np.inf, np.nan_to_num(heatmap, nan=-np.inf))
    return float(np.asarray(gtmask).ravel()[int(np.argmax(h))] > 0)


def threshold_mean(metric, gtmask, heatmap, nan, thresholds=THRESHOLDS):
    return float(np.mean([metric(gtmask, np.where(heatmap > t, 1, 0), nan)
                          for t in thresholds]))
