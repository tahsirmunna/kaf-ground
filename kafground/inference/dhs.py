"""
Dynamic Heatmap Sharpening (DHS).

A fixed threshold suits some maps and not others: a peaked map needs a high
operating point, a diffuse one a gentler one. DHS sets the operating point of
each map from its own intensity distribution, with no ground truth involved:

    u  = min-max normalised map (valid pixels only)
    q  = fraction of pixels above mean(u) + std(u)       (clipped to [1e-3, 0.6])
    b  = the (1 - q) quantile of u
    u' = sigmoid(a * (u - b)),  a = 96

The transform is monotone, so the location of the maximum, and therefore the
pointing game (P@1), does not change. In float the steep sigmoid saturates to
1.0 over a plateau, which makes argmax on the sharpened map ill-defined, so P@1
is always computed on the unsharpened map.
"""
import numpy as np

A = 96.0


def sharpen(hmap, nan, a=A):
    """
    hmap : raw heatmap (any range), NaN outside the evaluated region
    nan  : boolean mask of those pixels
    Returns the sharpened map (NaN outside) and the q that was used.
    """
    v = hmap[~nan]
    rng = float(v.max() - v.min()) if v.size else 0.0
    u = (hmap - v.min()) / rng if rng > 0 else np.zeros_like(hmap)

    flat = u[~nan]
    q = float((flat > flat.mean() + flat.std()).mean()) if flat.size else 0.1
    q_used = float(np.clip(q, 1e-3, 0.6))

    b = float(np.quantile(u[~nan], 1.0 - float(np.clip(q_used, 1e-6, 1.0))))
    z = np.clip(a * (np.nan_to_num(u) - b), -60, 60)
    out = np.where(nan, np.nan, 1.0 / (1.0 + np.exp(-z)))
    return out, q
