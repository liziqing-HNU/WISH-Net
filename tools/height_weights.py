# -*- coding: utf-8 -*-
"""The weight table of MODEL.WISHNET.HEIGHT.BIN_WEIGHTS from the training labels.

Per bin of the height head (BIN_MIN_M .. BIN_MAX_M by BIN_STEP_M, floor-aligned:
3, 6, ..., 180 m) the training buildings with a recorded height (bd_height of
at least --min-height) are counted; heights beyond the last bin are left out,
since the criterion gives them weight 1.  The counts are smoothed with a
Gaussian of --sigma-bins bins (LDS, label distribution smoothing; Yang et al.,
Delving into Deep Imbalanced Regression, ICML 2021), raised to the power
-POWER (1 the inverse frequency, 0.5 its square root), scaled so that the mean
weight over the training buildings is 1 and capped at --max (the scale is
solved with the cap in place, so the mean stays 1).

Printed: the table as a YAML list for the configuration, the weights at a few
heights, and every height band's share of the buildings and of the weighted
height loss.

    python tools/height_weights.py --json datasets/bdh/annotations_fixed/instances_train.json \\
        [--sigma-bins 2] [--power 0.25] [--max 3]
    python tools/height_weights.py --counts 1024,2079,... (one count per bin)

The configuration's table (wishnet/config.py, BDH_BIN_WEIGHTS) is this
script's output for annotations_fixed/instances_train.json with the defaults.
"""
import argparse
import json

import numpy as np

BANDS = ((3, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 1e9))


def bin_counts(heights, lo, hi, step):
    """Buildings per head bin; heights beyond the last edge left out."""
    n = int(round((hi - lo) / step)) + 1
    edges = lo - step / 2.0 + step * np.arange(n + 1)
    h = np.asarray(heights, float)
    h = h[h <= edges[-1]]
    k = np.clip(np.searchsorted(edges, h, side="right") - 1, 0, n - 1)
    return np.bincount(k, minlength=n).astype(float)


def weights_from_counts(counts, sigma_bins, power, cap):
    c = np.asarray(counts, float)
    r = int(np.ceil(3 * sigma_bins))
    ker = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_bins) ** 2)
    dens = np.convolve(c, ker / ker.sum(), mode="same")
    w0 = np.maximum(dens, 1e-6) ** (-power)
    lo, hi = 1e-9, 1e9
    for _ in range(300):
        s = np.sqrt(lo * hi)
        if (c * np.minimum(s * w0, cap)).sum() / c.sum() < 1.0:
            lo = s
        else:
            hi = s
    return np.minimum(np.sqrt(lo * hi) * w0, cap)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", help="training annotations (COCO with bd_height)")
    src.add_argument("--counts", help="comma-separated buildings per head bin")
    ap.add_argument("--min-height", type=float, default=3.0)
    ap.add_argument("--bin-min", type=float, default=3.0)
    ap.add_argument("--bin-max", type=float, default=180.0)
    ap.add_argument("--bin-step", type=float, default=3.0)
    ap.add_argument("--sigma-bins", type=float, default=2.0)
    ap.add_argument("--power", type=float, default=0.25)
    ap.add_argument("--max", type=float, default=3.0)
    a = ap.parse_args()
    if a.json:
        anns = json.load(open(a.json))["annotations"]
        h = [float(x.get("bd_height", 0.0)) for x in anns]
        h = [v for v in h if v >= a.min_height]
        c = bin_counts(h, a.bin_min, a.bin_max, a.bin_step)
        print("buildings with a recorded height: %d, beyond the last bin (weight 1): %d"
              % (len(h), len(h) - int(c.sum())))
    else:
        c = np.array([float(v) for v in a.counts.split(",")])
    w = weights_from_counts(c, a.sigma_bins, a.power, a.max)
    centres = a.bin_min + a.bin_step * np.arange(len(c))
    print("sigma %.2f bins, power %.2f, cap %.2f: mean weight over the %d buildings %.6f"
          % (a.sigma_bins, a.power, a.max, int(c.sum()), (c * w).sum() / c.sum()))
    print("weights at " + ", ".join("%g m %.2f" % (centres[i], w[i]) for i in
                                    np.searchsorted(centres, [3, 9, 15, 21, 30, 45, 60, 75, 90, 105, 150])))
    for lo, hi in BANDS:
        m = (centres >= lo) & (centres < hi)
        print("  %3d-%-4s m: buildings %5.1f %%, weighted height loss %5.1f %%" % (
            lo, "" if hi > 1e8 else "%d" % hi, 100 * c[m].sum() / c.sum(), 100 * (c[m] * w[m]).sum() / (c * w).sum()))
    print("BIN_WEIGHTS: [%s]" % ", ".join("%.4f" % v for v in w))


if __name__ == "__main__":
    main()
