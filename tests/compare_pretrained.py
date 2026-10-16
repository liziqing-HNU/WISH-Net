# -*- coding: utf-8 -*-
"""Compare two detectron2 weight files (.pkl) tensor by tensor: the same
names, and per group (batch-norm running statistics, other tensors) the
largest absolute and relative difference.

    python tests/compare_pretrained.py <a.pkl> <b.pkl>

Used to tell which GF-3 tile folder a ResNet-18 with re-estimated batch-norm
statistics (tools/pretrained/convert_r18_imagenet.py --adabn) was made from:
made again from the right folder the statistics agree to rounding (the windows
are the same, only the GPU's summation order differs), from another folder
they differ by orders of magnitude more.
"""
import pickle
import sys

import numpy as np


def load(p):
    with open(p, "rb") as f:
        ck = pickle.load(f, encoding="latin1")
    return ck.get("model", ck)


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    print("tensors: %d and %d; only in the first %s, only in the second %s" % (
        len(a), len(b), sorted(set(a) - set(b))[:5], sorted(set(b) - set(a))[:5]))
    groups = {"running statistics": lambda k: k.endswith("running_mean") or k.endswith("running_var"),
              "other tensors": lambda k: not (k.endswith("running_mean") or k.endswith("running_var"))}
    for name, sel in groups.items():
        keys = [k for k in sorted(set(a) & set(b)) if sel(k)]
        if not keys:
            continue
        ab, rel, worst = 0.0, 0.0, None
        for k in keys:
            x, y = np.asarray(a[k], np.float64), np.asarray(b[k], np.float64)
            if x.shape != y.shape:
                print("   %s: shapes %s and %s" % (k, x.shape, y.shape))
                continue
            d = np.abs(x - y)
            r = d / np.maximum(np.abs(y), 1e-12)
            if d.size and d.max() > ab:
                ab, worst = float(d.max()), k
            rel = max(rel, float(r.max()) if r.size else 0.0)
        print("%-19s %4d tensors: max absolute difference %.3e, max relative %.3e (largest in %s)"
              % (name, len(keys), ab, rel, worst))


if __name__ == "__main__":
    main()
