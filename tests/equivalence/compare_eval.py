# -*- coding: utf-8 -*-
"""Compare two evaluations of the same weights: folders of tools/evaluate.py
(orig/, avg/) of two checkouts, or the original implementation's evaluation
(its whole-tile evaluation, inference_bdh_val/, or the orig/ and avg/ folders
of its flip-averaging tool) and this code's: the predictions
(bdh_predictions.json: every detection's image, score, height, offset and
mask) and the metrics (bdh_metrics.json), value for value.

    python tests/equivalence/compare_eval.py <folder A> <folder B>

Exit status 1 on any difference.
"""
import json
import math
import os
import sys


def same(a, b):
    """Equal, NaN equal to NaN."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    da, db = sys.argv[1], sys.argv[2]
    pa = json.load(open(os.path.join(da, "bdh_predictions.json"), encoding="utf-8"))
    pb = json.load(open(os.path.join(db, "bdh_predictions.json"), encoding="utf-8"))
    bad = [i for i, (x, y) in enumerate(zip(pa, pb)) if not same(x, y)]
    ok_p = len(pa) == len(pb) and not bad
    print("predictions: %d and %d detections, %d differ%s -> %s" % (
        len(pa), len(pb), len(bad), (" (first: %s)" % bad[:3]) if bad else "", "ok" if ok_p else "FAIL"))
    ma = json.load(open(os.path.join(da, "bdh_metrics.json"), encoding="utf-8"))
    mb = json.load(open(os.path.join(db, "bdh_metrics.json"), encoding="utf-8"))
    oa, ob = ma.get("overall", ma), mb.get("overall", mb)
    keys = sorted(set(oa) | set(ob))
    diff = [k for k in keys if not same(oa.get(k), ob.get(k))]
    print("metrics: %d values, %d differ%s -> %s" % (len(keys), len(diff), (" (%s)" % diff[:5]) if diff else "",
                                                     "ok" if not diff else "FAIL"))
    for k in ("f1op/score_thresh", "f1op/F1@0.5", "segAP50", "f1op/MAE", "f1op/RMSE", "bdhnet/MAE"):
        print("   %-20s %r  %r" % (k, oa.get(k), ob.get(k)))
    ok = ok_p and not diff
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
