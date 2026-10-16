# -*- coding: utf-8 -*-
"""The results table of evaluated models, as the paper reports them.

Per model (the evaluate.json of tools/evaluate.py, e.g. <run>/final_val):
F1 and mAP50 of the model as it is (orig), the height MAE and RMSE at the
best-F1 operating point and the MAE per height bin with the flip-averaged
heights (avg; orig without --flip), and the MAE of the BDH protocol (no score
threshold, greedy matching) flip-averaged and, for comparison with published
numbers, unflipped.  With --paired, the paired gains of tools/paired_compare.py
(analysis.json) are added: gain over the comparison's reference with its 95 %
interval.

    python tools/make_table.py --run OPT=output/opt/final_val --run WISH-Net=output/wishnet/final_val \\
        [--paired <paired_compare out>/analysis.json] [--out table.md]
"""
import argparse
import json
import os

BINS = ["[3,10)", "[10,20)", "[20,30)", "[30,50)", "[50,100)", "[100,inf)"]


def fmt(v, nd=2, sign=False):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "-"
    if f != f:
        return "-"
    return ("%+.*f" if sign else "%.*f") % (nd, f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="NAME=<folder with evaluate.json>")
    ap.add_argument("--paired", action="append", default=[], help="analysis.json of tools/paired_compare.py")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    gains = {}
    for p in a.paired:
        with open(p, encoding="utf-8") as f:
            for e in json.load(f)["paired"]:
                gains[e["B"]] = e
    rows = []
    for spec in a.run:
        name, path = spec.split("=", 1)
        with open(os.path.join(path, "evaluate.json"), encoding="utf-8") as f:
            ev = json.load(f)
        o = ev["protocol"]["orig"]
        h = ev["protocol"].get("avg", o)
        g = gains.get(name)
        rows.append([name, fmt(o.get("f1op/F1@0.5")), fmt(o.get("segAP50")), fmt(h.get("f1op/MAE"), 3),
                     fmt(h.get("f1op/RMSE"), 3)]
                    + [fmt(h.get("f1op/MAE/%s" % b), 2) for b in BINS]
                    + [fmt(h.get("bdhnet/MAE"), 3), fmt(o.get("bdhnet/MAE"), 3),
                       "-" if g is None else "%s [%s, %s] vs %s" % (fmt(g["gain"], 3, True), fmt(g["ci95"][0], 3, True),
                                                                    fmt(g["ci95"][1], 3, True), g["A"]),
                       "yes" if "avg" in ev["protocol"] else "no"])
    head = (["model", "F1", "mAP50", "MAE (m)", "RMSE (m)"] + ["MAE %s" % b for b in BINS]
            + ["BDH-protocol MAE", "BDH-protocol MAE, unflipped", "paired gain [95% interval]", "flip-averaged"])
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    text = "\n".join(lines) + "\n"
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
