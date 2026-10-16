# -*- coding: utf-8 -*-
"""Several training runs of run_side.py from the same initial weights, two
groups (the runs of two codes), compared pair by pair:
whether the two codes differ by more than the GPU's own non-determinism.

    python tests/equivalence/compare_runs.py --run ref_a=<dir> --run ref_b=<dir> \\
        --run this_a=<dir> --run this_b=<dir> --group ref_a,ref_b --group this_a,this_b [--json <out>]

For every pair of runs: the earliest step at which a loss term differs (the
step, how many terms differ there and the one that differs most), the median
and the largest |difference| of the total loss over steps 1.. (step 0 is a
pure forward from the same weights), and, when both folders hold final.pt, the
largest difference of the parameters after the steps.

The rule, registered before the runs (2026-10-13): W, the
larger of the within-group medians of the total-loss difference; every
cross-group pair's median at most 3 W: no systematic difference; every one
above 3 W: a systematic difference on the GPU; otherwise undecided.  Exit
status 0 for "no systematic difference", 1 otherwise.
"""
import argparse
import itertools
import json
import os
import statistics
import sys

import torch

RATIO = 3.0


def load(d, name):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return None
    if name.endswith(".json"):
        with open(p) as f:
            return json.load(f)
    return torch.load(p, map_location="cpu", weights_only=False)


def first_difference(x, y):
    """(step, terms that differ there, the term that differs most, its two values)
    of the earliest step with a difference; None if none."""
    for i, (p, q) in enumerate(zip(x, y)):
        keys = sorted(set(p) | set(q))
        diff = [k for k in keys if p.get(k) != q.get(k)]
        if diff:
            def size(k):
                a, b = p.get(k), q.get(k)
                return float("inf") if a is None or b is None else abs(a - b)
            k = max(diff, key=size)
            return i, len(diff), k, p.get(k), q.get(k)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="NAME=<run_side.py --out folder>")
    ap.add_argument("--group", action="append", required=True, help="NAME,NAME: the runs of one code")
    ap.add_argument("--json", default="", help="write the numbers here")
    a = ap.parse_args()
    runs = dict(r.split("=", 1) for r in a.run)
    groups = [g.split(",") for g in a.group]
    if len(groups) != 2 or any(len(g) < 2 for g in groups) or any(n not in runs for g in groups for n in g):
        raise SystemExit("two --group of at least two --run names each")
    steps = {n: load(d, "steps.json") for n, d in runs.items()}
    finals = {n: load(d, "final.pt") for n, d in runs.items()}
    lengths = {len(s) for s in steps.values()}
    if len(lengths) != 1:
        raise SystemExit("the runs have different numbers of steps: %s" % {n: len(s) for n, s in steps.items()})
    n_steps = lengths.pop()

    res = {}
    print("%-17s %-34s %11s %11s %11s" % ("pair", "first difference (step, terms, largest)", "median",
                                           "max", "parameters"))
    for x, y in itertools.combinations(sorted(runs), 2):
        d = [abs(p["total"] - q["total"]) for p, q in zip(steps[x][1:], steps[y][1:])]
        fd = first_difference(steps[x], steps[y])
        par = None
        if finals[x] is not None and finals[y] is not None:
            par = max(float((finals[x][k].float() - finals[y][k].float()).abs().max())
                      for k in finals[x] if finals[x][k].numel())
        same = any(x in g and y in g for g in groups)
        res["%s|%s" % (x, y)] = {"within": same, "median": statistics.median(d) if d else 0.0,
                                 "max": max(d, default=0.0), "parameters": par,
                                 "first_difference": None if fd is None else
                                 {"step": fd[0], "terms": fd[1], "term": fd[2], "values": [fd[3], fd[4]]}}
        print("%-17s %-34s %11.4e %11.4e %11s" % (
            "%s|%s" % (x, y), "none" if fd is None else "step %d, %d terms, %s" % fd[:3],
            res["%s|%s" % (x, y)]["median"], res["%s|%s" % (x, y)]["max"],
            "-" if par is None else "%.4e" % par))

    within = [v for v in res.values() if v["within"]]
    cross = {k: v for k, v in res.items() if not v["within"]}
    W = max(v["median"] for v in within)
    above = [k for k, v in cross.items() if v["median"] > RATIO * W]
    if not above:
        verdict = "no systematic difference"
    elif len(above) == len(cross):
        verdict = "systematic difference on the GPU"
    else:
        verdict = "undecided"
    print("\nsteps 1-%d, total loss: W = %.4e (the larger within-group median), 3 W = %.4e" % (
        n_steps - 1, W, RATIO * W))
    for k, v in cross.items():
        print("   %-17s median %.4e = %.2f W  %s" % (k, v["median"], v["median"] / W if W > 0 else float("inf"),
                                                     "above 3 W" if k in above else ""))
    pw = [v["parameters"] for v in within if v["parameters"] is not None]
    pc = [v["parameters"] for v in cross.values() if v["parameters"] is not None]
    if pw and pc:
        print("parameters after the steps: within groups at most %.4e, across at most %.4e (%.2f times)"
              % (max(pw), max(pc), max(pc) / max(pw) if max(pw) > 0 else float("inf")))
    print("verdict: %s" % verdict)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"pairs": res, "W": W, "cross_above_3W": above, "verdict": verdict}, f, indent=1)
    sys.exit(0 if verdict == "no systematic difference" else 1)


if __name__ == "__main__":
    main()
