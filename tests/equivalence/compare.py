# -*- coding: utf-8 -*-
"""Compare the two sides of run_side.py: the initial state_dicts (keys,
shapes, values), the evaluation outputs, every loss term of every training
step and the final parameters, all bit for bit -- what a CPU run gives.

    python tests/equivalence/compare.py <ref dir> <this dir> [--noise <second run of this>] [--table <csv>]

On a GPU the training steps after the first are not reproducible (run_side.py),
and --noise names a second run of the side 'this' from the same
weights: the initial values and the first step's losses must still agree bit
for bit, and over the later steps the reference side's differences from the
first run must not exceed the second run's by more than RATIO: for the total
loss and for the final parameters, the largest difference of the reference
side at most three times that of the second run (both zero: bit for bit).
Exit status 1 on a failure.
"""
import argparse
import json
import os
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("this")
    ap.add_argument("--noise", default="", help="a second run of the side 'this' (GPU)")
    ap.add_argument("--top", type=int, default=12, help="loss terms listed (--noise), the largest differences first")
    ap.add_argument("--table", default="", help="CSV of every loss term's differences (--noise)")
    a = ap.parse_args()
    fails = []

    def check(name, ok, detail=""):
        print("%-66s %s%s" % (name, "ok" if ok else "FAIL", ("  " + detail) if detail and not ok else ""))
        if not ok:
            fails.append(name)

    ti, ri = load(a.ref, "init.pt"), load(a.this, "init.pt")
    if ti is not None and ri is not None:
        check("state_dict: the same keys", set(ti) == set(ri),
              "only in ref %s, only in this %s" % (sorted(set(ti) - set(ri))[:5], sorted(set(ri) - set(ti))[:5]))
        shared = sorted(set(ti) & set(ri))
        check("state_dict: the same shapes", all(ti[k].shape == ri[k].shape for k in shared),
              str([k for k in shared if ti[k].shape != ri[k].shape][:5]))
        same = [k for k in shared if ti[k].shape == ri[k].shape and torch.equal(ti[k], ri[k])]
        check("state_dict: the same initial values (same seed)", len(same) == len(shared),
              "%d of %d differ, e.g. %s" % (len(shared) - len(same), len(shared),
                                             [k for k in shared if k not in same][:5]))

    to, ro = load(a.ref, "eval_outputs.pt"), load(a.this, "eval_outputs.pt")
    if to is not None and ro is not None:
        bad = []
        for i, (x, y) in enumerate(zip(to, ro)):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y or x[k].shape != y[k].shape or not torch.equal(x[k], y[k]):
                    bad.append("tile %d %s" % (i, k))
        check("evaluation outputs bit for bit (masks, boxes, scores, heights, offsets)",
              len(to) == len(ro) and not bad, str(bad[:5]))

    ts, rs = load(a.ref, "steps.json"), load(a.this, "steps.json")
    if ts is not None and rs is not None:
        n = min(len(ts), len(rs))
        # the denoising terms are missing in a step whose tile has more buildings than
        # MaskDINO.DN_NUM, so the terms are compared step by step
        terms = [sorted(set(x) ^ set(y)) for x, y in zip(ts, rs)]
        check("training: the same loss terms in every step (%d steps, %d to %d terms)"
              % (n, min(map(len, rs), default=0), max(map(len, rs), default=0)),
              len(ts) == len(rs) and not any(terms), str([t for t in terms if t][:2]))
        tf, rf = load(a.ref, "final.pt"), load(a.this, "final.pt")
        if not a.noise:
            diff = ["step %d %s %r vs %r" % (i, k, x[k], y[k]) for i, (x, y) in enumerate(zip(ts, rs))
                    for k in sorted(set(x) & set(y)) if x[k] != y[k]]
            check("training: every loss term of every step bit for bit", not diff, "; ".join(diff[:4]))
            tc, rc = load(a.ref, "checksums.json"), load(a.this, "checksums.json")
            if tc is not None and rc is not None:
                print("   checksums: threads %s / %s, deterministic algorithms %s / %s" % (
                    tc["threads"], rc["threads"], tc["deterministic"], rc["deterministic"]))
                for what in ("grads", "params"):
                    bad = [(i, k) for i, (x, y) in enumerate(zip(tc["steps"], rc["steps"]))
                           for k in sorted(set(x[what]) | set(y[what])) if x[what].get(k) != y[what].get(k)]
                    first = [k for i, k in bad if i == bad[0][0]] if bad else []
                    check("training: %s bit for bit (%d steps, %d tensors)" % (
                        "the gradients of every step" if what == "grads" else "the parameters after every step",
                        len(tc["steps"]),
                        len(tc["steps"][0][what]) if tc["steps"] else 0), not bad,
                          "first at step %d: %d tensors, e.g. %s" % (bad[0][0], len(first), first[:4]) if bad else "")
            if tf is not None and rf is not None:
                bad = [k for k in sorted(set(tf) & set(rf)) if not torch.equal(tf[k], rf[k])]
                check("training: the parameters after the steps bit for bit", set(tf) == set(rf) and not bad,
                      str(bad[:5]))
        else:
            ns, nf = load(a.noise, "steps.json"), load(a.noise, "final.pt")
            first = [k for k in ts[0] if not (ts[0][k] == rs[0].get(k) == ns[0].get(k))] if n else ["no steps"]
            check("training: the first step's loss terms bit for bit (all three runs)", not first, str(first[:5]))

            def largest(x, y, key):
                """The largest |x - y| of a term over steps 1.., where both runs have it, and in how many."""
                d = [abs(p[key] - q[key]) for p, q in zip(x[1:], y[1:]) if key in p and key in q]
                return max(d, default=0.0), len(d)

            rows = [("total loss, steps 1-%d" % (n - 1), largest(ts, rs, "total")[0], largest(ns, rs, "total")[0])]
            if tf is not None and rf is not None and nf is not None:
                pt = max((float((tf[k].float() - rf[k].float()).abs().max()) for k in rf if rf[k].numel()), default=0.0)
                pn = max((float((nf[k].float() - rf[k].float()).abs().max()) for k in rf if rf[k].numel()), default=0.0)
                rows.append(("parameters after the steps", pt, pn))
            else:
                print("(no final.pt in all three folders: the parameters are not compared)")
            for name, x, y in rows:
                ok = x == 0 or (y > 0 and x / y <= RATIO)
                check("training: %s, largest difference ref %.3e vs second run %.3e (ratio %s)"
                      % (name, x, y, "%.2f" % (x / y) if y > 0 else "-"), ok)
            keys = sorted(set().union(*rs[1:])) if n > 1 else []
            per = []
            for k in keys:
                (x, m), (y, _) = largest(ts, rs, k), largest(ns, rs, k)
                per.append((x, y, m, k))
            per.sort(key=lambda r: (-r[0], r[3]))
            over = [r for r in per if r[0] > 0 and not (r[1] > 0 and r[0] / r[1] <= RATIO)]
            print("   per loss term, steps 1-%d: largest |ref - first run|, largest |second run - first run|, "
                  "steps with the term; %d of %d terms beyond %g times the second run's" % (
                      n - 1, len(over), len(per), RATIO))
            for x, y, m, k in per[:a.top]:
                print("     %-40s %.3e  %.3e  %3d" % (k, x, y, m))
            if a.table:
                with open(a.table, "w") as f:
                    f.write("term,steps,ref_vs_first,second_vs_first,ratio\n")
                    for x, y, m, k in per:
                        f.write("%s,%d,%.6e,%.6e,%s\n" % (k, m, x, y, "%.4f" % (x / y) if y > 0 else ""))
                print("   the table of all %d terms: %s" % (len(per), a.table))
    print("PASS" if not fails else "FAIL: %s" % fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
