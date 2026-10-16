# -*- coding: utf-8 -*-
"""The run's weights among its candidates (tools/run.sh): the rule of
wishnet/evaluation/selection.py over the candidates' whole-tile validations
-- drop those whose F1 is more than 1 point below the highest, then the lowest
best-F1 MAE.

    python tools/select_weights.py [--reason-file F] NAME=<bdh_metrics.json> ...

Prints the chosen NAME; the one-line reason (every candidate's F1 and best-F1
MAE, the F1 floor, what was dropped) goes to stderr and to --reason-file.
Exit status 1 when no candidate has a finite F1 and MAE.  The candidates are
listed in the order that breaks exact ties.
"""
import argparse
import importlib.util
import json
import os
import sys


def load_selection():
    """wishnet/evaluation/selection.py itself, without importing the package
    (and with it torch and detectron2)."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "wishnet_selection", os.path.join(repo, "wishnet", "evaluation", "selection.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_candidates(specs, sel):
    """["NAME=path", ...] -> [(NAME, F1, best-F1 MAE)] from bdh_metrics.json files"""
    out = []
    for spec in specs:
        name, _, path = spec.partition("=")
        with open(path, encoding="utf-8") as f:
            o = json.load(f)["overall"]
        out.append((name, o.get(sel.F1_KEY), o.get(sel.MAE_KEY)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cands", nargs="+", help="NAME=<bdh_metrics.json>")
    ap.add_argument("--reason-file", default=None)
    args = ap.parse_args()
    sel = load_selection()
    cands = read_candidates(args.cands, sel)
    pick = sel.choose(cands)
    reason = sel.explain(cands, pick)
    print(reason, file=sys.stderr)
    if args.reason_file:
        with open(args.reason_file, "w", encoding="utf-8") as f:
            f.write(reason + "\n")
    if pick is None:
        sys.exit(1)
    print(pick[0])


if __name__ == "__main__":
    main()
