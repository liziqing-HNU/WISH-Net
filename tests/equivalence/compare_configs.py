# -*- coding: utf-8 -*-
"""Two checkouts' effective configurations (dump_config.py on each side), key
by key.  The configuration namespace and the registry names of the code under
its earlier name (MODEL.GFOPT, GFOPTNet, GFOPTDecoder) are first mapped onto
this code's (MODEL.WISHNET, WISHNet, WISHDecoder).  Data and weight paths and
OUTPUT_DIR may differ; any other difference must be named with --allow.

    python tests/equivalence/compare_configs.py <a.json> <b.json> [--pair A.yaml=B.yaml ...] \\
        [--allow KEY ...] [--same-keys]

Compared are the configurations of the same file name in both dumps and the
pairs given with --pair.  --same-keys: a key in one dump only is a failure
too.  Exit status 1 on any other difference.

Against the original implementation, the differences by design are the 22k
stop written into SOLVER.MAX_ITER (its runs were configured to 24k and
stopped at 22k, tests/check_schedule.py) and its optional readouts, off here
(they read, they do not train): --allow SOLVER.MAX_ITER --allow
SOLVER.KEEP_BEST_HEIGHT and --allow for MODEL.WISHNET.FIXED_SET_HEIGHT.ENABLED,
.BANDS, .DETECTION and MODEL.WISHNET.SAR_MEAN_READOUT.ENABLED, .PER_MODALITY.
"""
import argparse
import json
import sys

PATHS = {"DATASETS.BDH_ROOT", "DATASETS.BDH_PREPARED", "MODEL.WISHNET.SAR.ROOT", "MODEL.WISHNET.INSAR.ROOT",
         "MODEL.WISHNET.OPT_BACKBONE_WEIGHTS", "MODEL.WISHNET.SAR.WEIGHTS", "MODEL.WISHNET.INSAR.WEIGHTS",
         "OUTPUT_DIR"}
NAMES = {"GFOPTNet": "WISHNet", "GFOPTDecoder": "WISHDecoder"}


def mapped(flat):
    """A dump with the earlier namespace and registry names mapped onto this code's."""
    out = {}
    for k, v in flat.items():
        if k.startswith("MODEL.GFOPT."):
            k = "MODEL.WISHNET." + k[len("MODEL.GFOPT."):]
        out[k] = NAMES.get(v, v) if isinstance(v, str) else v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--pair", action="append", default=[], help="A.yaml=B.yaml: configurations of different names")
    ap.add_argument("--allow", action="append", default=[], help="a key that differs by design")
    ap.add_argument("--same-keys", action="store_true", help="a key in one dump only is a failure")
    x = ap.parse_args()
    da, db = json.load(open(x.a)), json.load(open(x.b))
    pairs = [(n, n) for n in sorted(set(da) & set(db))] + [tuple(p.split("=", 1)) for p in x.pair]
    allow = set(x.allow)
    ok = bool(pairs)
    for na, nb in pairs:
        if na not in da or nb not in db:
            print("== %s / %s: not in the dumps" % (na, nb))
            ok = False
            continue
        a, b = mapped(da[na]), mapped(db[nb])
        shared = set(a) & set(b)
        diff = {k: (a[k], b[k]) for k in shared if a[k] != b[k]}
        unexpected = {k: v for k, v in diff.items() if k not in allow | PATHS}
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        print("== %s vs %s: %d shared keys, %d differ; keys only in A: %d, only in B: %d"
              % (na, nb, len(shared), len(diff), len(only_a), len(only_b)))
        for k, v in sorted(diff.items()):
            tag = "allowed   " if k in allow else "path      " if k in PATHS else "UNEXPECTED"
            print("   %s %-48s %r -> %r" % (tag, k, v[0], v[1]))
        if x.same_keys and (only_a or only_b):
            print("   only in A: %s; only in B: %s" % (only_a[:5], only_b[:5]))
            ok = False
        ok &= not unexpected
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
