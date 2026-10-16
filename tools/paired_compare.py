# -*- coding: utf-8 -*-
"""Paired per-building comparison of evaluated models: the paired MAE gains
and their 95 % intervals of the paper.

Every valid ground-truth building of the evaluation crops (not truncated, not
an ignore region, a recorded height of at least 3 m) gets each model's
prediction at the model's best-F1 operating point: COCO segm matching at IoU
0.5 over the model's detections (maxDets = BDH_EVAL.MAX_DETS), keeping the
detections at or above the threshold of the model's bdh_metrics.json -- the
matching the evaluator scores f1op/MAE with (checked against it).  A model B
is compared with a reference A on the buildings both detect: gain = MAE_A -
MAE_B (> 0: B better), overall and per height bin, with a 95 % interval from a
bootstrap over tiles (2000 resamples, multinomial tile weights, seed 0).

    python tools/paired_compare.py --gt datasets/bdh/prepared_v2/instances_val_crop2048.json \\
        --run OPT=output/opt/final_val/avg --run WISH-Net=output/wishnet/final_val/avg \\
        --ref OPT --out <dir> [--pair A,B ...] [--exclude-gt FILE | --only-gt FILE]

A run folder holds the bdh_predictions.json and bdh_metrics.json of one
evaluation (tools/evaluate.py: <out>/avg for the flip-averaged heights,
<out>/orig for the model as it is; a zeroing evaluation carries its frozen
threshold in its bdh_metrics.json).  --thresh NAME=value overrides a run's
threshold.  Every run is compared with --ref, and --pair adds comparisons.
--exclude-gt / --only-gt (a file of annotation ids of --gt, one per line or a
CSV with an ann_id column) leave buildings out of, or keep only them in, the
comparisons; the matching does not change.

Writes <out>/analysis.json (schema "wishnet-paired-1": per comparison A, B,
n, mae_A, mae_B, gain, ci95, the recalls, and the same per height bin),
<out>/analysis.md (the table) and <out>/per_building.csv.  The functions are
those of the analysis the paper's numbers come from, without its
building-geometry dimensions.
"""
import argparse
import contextlib
import csv
import io
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.optimize import linear_sum_assignment

# the evaluator's height bins: inner cut points (a value below the first cut
# falls in the first bin) and labels
H_CUTS = [10, 20, 30, 50, 100]
EVAL_BIN_LABELS = ["[3,10)", "[10,20)", "[20,30)", "[30,50)", "[50,100)", "[100,inf)"]
PAIRED_SCHEMA = "wishnet-paired-1"


def digitize(x, cuts):
    out = np.searchsorted(np.asarray(cuts, float), x, side="right")
    out = out.astype(int)
    out[~np.isfinite(x)] = -1
    return out


# ------------------------------------------------------------ ground truth
def load_gt(path):
    """The evaluator's GT: truncated buildings and <json>_ignore.json become
    ignore regions (iscrowd), ids as BDHEvaluator._apply_ignore gives them."""
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(path)
    ds = coco.dataset
    for a in ds["annotations"]:
        if a.get("truncated", False):
            a["iscrowd"] = 1
    ign = os.path.splitext(path)[0] + "_ignore.json"
    with open(ign, "r", encoding="utf-8") as f:
        extra = json.load(f)["annotations"]
    nid = max([a["id"] for a in ds["annotations"]] + [0]) + 1
    for a in extra:
        b = dict(a)
        b["id"] = nid
        b["iscrowd"] = 1
        b["bd_ignore_region"] = True
        ds["annotations"].append(b)
        nid += 1
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def valid_buildings(coco):
    """One row per valid GT building (not an ignore region, not truncated, at
    least 3 m), tiles in id order: gt_id, image_id, h."""
    rows = []
    for img_id in sorted(coco.getImgIds()):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        for a in anns:
            h = float(a.get("bd_height", 0.0))
            if a.get("bd_ignore_region", False) or a.get("truncated", False) or h < 3.0:
                continue
            rows.append({"gt_id": int(a["id"]), "image_id": int(img_id), "h": h})
    return rows


def read_ids(path):
    """GT annotation ids from a file: one per line, or a CSV with an ``ann_id`` (else ``id``) column."""
    with open(path, "r", encoding="utf-8-sig") as f:
        text = f.read()
    lines = [l for l in text.splitlines() if l.strip()]
    if lines and not lines[0].strip().lstrip("-").isdigit():
        r = list(csv.DictReader(lines))
        col = "ann_id" if "ann_id" in r[0] else "id"
        return {int(x[col]) for x in r if str(x.get(col, "")).strip()}
    return {int(l.split(",")[0]) for l in lines}


# --------------------------------------------------------------------- runs
def _metrics(run_dir):
    p = os.path.join(run_dir, "bdh_metrics.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def read_run(coco, gt_ids, run_dir, max_dets, thresh=None):
    """Per valid GT (order of ``gt_ids``): the operating-point detection and the
    Hungarian assignment.  Returns arrays and the reproduction check."""
    t0 = time.time()
    with open(os.path.join(run_dir, "bdh_predictions.json"), "r", encoding="utf-8") as f:
        preds = json.load(f)
    met = _metrics(run_dir)
    if thresh is None:
        if met is None:
            raise SystemExit("%s: no bdh_metrics.json and no --thresh" % run_dir)
        thresh = float(met["points"]["f1op"]["score_thresh"])

    with contextlib.redirect_stdout(io.StringIO()):
        cdt = coco.loadRes([dict(p) for p in preds])
        E = COCOeval(coco, cdt, iouType="segm")
        E.params.iouThrs = np.array([0.5])
        E.params.areaRng = [[0, 1e10]]
        E.params.areaRngLbl = ["all"]
        E.params.maxDets = [int(max_dets)]
        E.evaluate()

    pos = {g: i for i, g in enumerate(gt_ids)}
    n = len(gt_ids)
    op = {k: np.full(n, np.nan) for k in ("h", "score", "iou")}
    matched = np.zeros(n, dtype=bool)
    n_tp_all = 0
    for e in E.evalImgs:
        if e is None:
            continue
        dm = np.asarray(e["dtMatches"])[0]
        di = np.asarray(e["dtIgnore"])[0].astype(bool)
        for j, did in enumerate(e["dtIds"]):
            if di[j] or e["dtScores"][j] < thresh or dm[j] <= 0:
                continue
            n_tp_all += 1
            g = int(dm[j])
            if g not in pos:
                continue
            k = pos[g]
            d = cdt.anns[did]
            gt = coco.anns[g]
            matched[k] = True
            op["h"][k] = float(d["bd_height"])
            op["score"][k] = float(d["score"])
            op["iou"][k] = float(mask_util.iou([d["segmentation"]], [coco.annToRLE(gt)], [0])[0][0])

    # the Hungarian schedule metric, as wishnet.evaluation.evaluator._hungarian_height
    by_img = {}
    for p in preds:
        by_img.setdefault(p["image_id"], []).append(p)
    hung = {k: np.full(n, np.nan) for k in ("h", "iou")}
    herr = []
    for img_id in coco.getImgIds():
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
                if not a.get("bd_ignore_region", False)]
        if not anns:
            continue
        scored = [not a.get("truncated", False) for a in anns]
        dts = by_img.get(img_id, [])
        got = {}
        if dts:
            g_rle = [coco.annToRLE(a) for a in anns]
            iou = np.asarray(mask_util.iou([d["segmentation"] for d in dts], g_rle, [0] * len(g_rle)),
                             dtype=np.float64).reshape(len(dts), len(anns))
            for a_, b_ in zip(*linear_sum_assignment(-iou)):
                got[b_] = (float(dts[a_]["bd_height"]), float(iou[a_, b_]))
        for k, a in enumerate(anns):
            if not scored[k]:
                continue
            hd, v = got.get(k, (0.0, 0.0))
            herr.append(hd - float(a["bd_height"]))
            if a["id"] in pos:
                hung["h"][pos[a["id"]]] = hd
                hung["iou"][pos[a["id"]]] = v

    check = OrderedDict()
    check["thresh"] = thresh
    hgt = np.array([float(coco.anns[g]["bd_height"]) for g in gt_ids])
    check["f1op_MAE"] = float(np.abs(op["h"][matched] - hgt[matched]).mean())
    check["f1op_n"] = int(matched.sum())
    check["hung_MAE"] = float(np.abs(np.asarray(herr)).mean()) if herr else float("nan")
    if met is not None:
        o = met.get("overall", {})
        check["ref_f1op_MAE"] = o.get("f1op/MAE")
        check["ref_f1op_n"] = o.get("f1op/n_matched")
        check["ref_hung_MAE"] = o.get("hungarian/MAE")
        check["ref_F1"] = o.get("f1op/F1@0.5")
    check["n_tp_all"] = n_tp_all
    check["seconds"] = round(time.time() - t0, 1)
    return {"op": op, "matched": matched, "hung": hung, "check": check}


# --------------------------------------------------------------- statistics
class Boot:
    """Bootstrap over tiles: per-tile sums, resampled with multinomial counts."""

    def __init__(self, tile, n_boot, seed):
        self.tile = tile
        self.T = int(tile.max()) + 1
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(self.T, np.full(self.T, 1.0 / self.T), size=n_boot).astype(np.float64)

    def diff_ci(self, a, b, sel):
        """95 % interval of mean(b) - mean(a) over the rows in ``sel``."""
        t = self.tile[sel]
        sa = np.bincount(t, a[sel], self.T)
        sb = np.bincount(t, b[sel], self.T)
        c = np.bincount(t, None, self.T)
        d = (self.W @ (sb - sa)) / np.maximum(self.W @ c, 1.0)
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def compare(hgt, A, B, boot, sel_all=None):
    """Reference A against run B overall and per height bin.  Gain = MAE_A -
    MAE_B (> 0: B better), on the buildings both detect at their operating
    points; recall of each; and the Hungarian MAE over all valid buildings."""
    n = len(hgt)
    sel_all = np.ones(n, dtype=bool) if sel_all is None else sel_all
    both = A["matched"] & B["matched"] & sel_all
    eA = A["op"]["h"] - hgt
    eB = B["op"]["h"] - hgt
    hA = A["hung"]["h"] - hgt
    hB = B["hung"]["h"] - hgt

    def row(sel_bin):
        s = both & sel_bin
        r = OrderedDict()
        r["n"] = int(s.sum())
        r["n_all"] = int((sel_bin & sel_all).sum())
        if r["n"]:
            aA, aB = np.abs(eA), np.abs(eB)
            r["mae_A"] = float(aA[s].mean())
            r["mae_B"] = float(aB[s].mean())
            r["gain"] = r["mae_A"] - r["mae_B"]
            lo, hi = boot.diff_ci(aB, aA, s)
            r["gain_lo"], r["gain_hi"] = lo, hi
            r["bias_A"] = float(eA[s].mean())
            r["bias_B"] = float(eB[s].mean())
            r["win_B"] = float((aB[s] < aA[s]).mean())
        sa = sel_bin & sel_all
        if sa.any():
            r["rec_A"] = float(A["matched"][sa].mean())
            r["rec_B"] = float(B["matched"][sa].mean())
            gA, gB = np.abs(hA), np.abs(hB)
            ok = sa & np.isfinite(gA) & np.isfinite(gB)
            if ok.any():
                r["hung_A"] = float(gA[ok].mean())
                r["hung_B"] = float(gB[ok].mean())
                r["hung_gain"] = r["hung_A"] - r["hung_B"]
                lo, hi = boot.diff_ci(gB, gA, ok)
                r["hung_lo"], r["hung_hi"] = lo, hi
        return r

    out = OrderedDict()
    out["overall"] = row(np.ones(n, dtype=bool))
    idx = digitize(hgt, H_CUTS)
    out["h"] = OrderedDict((lab, row(idx == i)) for i, lab in enumerate(EVAL_BIN_LABELS))
    return out


def paired_entry(a, b, cmp):
    def one(r):
        return [("n", r.get("n", 0)), ("n_all", r.get("n_all", 0)), ("mae_A", r.get("mae_A")),
                ("mae_B", r.get("mae_B")), ("gain", r.get("gain")),
                ("ci95", [r.get("gain_lo"), r.get("gain_hi")]),
                ("recall_A", r.get("rec_A")), ("recall_B", r.get("rec_B"))]
    e = OrderedDict([("A", a), ("B", b)] + one(cmp["overall"]))
    e["bins"] = [OrderedDict([("bin", lab)] + one(cmp["h"][lab])) for lab in EVAL_BIN_LABELS]
    return e


def clean_json(x):
    """numpy scalars to Python numbers, NaN and infinities to null."""
    if isinstance(x, dict):
        return OrderedDict((k, clean_json(v)) for k, v in x.items())
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    return x


def fmt(v, nd=2, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return ("%+.*f" if sign else "%.*f") % (nd, v)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="instances_val_crop2048.json (its _ignore.json next to it)")
    ap.add_argument("--run", action="append", required=True, help="NAME=DIR (bdh_predictions.json, bdh_metrics.json)")
    ap.add_argument("--ref", required=True, help="the reference run, e.g. OPT")
    ap.add_argument("--pair", action="append", default=[], help="another comparison A,B (A the reference)")
    ap.add_argument("--thresh", action="append", default=[], help="NAME=score threshold (overrides bdh_metrics.json)")
    ap.add_argument("--max-dets", type=int, default=600, help="BDH_EVAL.MAX_DETS of the evaluation")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    flt = ap.add_mutually_exclusive_group()
    flt.add_argument("--exclude-gt", default=None, help="file of GT annotation ids of --gt (one per line, or a CSV "
                     "with an ann_id column): left out of the comparisons; the matching is unchanged")
    flt.add_argument("--only-gt", default=None, help="the same file form: the comparisons on these buildings only")
    args = ap.parse_args()

    runs = OrderedDict()
    for r in args.run:
        name, path = r.rsplit("=", 1)
        runs[name] = path
    if args.ref not in runs:
        raise SystemExit("--ref %s is not among the runs" % args.ref)
    th = {k: float(v) for k, v in (t.rsplit("=", 1) for t in args.thresh)}
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    coco = load_gt(args.gt)
    rows = valid_buildings(coco)
    hgt = np.array([r["h"] for r in rows])
    gt_ids = [r["gt_id"] for r in rows]
    image_id = np.array([r["image_id"] for r in rows])
    print("[paired] %d valid GT buildings on %d tiles" % (len(rows), len(np.unique(image_id))), flush=True)

    gt_filter, keep = None, np.ones(len(rows), dtype=bool)
    if args.exclude_gt or args.only_gt:
        path = args.exclude_gt or args.only_gt
        ids = read_ids(path)
        hit = np.isin(np.asarray(gt_ids), list(ids))
        keep = ~hit if args.exclude_gt else hit
        gt_filter = OrderedDict([("mode", "exclude" if args.exclude_gt else "only"), ("file", os.path.abspath(path)),
                                 ("n_ids", len(ids)), ("n_ids_valid", int(hit.sum())),
                                 ("n_ids_not_valid", len(ids) - int(hit.sum())), ("n_rows_kept", int(keep.sum()))])
        print("[paired] %s %d GT ids from %s: %d are valid buildings; %d buildings left in the comparisons"
              % (gt_filter["mode"], len(ids), path, gt_filter["n_ids_valid"], int(keep.sum())), flush=True)

    _, tile = np.unique(image_id, return_inverse=True)
    boot = Boot(tile, args.boot, args.seed)

    R = OrderedDict()
    for name, path in runs.items():
        R[name] = read_run(coco, gt_ids, path, args.max_dets, th.get(name))
        c = R[name]["check"]
        print("[paired] %-14s thr %.4f  f1op MAE %.4f (n %d) vs the evaluator's %s (n %s)  "
              "hungarian %.4f vs %s  [%.0f s]"
              % (name, c["thresh"], c["f1op_MAE"], c["f1op_n"], c.get("ref_f1op_MAE"), c.get("ref_f1op_n"),
                 c["hung_MAE"], c.get("ref_hung_MAE"), c["seconds"]), flush=True)

    pairs = [(args.ref, n) for n in runs if n != args.ref]
    pairs += [tuple(p.split(",")) for p in args.pair]
    compares = OrderedDict(("%s|%s" % (a, b), compare(hgt, R[a], R[b], boot, keep)) for a, b in pairs)
    paired = [paired_entry(*key.split("|"), cmp) for key, cmp in compares.items()]

    with open(os.path.join(args.out, "per_building.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt_id", "image_id", "h"] + sum([["%s_op_h" % n, "%s_op_score" % n, "%s_op_iou" % n,
                                                      "%s_hung_h" % n, "%s_hung_iou" % n] for n in R], [])
                   + (["in_comparisons"] if gt_filter else []))
        for i, r in enumerate(rows):
            vals = [r["gt_id"], r["image_id"], r["h"]]
            for n in R:
                vals += [R[n]["op"]["h"][i], R[n]["op"]["score"][i], R[n]["op"]["iou"][i],
                         R[n]["hung"]["h"][i], R[n]["hung"]["iou"][i]]
            if gt_filter:
                vals.append(int(keep[i]))
            w.writerow(["" if isinstance(v, float) and not np.isfinite(v) else
                        (round(v, 4) if isinstance(v, (float, np.floating)) else v) for v in vals])
    out = OrderedDict()
    out["schema"] = PAIRED_SCHEMA
    out["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out["gt"] = os.path.abspath(args.gt)
    out["n_valid"] = len(rows)
    if gt_filter:
        out["gt_filter"] = gt_filter
    out["bootstrap"] = {"resamples": args.boot, "seed": args.seed, "unit": "tile"}
    out["runs"] = OrderedDict((n, OrderedDict([
        ("dir", runs[n]), ("thresh", R[n]["check"]["thresh"]),
        ("f1op_MAE", R[n]["check"]["f1op_MAE"]), ("f1op_n", R[n]["check"]["f1op_n"]),
        ("F1", R[n]["check"].get("ref_F1")), ("hungarian_MAE", R[n]["check"]["hung_MAE"])])) for n in R)
    out["paired"] = paired
    out["checks"] = {n: R[n]["check"] for n in R}
    out["compare"] = compares
    tmp = os.path.join(args.out, "analysis.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean_json(out), f, indent=1, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, os.path.join(args.out, "analysis.json"))

    L = ["# Paired comparison", "",
         "%d valid ground-truth buildings (not truncated, not ignore regions) of %s; each model matched at its "
         "best-F1 operating point (COCO segm, IoU 0.5).  Gain = MAE(A) - MAE(B) on the buildings both detect, "
         "> 0: B better; 95 %% interval from a bootstrap over tiles (%d resamples, seed %d)."
         % (len(rows), os.path.basename(args.gt), args.boot, args.seed), ""]
    if gt_filter:
        L += ["Buildings: %s the %d ids of %s (%d valid), %d left." % (
            gt_filter["mode"], gt_filter["n_ids"], os.path.basename(gt_filter["file"]), gt_filter["n_ids_valid"],
            gt_filter["n_rows_kept"]), ""]
    L += ["| A | B | n | MAE A | MAE B | gain | 95% interval | " + " | ".join(EVAL_BIN_LABELS) + " |",
          "|---|---|---|---|---|---|---|" + "---|" * len(EVAL_BIN_LABELS)]
    for p in paired:
        L.append("| %s | %s | %d | %s | %s | %s | [%s, %s] | " % (
            p["A"], p["B"], p["n"], fmt(p["mae_A"], 3), fmt(p["mae_B"], 3), fmt(p["gain"], 3, True),
            fmt(p["ci95"][0], 3, True), fmt(p["ci95"][1], 3, True))
            + " | ".join(fmt(b["gain"], 2, True) for b in p["bins"]) + " |")
    L += ["", "Reproduction check (this script against each run's bdh_metrics.json):", "",
          "| run | threshold | best-F1 MAE | evaluator | matched | evaluator |", "|---|---|---|---|---|---|"]
    for n, c in out["checks"].items():
        L.append("| %s | %.4f | %.4f | %s | %d | %s |" % (n, c["thresh"], c["f1op_MAE"], fmt(c.get("ref_f1op_MAE"), 4),
                                                           c["f1op_n"], c.get("ref_f1op_n")))
    with open(os.path.join(args.out, "analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("[paired] wrote %s (%.0f s)" % (args.out, time.time() - t0), flush=True)


if __name__ == "__main__":
    sys.exit(main())
