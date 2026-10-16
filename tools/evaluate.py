# -*- coding: utf-8 -*-
"""Whole-tile evaluation of one checkpoint on the protocol of the paper
(wishnet/evaluation/evaluator.py), optionally with the up-down flip
averaging of the heights (wishnet/evaluation/flip.py) and with an input
zeroed.

    python tools/evaluate.py --config-file configs/bdh/wishnet.yaml \\
        --weights output/wishnet/model_best_h.pth --out <dir> \\
        [--dataset bdh_val] [--flip] [--zero gf3|insar|both] [--score-thresh T [--score-thresh-mae T]] \\
        [config overrides ...]

Writes <dir>/orig/ (the model as it is) and with --flip <dir>/avg/ (the same
detections, each paired detection's height the mean of the original and the
flipped tile's pass), each with bdh_metrics.json and bdh_predictions.json (for
tools/paired_compare.py), and <dir>/evaluate.json with both results and the
pairing rate.  The paper's model comparisons read the heights of avg (MAE,
RMSE, per band, and the MAE of the BDH protocol, bdhnet/*) and the detection
of orig (F1, mAP50 and the threshold; avg has the same detections); orig's
bdhnet/MAE is the one set against published numbers.  The zeroing diagnostics
and the development evidence of the paper are unflipped (orig).

--zero sets MODEL.WISHNET.SAR.ZERO and / or MODEL.WISHNET.INSAR.ZERO: the
standardised input replaced by zeros, the training mean.  --score-thresh and
--score-thresh-mae freeze the best-F1 and the best-MAE operating points
(BDH_EVAL.SCORE_THRESH_F1 / _MAE) at thresholds found on another evaluation,
as the zeroing diagnostics do with the full input's.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from detectron2.checkpoint import DetectionCheckpointer

import train_net as T
from wishnet import WISHDatasetMapper
from wishnet.evaluation.flip import averaged, flipped, pair, shallow, write_back

ZERO = {"gf3": ["MODEL.WISHNET.SAR.ZERO", "True"], "insar": ["MODEL.WISHNET.INSAR.ZERO", "True"],
        "both": ["MODEL.WISHNET.SAR.ZERO", "True", "MODEL.WISHNET.INSAR.ZERO", "True"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="bdh_val", help="a registered dataset (wishnet/data/register.py)")
    ap.add_argument("--flip", action="store_true", help="the up-down flip averaging of the heights")
    ap.add_argument("--zero", choices=sorted(ZERO), default=None)
    ap.add_argument("--score-thresh", type=float, default=None,
                    help="freeze the best-F1 operating point at this score threshold")
    ap.add_argument("--score-thresh-mae", type=float, default=None,
                    help="freeze the best-MAE operating point at this score threshold")
    ap.add_argument("--min-iou", type=float, default=0.5, help="the flip pairing's mask IoU")
    ap.add_argument("--n", type=int, default=0, help="tiles (0 = all; fewer for a smoke test)")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    # the overrides first: the dataset, the zeroing, the frozen thresholds and the
    # output folder of this evaluation are set after them
    opts = list(a.opts) + ["DATASETS.TEST", '("%s",)' % a.dataset] + (ZERO[a.zero] if a.zero else [])
    if a.score_thresh is not None:
        opts += ["BDH_EVAL.SCORE_THRESH_F1", str(a.score_thresh)]
    if a.score_thresh_mae is not None:
        opts += ["BDH_EVAL.SCORE_THRESH_MAE", str(a.score_thresh_mae)]
    opts += ["OUTPUT_DIR", os.path.join(a.out, "work")]
    args = argparse.Namespace(config_file=a.config_file, opts=opts, eval_only=True, resume=False, num_gpus=1,
                              machine_rank=0, num_machines=1, dist_url="auto")
    cfg = T.setup(args)
    model = T.Trainer.build_model(cfg)
    DetectionCheckpointer(model).load(a.weights)
    model.eval()
    ds = cfg.DATASETS.TEST[0]
    names = ["orig"] + (["avg"] if a.flip else [])
    evals = {k: T.Trainer.build_evaluator(cfg, ds, os.path.join(a.out, k)) for k in names}
    for e in evals.values():
        e.reset()
    mapper = WISHDatasetMapper(cfg, False)
    loader = T.Trainer.build_test_loader(cfg, ds)
    n_det = n_pair = n_tiles = 0
    t0 = time.time()
    for it, batch in enumerate(loader):
        if a.n and it >= a.n:
            break
        d = batch[0]
        n_tiles += 1
        with torch.no_grad():
            out1 = model([d])[0]["instances"]
            if a.flip:
                second = model([flipped(mapper, d)])[0]["instances"]
                out2 = write_back(second, float(d["height"]))
                del second
        evals["orig"].process([d], [{"instances": shallow(out1)}])
        if a.flip:
            match, _ = pair(out1.pred_masks, out2.pred_masks, out1.scores, a.min_iou)
            n_det += len(match)
            n_pair += int((match >= 0).sum())
            evals["avg"].process([d], [{"instances": averaged(out1, out2, match)}])
            del out2
        del out1
        if (it + 1) % 25 == 0:
            print("[evaluate] %d tiles, %.1f s" % (it + 1, time.time() - t0), flush=True)
    res = {k: e.evaluate() for k, e in evals.items()}
    protocol = {k: v.get("bdh", v) for k, v in res.items()}
    result = {"config": a.config_file, "weights": a.weights, "dataset": ds, "tiles": n_tiles,
              "zero": a.zero, "score_thresh_frozen": {"f1op": a.score_thresh, "maeop": a.score_thresh_mae},
              "protocol": protocol,
              "seconds": time.time() - t0}
    if a.flip:
        result.update(min_iou=a.min_iou, detections=n_det, paired=n_pair,
                      paired_fraction=n_pair / max(n_det, 1))
    with open(os.path.join(a.out, "evaluate.json"), "w") as f:
        json.dump(result, f, indent=1)
    for k in names:
        p = protocol[k]
        print("[evaluate] %-4s F1 %s  mAP50 %s  MAE %s  RMSE %s  BDH-Net MAE %s" % (
            k, p.get("f1op/F1@0.5"), p.get("segAP50"), p.get("f1op/MAE"), p.get("f1op/RMSE"),
            p.get("bdhnet/MAE")))


if __name__ == "__main__":
    main()
