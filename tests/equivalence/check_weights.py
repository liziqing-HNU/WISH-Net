# -*- coding: utf-8 -*-
"""Load a checkpoint (model_best_h.pth, model_final.pth, ...; of this code, of
the code under its earlier name or of the original implementation) into this
code's model, strictly: every parameter and buffer of the model must be in the
checkpoint and the checkpoint must hold nothing else.

    python tests/equivalence/check_weights.py --config-file configs/bdh/wishnet.yaml \\
        --weights <model_best_h.pth> [config overrides ...]

The checkpoint's "model" is the moving average of the weights (the training
hooks save it so); model_final.pth also holds the raw weights as "model_raw",
which are checked too.  Exit status 1 on any mismatch.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

import train_net as T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    args = argparse.Namespace(config_file=a.config_file, opts=list(a.opts) + ["MODEL.DEVICE", "cpu"],
                              eval_only=True, resume=False, num_gpus=1, machine_rank=0, num_machines=1,
                              dist_url="auto")
    cfg = T.setup(args)
    model = T.Trainer.build_model(cfg)
    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    print("checkpoint %s: %s" % (a.weights, sorted(k for k in ck if k != "model" and k != "model_raw")))
    ok = True
    for key in ("model", "model_raw"):
        if key not in ck:
            continue
        sd = ck[key]
        res = model.load_state_dict(sd, strict=False)
        mine = model.state_dict()
        shapes = [k for k in sd if k in mine and tuple(sd[k].shape) != tuple(mine[k].shape)]
        good = not res.missing_keys and not res.unexpected_keys and not shapes
        print("%-9s %d tensors: missing %d %s, unexpected %d %s, other shape %d %s -> %s" % (
            key, len(sd), len(res.missing_keys), res.missing_keys[:3], len(res.unexpected_keys),
            res.unexpected_keys[:3], len(shapes), shapes[:3], "ok" if good else "FAIL"))
        ok &= good
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
