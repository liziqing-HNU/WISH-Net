# -*- coding: utf-8 -*-
"""One side of an equivalence check between two checkouts: this code and
another (the original implementation the paper's runs were trained with, or an
earlier version of this code).  Run from the root of the checkout under test,
it builds the model of a configuration with that checkout's own setup, then

    init    saves the initial state_dict (seed of the configuration);
    run     loads a state_dict (the other side's init, for the same starting
            weights, or a checkpoint's "model"), runs the evaluation forward
            on the validation tiles and ``--steps`` training steps on the
            training tiles, and saves the outputs, every loss term of every
            step and the final parameters.

    python <this file> --side <label> --mode init|run --config-file <cfg> --out <dir>
        [--weights <init.pt>] [--steps 3] [--eval-tiles -1] [config overrides ...]

--eval-tiles limits the evaluation forward to the first N validation tiles (0:
none, -1: all); its outputs keep every mask, so on the 2048 px tiles of the
real data it is for a few tiles only -- tools/evaluate.py and
compare_eval.py compare whole evaluations.

The training steps are the trainer's computation without its hooks: AdamW
with full-model clipping as Trainer.build_optimizer builds it, the gates
held as GateWarmupHook holds them in the first steps, batches of the
training tiles in a fixed order through the training mapper (its random
crops, jitter, flips and speckle from the seeded numpy state).  Every random
generator is seeded with SEED before the model is built and again before the
steps, so both sides draw the same numbers.

Determinism.  The forward is deterministic, on a CPU and on a GPU.  Training
is not, run to run: on a CPU with several threads some reductions of the
backward are not (two runs of one code part after a few steps), and on a GPU
MaskDINO's deformable attention accumulates its gradients with atomic
additions.  ``--threads 1 --deterministic`` makes CPU training deterministic
(torch.set_num_threads(1), torch.use_deterministic_algorithms(True));
``--checksums`` writes checksums.json, a bit-exact checksum of every
parameter's gradient and of every parameter after each step, which compare.py
compares step by step.  On a GPU compare.py --noise measures the two codes'
difference against a second run of one side.
"""
import argparse
import importlib
import importlib.util
import json
import os
import random
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from detectron2.data import DatasetCatalog
from detectron2.utils.events import EventStorage


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


if not torch.cuda.is_available():
    # MaskDINO's denoising queries and some zero losses are put on the GPU with
    # .cuda() and .to("cuda"); on a CPU both sides keep them where they are
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    _to = torch.Tensor.to

    def _cuda(d):
        return (isinstance(d, str) and d.startswith("cuda")) or (isinstance(d, torch.device) and d.type == "cuda")

    def _to_cpu(self, *args, **kwargs):
        args = tuple("cpu" if _cuda(a) else a for a in args)
        if _cuda(kwargs.get("device")):
            kwargs["device"] = "cpu"
        return _to(self, *args, **kwargs)

    torch.Tensor.to = _to_cpu


def checksum(t):
    """An integer that changes with any bit of a float32 tensor."""
    x = t.detach().cpu().contiguous().view(-1)
    v = (x if x.dtype == torch.float32 else x.float()).view(torch.int32).to(torch.int64)
    w = torch.arange(1, v.numel() + 1, dtype=torch.int64) % 1000003
    return int((v * w).sum())


def side_modules():
    """(setup, Trainer, dataset mapper, the configuration's namespace) of the
    checkout in the working directory: this code (train_net.py, wishnet,
    MODEL.WISHNET), or the code under its earlier name (gfoptnet, MODEL.GFOPT;
    the original implementation's training script is train_gfoptnet.py)."""
    T = importlib.import_module("train_net" if os.path.exists("train_net.py") else "train_gfoptnet")
    if importlib.util.find_spec("wishnet") is not None:
        from wishnet import WISHDatasetMapper as Mapper
        return T.setup, T.Trainer, Mapper, "WISHNET"
    from gfoptnet import GFOPTDatasetMapper as Mapper
    return T.setup, T.Trainer, Mapper, "GFOPT"


def instances_fields(inst):
    return {k: v.detach().cpu().clone() for k, v in inst.get_fields().items() if torch.is_tensor(v)} | {
        "pred_boxes": inst.pred_boxes.tensor.detach().cpu().clone()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", default="side", help="a label for the messages")
    ap.add_argument("--mode", choices=("init", "run"), required=True)
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--eval-tiles", type=int, default=-1, help="validation tiles of the evaluation forward "
                    "(-1: all, 0: none)")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0: as they are)")
    ap.add_argument("--deterministic", action="store_true", help="torch.use_deterministic_algorithms(True)")
    ap.add_argument("--checksums", action="store_true", help="per step, checksums of the gradients and parameters")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
        torch.set_num_interop_threads(a.threads)
    if a.deterministic:
        torch.use_deterministic_algorithms(True)
    os.makedirs(a.out, exist_ok=True)
    setup, Trainer, Mapper, ns = side_modules()
    args = argparse.Namespace(config_file=a.config_file, opts=list(a.opts) + ["OUTPUT_DIR", os.path.join(a.out, "work")],
                              eval_only=True, resume=False, num_gpus=1, machine_rank=0, num_machines=1,
                              dist_url="auto")
    cfg = setup(args)
    seed = int(cfg.SEED)
    seed_all(seed)
    model = Trainer.build_model(cfg)
    if a.mode == "init":
        sd = model.state_dict()
        torch.save(sd, os.path.join(a.out, "init.pt"))
        print("[%s] %d tensors, %d parameters" % (a.side, len(sd), sum(p.numel() for p in model.parameters())))
        return
    sd = torch.load(a.weights, map_location="cpu", weights_only=False)
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]                                  # a checkpoint: the moving average
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit("state_dict mismatch: missing %s, unexpected %s" % (missing[:5], unexpected[:5]))

    # evaluation forward on the validation tiles
    model.eval()
    val = DatasetCatalog.get(cfg.DATASETS.TEST[0])
    if a.eval_tiles >= 0:
        val = val[:a.eval_tiles]
    m_eval = Mapper(cfg, False)
    outs = []
    with torch.no_grad():
        for d in val:
            x = m_eval(d)
            outs.append(instances_fields(model([x])[0]["instances"]))
    if val:
        torch.save(outs, os.path.join(a.out, "eval_outputs.pt"))

    # training steps
    seed_all(seed)
    model.train()
    g = cfg.MODEL[ns]
    if g.MODE == "fusion" and g.FUSION.GATE_WARMUP_ITERS > 0:
        for fuse in model.sem_seg_head.predictor.sar_fuse.values():
            for p in list(fuse.gate.parameters()) + list(fuse.value.parameters()):
                p.requires_grad_(False)
    opt = Trainer.build_optimizer(cfg, model)
    train = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
    m_train = Mapper(cfg, True)
    n = int(cfg.SOLVER.IMS_PER_BATCH)
    steps, sums = [], []
    with EventStorage(0) as storage:
        for it in range(a.steps):
            storage.iter = it
            batch = [m_train(train[(it * n + j) % len(train)]) for j in range(n)]
            losses = model(batch)
            total = sum(losses.values())
            opt.zero_grad()
            total.backward()
            if a.checksums:
                grads = {k: checksum(p.grad) for k, p in model.named_parameters() if p.grad is not None}
            opt.step()
            if a.checksums:
                sums.append({"grads": grads, "params": {k: checksum(p) for k, p in model.named_parameters()}})
            steps.append({k: float(v.detach()) for k, v in losses.items()} | {"total": float(total.detach())})
            print("[%s] step %d: total %.6f over %d terms" % (a.side, it, steps[-1]["total"], len(losses)), flush=True)
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, os.path.join(a.out, "final.pt"))
    with open(os.path.join(a.out, "steps.json"), "w") as f:
        json.dump(steps, f, indent=1)
    if a.checksums:
        with open(os.path.join(a.out, "checksums.json"), "w") as f:
            json.dump({"threads": torch.get_num_threads(),
                       "deterministic": torch.are_deterministic_algorithms_enabled(), "steps": sums}, f)


if __name__ == "__main__":
    main()
