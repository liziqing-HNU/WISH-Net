# -*- coding: utf-8 -*-
"""The training schedule of SOLVER.MAX_ITER 22000 against the paper
runs' 24000 stopped at 22000 (a STOP file after step 21999).

The paper's runs were configured with MAX_ITER 24000 and stopped at 22k;
this repository writes the stop into MAX_ITER 22000.  The one schedule that
reads MAX_ITER is detectron2's learning-rate schedule (build_lr_scheduler):
the warm-up length is WARMUP_ITERS / MAX_ITER and every iteration t is
evaluated at t / MAX_ITER.  This script builds both schedules with the
installed detectron2 / fvcore and compares, bit for bit, the learning rate
of every iteration 0-21999 for the base rate and the optical backbone's rate
(0.1x); it also lists the iterations of the validations, the periodic
checkpoints and model_final of both runs (the hooks' rules, with the STOP
lowering max_iter to 22000 after step 21999 in the original implementation).  The other
schedules of the training do not read MAX_ITER: the gate warm-up counts
iterations (GATE_WARMUP_ITERS), the moving average counts its updates
(SOLVER.EMA), the denoising, the height weights and the exchange have none.

    python tests/check_schedule.py

Exit status 1 on any difference.
"""
import sys

import torch
from detectron2.config import get_cfg
from detectron2.solver import build_lr_scheduler

BASE_LR, BACKBONE = 1e-4, 0.1
STEPS, WARMUP_ITERS, WARMUP_FACTOR = (20000,), 10, 1.0
EVAL_PERIOD, CHECKPOINT_PERIOD, STOP_AT = 2000, 4000, 22000


def schedule(max_iter, n):
    """The learning rate of each of the first n iterations, per parameter group."""
    cfg = get_cfg()
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    cfg.SOLVER.BASE_LR = BASE_LR
    cfg.SOLVER.STEPS = STEPS
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.WARMUP_FACTOR = WARMUP_FACTOR
    cfg.SOLVER.WARMUP_ITERS = WARMUP_ITERS
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.GAMMA = 0.1
    p1, p2 = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([{"params": [p1], "lr": BASE_LR}, {"params": [p2], "lr": BASE_LR * BACKBONE}],
                            BASE_LR)
    sched = build_lr_scheduler(cfg, opt)
    out = []
    for _ in range(n):
        out.append(tuple(g["lr"] for g in opt.param_groups))      # the rate the step of this iteration uses
        opt.step()
        sched.step()                                              # detectron2's LRScheduler hook, after_step
    return out


def events(max_iter, stop_at=None):
    """The iterations (0-based, of the step after which it happens) of the
    validations, periodic checkpoints and model_final (detectron2's EvalHook and
    PeriodicCheckpointer; a STOP lowers max_iter to stop_at after step stop_at - 1)."""
    end = stop_at or max_iter
    ev = [it for it in range(end) if (it + 1) % EVAL_PERIOD == 0 and it + 1 != end] + [end - 1]
    ck = [it for it in range(end - 1) if (it + 1) % CHECKPOINT_PERIOD == 0]
    return {"validations": ev, "checkpoints": ck, "model_final": end - 1}


def main():
    a, b = schedule(24000, STOP_AT), schedule(22000, STOP_AT)
    diff = [t for t in range(STOP_AT) if a[t] != b[t]]
    print("learning rate, iterations 0-%d: %d differ bit for bit" % (STOP_AT - 1, len(diff)))
    for t in list(range(0, 12)) + [19998, 19999, 20000, 20001, 21999]:
        print("  t %5d  24000: %r  22000: %r  %s" % (t, a[t], b[t], "same" if a[t] == b[t] else "DIFFERENT"))
    for t in diff[:10]:
        print("  differs at t %d: %r vs %r" % (t, a[t], b[t]))
    ea, eb = events(24000, STOP_AT), events(22000)
    same_events = ea == eb
    print("validations, checkpoints, model_final: %s" % ("the same" if same_events else "DIFFERENT"))
    print("  validations after steps %s ... %s (%d), checkpoints %s, model_final after step %d" % (
        eb["validations"][:3], eb["validations"][-2:], len(eb["validations"]), eb["checkpoints"],
        eb["model_final"]))
    ok = not diff and same_events
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
