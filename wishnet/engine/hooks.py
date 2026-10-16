# -*- coding: utf-8 -*-
"""Training hooks: the best checkpoints of a run (by a validation metric, and by
the weight-selection rule), and a STOP file that ends the training cleanly."""
import json
import logging
import os

import torch
from detectron2.engine.hooks import PeriodicCheckpointer
from detectron2.engine.train_loop import HookBase
from detectron2.utils import comm

logger = logging.getLogger("wishnet.engine")


def _get_metric(storage, key):
    """Latest value of ``key`` in the event storage, or None if never written."""
    try:
        hist = storage.history(key)
    except KeyError:
        return None
    vals = hist.values()
    return float(vals[-1][0]) if vals else None


class BestCheckpointHook(HookBase):
    """Keep a copy of the best checkpoint by a validation metric, written as
    ``<name>.pth`` next to the periodic checkpoints, with ``<name>.json`` (the
    metric, its value, the iteration)."""

    def __init__(self, metric, mode="min", checkpointer=None, name="model_best", state_fn=None, record=()):
        self.metric = metric
        # further metrics whose values at the best validation go into the json
        self.record = tuple(record)
        self.sign = -1.0 if mode == "min" else 1.0
        self.checkpointer = checkpointer
        self.name = name
        # SOLVER.EMA: () -> the averaged state_dict.  The file is then the model
        # only (the average, read with MODEL.WEIGHTS, never resumed from) and
        # last_checkpoint stays on the periodic checkpoint
        self.state_fn = state_fn
        self.best = None
        self._seen = 0

    # A resumed run that forgot its best value would overwrite <name>.pth with
    # the first evaluation after the restart, however bad.
    def state_dict(self):
        return {"best": self.best}

    def load_state_dict(self, state):
        self.best = state.get("best")
        logger.info("[best] resumed: best %s = %s", self.metric, self.best)

    def _ckpt(self):
        return self.checkpointer or getattr(self.trainer, "checkpointer", None)

    def _better(self, value):
        return self.best is None or self.sign * (value - self.best) > 0

    def before_train(self):
        # The periodic checkpoint a run resumes from can predate the last
        # <name>.pth, so the record written next to that file is the authority
        # on what "best" is.  Only on a resume: a fresh run never inherits a
        # number from whatever was in the directory before.
        ckpt = self._ckpt()
        if (not comm.is_main_process() or ckpt is None
                or getattr(self.trainer, "start_iter", 0) == 0):
            return
        side = os.path.join(ckpt.save_dir, self.name + ".json")
        if os.path.exists(side):
            with open(side) as f:
                rec = json.load(f)
            if rec.get("metric") == self.metric and self._better(float(rec["value"])):
                self.best = float(rec["value"])
                logger.info("[best] %s says best %s = %.4f (iter %s)", side,
                            self.metric, self.best, rec.get("iteration"))

    def _consume(self):
        try:
            n = len(self.trainer.storage.history(self.metric).values())
        except KeyError:
            return
        if n <= self._seen:
            return
        self._seen = n
        value = _get_metric(self.trainer.storage, self.metric)
        if value is None or not self._better(value):
            return
        self.best = value
        ckpt = self._ckpt()
        if ckpt is None:
            logger.warning("[best] no checkpointer; not saving")
            return
        extra = {"best_%s" % self.metric.replace("/", "_"): value}
        if self.state_fn is not None:
            path = os.path.join(ckpt.save_dir, self.name + ".pth")
            torch.save(dict({"model": self.state_fn(), "iteration": int(self.trainer.iter)}, **extra),
                       path + ".tmp")
            os.replace(path + ".tmp", path)
        else:
            ckpt.save(self.name, iteration=self.trainer.iter, **extra)
        rec = {"metric": self.metric, "value": value, "iteration": int(self.trainer.iter)}
        also = {k: _get_metric(self.trainer.storage, k) for k in self.record}
        also = {k: v for k, v in also.items() if v is not None}
        if also:
            rec["also"] = also
        with open(os.path.join(ckpt.save_dir, self.name + ".json"), "w") as f:
            json.dump(rec, f)
        logger.info("[best] iter %d: %s %.4f -- saved %s.pth",
                    self.trainer.iter, self.metric, value, self.name)

    def after_step(self):
        if comm.is_main_process():
            self._consume()

    def after_train(self):
        if not comm.is_main_process():
            return
        # EvalHook runs the evaluation at max_iter in ITS after_train, which
        # comes before this one; without this it could never become the best.
        self._consume()
        if self.best is not None:
            logger.info("[best] final: best %s = %.4f", self.metric, self.best)


class RuleBestCheckpointHook(HookBase):
    """``model_best_h.pth``: the validation the selection rule picks among all
    validations so far (wishnet/evaluation/selection.py: drop those whose F1 is
    more than F1_TOL below the highest, then the lowest best-F1 MAE).

    A later validation can raise the highest F1 and so hand the choice back to
    an earlier one, so every validation that could still be chosen (the front)
    keeps its weights, model only, as ``model_best_h_<iter>.pth`` until it no
    longer can.  model_best.pth and the periodic checkpoints are untouched, and
    ``last_checkpoint`` is never re-pointed.

    The record of every validation is ``model_best_h.json``, rewritten after
    each one; on a resume it is the hook's state, as model_best.json is for
    BestCheckpointHook.
    """

    PREFIX = "bdh/"
    NAME = "model_best_h"

    def __init__(self, state_fn=None):
        from wishnet.evaluation import selection
        self.sel = selection
        self.records = []       # {"iteration", "f1", "mae", "file"}, in validation order
        self._seen = 0
        self._dir = None
        # SOLVER.EMA: () -> the averaged state_dict, saved instead of the model's
        self.state_fn = state_fn

    def _path(self, name):
        return os.path.join(self._dir, name)

    def before_train(self):
        if not comm.is_main_process():
            return
        ck = getattr(self.trainer, "checkpointer", None)
        self._dir = getattr(ck, "save_dir", None)
        if not self._dir or getattr(self.trainer, "start_iter", 0) == 0:
            return              # a fresh run never inherits a record
        side = self._path(self.NAME + ".json")
        if os.path.exists(side):
            with open(side) as f:
                self.records = json.load(f).get("records", [])
            for r in self.records:
                if r.get("file") and not os.path.exists(self._path(r["file"])):
                    logger.warning("[best_h] %s is gone; iteration %d can no longer be chosen",
                                   r["file"], r["iteration"])
                    r["file"] = None
            logger.info("[best_h] resumed: %d validations, chosen %s", len(self.records),
                        self._chosen_text())

    # a candidate is (position in self.records, f1, mae): a resumed run can
    # validate the same iteration twice (once on the abandoned branch)
    def _cands(self):
        return [(i, r["f1"], r["mae"]) for i, r in enumerate(self.records)]

    def _chosen_text(self):
        c = self.sel.choose(self._cands())
        if c is None:
            return "none"
        return "iteration %d (F1 %.2f, MAE %.3f)" % (self.records[c[0]]["iteration"], c[1], c[2])

    def _consume(self):
        st = self.trainer.storage
        f1_key, mae_key = self.PREFIX + self.sel.F1_KEY, self.PREFIX + self.sel.MAE_KEY
        try:
            n = len(st.history(f1_key).values())
        except KeyError:
            return
        if n <= self._seen or not self._dir:
            return
        self._seen = n
        f1, mae = _get_metric(st, f1_key), _get_metric(st, mae_key)
        it = int(self.trainer.iter)
        new = {"iteration": it, "f1": f1, "mae": mae, "file": None}
        self.records.append(new)

        cands = self._cands()
        win = self.sel.choose(cands)
        keep = {c[0]: (c is win) for c in self.sel.front(cands)}
        iters = [r["iteration"] for r in self.records]

        def target(i):
            if keep[i]:
                return self.NAME + ".pth"
            r = self.records[i]
            dup = "_%d" % i if iters.count(r["iteration"]) > 1 else ""
            return "%s_%07d%s.pth" % (self.NAME, r["iteration"], dup)

        # weights that can no longer be chosen go; the rest move to their names
        # through temporary ones, so that no file overwrites another
        for i, r in enumerate(self.records):
            if r["file"] and i not in keep:
                os.remove(self._path(r["file"]))
                r["file"] = None
        moving = [i for i, r in enumerate(self.records) if r["file"] and r["file"] != target(i)]
        for i in moving:
            os.replace(self._path(self.records[i]["file"]), self._path(target(i) + ".mv"))
        for i in moving:
            os.replace(self._path(target(i) + ".mv"), self._path(target(i)))
            self.records[i]["file"] = target(i)
        last = len(self.records) - 1
        if last in keep:
            model = self.trainer.model
            if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
                model = model.module
            tmp = self._path(target(last) + ".tmp")
            # model only: read with MODEL.WEIGHTS, never resumed from
            torch.save({"model": model.state_dict() if self.state_fn is None else self.state_fn(),
                        "iteration": it, "f1": f1, "mae": mae}, tmp)
            os.replace(tmp, self._path(target(last)))
            new["file"] = target(last)

        side = self._path(self.NAME + ".json")
        with open(side + ".tmp", "w") as f:
            json.dump({"rule": "drop F1 more than %g below the highest, then the lowest %s"
                               % (self.sel.F1_TOL, self.sel.MAE_KEY),
                       "chosen": None if win is None else self.records[win[0]]["iteration"],
                       "records": self.records}, f, indent=1)
        os.replace(side + ".tmp", side)
        logger.info("[best_h] iter %d: F1 %s, MAE %s -> chosen %s; kept %s", it,
                    "n/a" if f1 is None else "%.2f" % f1, "n/a" if mae is None else "%.3f" % mae,
                    self._chosen_text(), sorted(self.records[i]["iteration"] for i in keep) or "none")

    def after_step(self):
        if comm.is_main_process():
            self._consume()

    def after_train(self):
        if comm.is_main_process():
            self._consume()         # the evaluation at max_iter, as in BestCheckpointHook


class StopFileHook(HookBase):
    """A file named STOP in the run directory ends the training after the
    current step, as if MAX_ITER had been that step + 1: after_train then runs
    as at the planned end -- the final evaluation, model_final (with SOLVER.EMA
    the average, the training weights as model_raw), the best-checkpoint hooks.

    Checked after every step; the trainer's max_iter and the periodic
    checkpointer's copy of it are lowered.  The hook goes before EvalHook and
    the periodic checkpointer, which see the lowered max_iter at the same step:
    no extra periodic evaluation, and model_final is written at that step.  The
    file is renamed STOP.done, so that a resume does not stop again at once.
    Without the file nothing changes."""

    def __init__(self, run_dir, name="STOP"):
        self.path = os.path.join(os.path.abspath(run_dir), name)

    def after_step(self):
        stop = comm.is_main_process() and os.path.exists(self.path)
        if comm.get_world_size() > 1:
            stop = any(comm.all_gather(stop))
        if not stop:
            return
        n = min(self.trainer.max_iter, self.trainer.iter + 1)
        self.trainer.max_iter = n
        for h in getattr(self.trainer, "_hooks", []):
            if isinstance(h, PeriodicCheckpointer):
                h.max_iter = n
        self.trainer.storage.put_scalar("stop_file/iter", n, smoothing_hint=False)
        logger.info("[stop] stopped at %d by STOP (%s)", n, self.path)
        if not comm.is_main_process():
            return
        try:
            os.replace(self.path, self.path + ".done")
        except OSError:
            logger.exception("[stop] could not rename %s", self.path)
