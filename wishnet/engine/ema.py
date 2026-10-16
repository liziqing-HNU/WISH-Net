# -*- coding: utf-8 -*-
"""Exponential moving average of the model's weights (SOLVER.EMA).

Every parameter of the model is averaged, frozen ones included; buffers are
not (the model's floating buffers are constants: the optical backbone's frozen
batch-norm statistics).  After every optimiser step, n counting the updates
before this one,

    d_n = min(DECAY, (1 + n) / (WARMUP + n))        (WARMUP 0: d_n = DECAY)
    ema <- d_n * ema + (1 - d_n) * weights

-- the ramp of d2go and timm: early on the average covers about the last
WARMUP-th part of the steps instead of dragging the initialisation along; with
WARMUP 10 it reaches 0.9997 at 28k steps (a time constant of about 3.5k steps)
and DECAY is its cap.

The average is a checkpointable of the trainer's checkpointer (``ema``: the
weights and n), so a resumed run continues the same average.  ``swapped``
puts it into the model for an evaluation or a save and restores the training
weights bit for bit afterwards.

MaskDINO holds some parameters under several names (decoder_norm =
decoder.norm, _bbox_embed = bbox_embed.* = decoder.bbox_embed.*).  The average
is kept once per parameter, by its named_parameters name; every state_dict the
module writes (``model_state``, model_final's "model_raw") carries every name,
matched by the parameter object, so a load can never put a training weight
under one name over the average under another, or the reverse.  ``EMACheckpointer`` writes model_final with
the average as its "model" and the training weights as "model_raw", and a
resume (only a resume) takes "model_raw" back; every other reader of
model_final -- evaluation, the selection of weights, the diagnostics -- gets
the average.
"""
import contextlib
import logging

import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.train_loop import HookBase

logger = logging.getLogger("wishnet.engine")


def _unwrap(model):
    return model.module if isinstance(model, (torch.nn.parallel.DistributedDataParallel,
                                              torch.nn.DataParallel)) else model


class ModelEMA:
    def __init__(self, model, decay=0.9998, warmup=10.0):
        if not 0.0 < float(decay) < 1.0:
            raise ValueError("SOLVER.EMA.DECAY must be in (0, 1)")
        self.model = _unwrap(model)
        self.decay = float(decay)
        self.warmup = float(warmup)
        self.updates = 0
        self.weights = {}           # parameter name -> its average
        self.loaded = False         # set by load_state_dict: a resumed run

    def current_decay(self):
        n = float(self.updates)
        if self.warmup <= 0:
            return self.decay
        return min(self.decay, (1.0 + n) / (self.warmup + n))

    @torch.no_grad()
    def init_from_model(self):
        self.weights = {k: p.detach().clone() for k, p in self.model.named_parameters()}
        self._lists = None
        self.updates = 0

    def _pairs(self):
        """The averages and the parameters as two aligned lists, built once (the
        parameters are updated in place, so the tensors stay the same)."""
        if getattr(self, "_lists", None) is None:
            params = dict(self.model.named_parameters())
            names = list(self.weights)
            self._lists = ([self.weights[k] for k in names], [params[k].detach() for k in names])
        return self._lists

    @torch.no_grad()
    def update(self):
        d = self.current_decay()
        avg, cur = self._pairs()
        torch._foreach_mul_(avg, d)
        torch._foreach_add_(avg, cur, alpha=1.0 - d)
        self.updates += 1
        return d

    @torch.no_grad()
    def rel_dist(self):
        """||average - weights|| / ||weights|| over all parameters."""
        params = dict(self.model.named_parameters())
        names = list(self.weights)
        diff = torch._foreach_sub([self.weights[k] for k in names], [params[k].detach() for k in names])
        num = torch.stack(torch._foreach_norm(diff)).float().pow(2).sum().sqrt()
        den = torch.stack(torch._foreach_norm([params[k].detach() for k in names])).float().pow(2).sum().sqrt()
        return float(num / den.clamp(min=1e-12))

    def _state_names(self):
        """state_dict name -> the parameter name it holds.  MaskDINO puts one
        Parameter under several names (decoder_norm = decoder.norm, _bbox_embed
        = bbox_embed.* = decoder.bbox_embed.*): named_parameters lists each
        once, state_dict every name, so the names are matched by the object."""
        if getattr(self, "_names", None) is None:
            ids = {id(p): k for k, p in self.model.named_parameters()}
            self._names = {k: ids[id(v)] for k, v in self.model.state_dict(keep_vars=True).items()
                           if id(v) in ids}
        return self._names

    def model_state(self):
        """The model's state_dict with the averaged parameters under every name
        they have (the buffers as they are)."""
        if not self.weights:
            raise RuntimeError("SOLVER.EMA: there is no average yet -- it starts at the training weights "
                               "in EMAHook.before_train, or comes from the checkpoint a run resumes from")
        sd = self.model.state_dict()
        names = self._state_names()
        return {k: (self.weights[names[k]] if k in names else v) for k, v in sd.items()}

    @contextlib.contextmanager
    def swapped(self):
        """The average in the model inside the block; yields the training
        parameters (name -> copy), which are put back bit for bit afterwards."""
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            raw = {k: params[k].detach().clone() for k in self.weights}
            for k, v in self.weights.items():
                params[k].data.copy_(v)
        try:
            yield raw
        finally:
            with torch.no_grad():
                for k, v in raw.items():
                    params[k].data.copy_(v)

    # -- checkpointable --------------------------------------------------
    def state_dict(self):
        return {"weights": self.weights, "updates": int(self.updates), "decay": self.decay,
                "warmup": self.warmup}

    def load_state_dict(self, state):
        params = dict(self.model.named_parameters())
        missing = sorted(set(params) - set(state["weights"]))
        if missing:
            raise KeyError("the checkpoint's EMA lacks %d parameters, e.g. %s" % (len(missing), missing[:3]))
        self.weights = {k: v.to(device=params[k].device, dtype=params[k].dtype).clone()
                        for k, v in state["weights"].items() if k in params}
        self._lists = None
        self.updates = int(state["updates"])
        self.loaded = True
        logger.info("[EMA] resumed: %d updates, decay now %.6f", self.updates, self.current_decay())


class EMAHook(HookBase):
    """Starts the average at the training weights (unless a resume loaded it),
    updates it after every optimiser step and logs ema/decay (every step) and
    ema/rel_dist (every ``log_period`` steps).  Goes before EvalHook, so an
    evaluation sees the average including its own step."""

    def __init__(self, ema, log_period=100):
        self.ema = ema
        self.log_period = int(log_period)

    def before_train(self):
        if not self.ema.loaded:
            self.ema.init_from_model()
            logger.info("[EMA] started at the weights of iteration %d (decay %.6g, warm-up %g)",
                        self.trainer.start_iter, self.ema.decay, self.ema.warmup)

    def after_step(self):
        d = self.ema.update()
        st = self.trainer.storage
        st.put_scalar("ema/decay", d, smoothing_hint=False)
        if self.log_period > 0 and (self.trainer.iter + 1) % self.log_period == 0:
            st.put_scalar("ema/rel_dist", self.ema.rel_dist(), smoothing_hint=False)


class EMACheckpointer(DetectionCheckpointer):
    """DetectionCheckpointer for runs with SOLVER.EMA: model_final holds the
    average as "model" (what every reader of the file evaluates) and the
    training weights as "model_raw"; a resume loads "model_raw" into the model.
    The average itself is the checkpointable "ema" of every checkpoint."""

    def __init__(self, model, save_dir="", *, ema, **checkpointables):
        super().__init__(model, save_dir, **checkpointables)
        self.ema = ema
        self.checkpointables["ema"] = ema
        self._resuming = False

    def save(self, name, **kwargs):
        if name != "model_final":
            return super().save(name, **kwargs)
        # the training weights under every state_dict name (a shared parameter
        # has several), taken before the average goes in
        raw = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        with self.ema.swapped():
            return super().save(name, model_raw=raw, **kwargs)

    def load(self, path, checkpointables=None):
        # resume_or_load: a resume loads every checkpointable (None), a plain
        # MODEL.WEIGHTS load none ([])
        self._resuming = checkpointables is None
        try:
            return super().load(path, checkpointables)
        finally:
            self._resuming = False

    def _load_model(self, checkpoint):
        raw = checkpoint.pop("model_raw", None)
        if raw is not None and self._resuming:
            logger.info("[EMA] resuming from a model_final: the training weights (model_raw), not the average")
            checkpoint["model"] = raw
        return super()._load_model(checkpoint)
