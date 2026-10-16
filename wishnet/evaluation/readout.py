# -*- coding: utf-8 -*-
"""The optional fixed-set readout after every validation (ValidationReadout):
on fixed training and validation tiles, from one decoding of their
ground-truth buildings.  A monitor: the training computation is the same with
or without it.

* the fixed-set heights (MODEL.WISHNET.FIXED_SET_HEIGHT): the height MAE of the
  ground-truth buildings decoded as queries with the model as it is, on the
  training tiles and on the validation tiles, the gap between the two and the
  validation buildings of 50 m and more.  The set of buildings never changes,
  so the numbers are free of the recall that moves the detection-based MAEs.
  Scalars fixed/train_mae, fixed/val_mae, fixed/gap, fixed/val_mae_50 (and the
  counts fixed/*_n*), with BANDS also per height band; every reading is
  appended to <OUTPUT_DIR>/fixed_set_height.json.  A model trained with
  MODEL.WISHNET.OPTICAL_PASS also gets fixed/train_mae_optpass and
  fixed/val_mae_optpass: the same buildings decoded with every SAR step
  skipped, the state that pass trains;
* the SAR readout (MODEL.WISHNET.SAR_MEAN_READOUT, mode fusion), described below;
* the detection readings (FIXED_SET_HEIGHT.DETECTION): from the same
  decodings, the proposal recall, F1 and AP50 of the model as it is and their
  changes in the other SAR states (fixdet/*, appended to
  <OUTPUT_DIR>/fixed_set_detection.json).

With MODEL.WISHNET.ENCODER_EXCHANGE every SAR state below -- zeroed, mean,
GF-3 or InSAR alone zeroed -- goes through the encoder exchange as well, so it
reaches the optical memory too, and the optical pass's state decodes the
optical memory before the exchange.

The SAR readout: how much of a fusion model's heights comes from the SAR
content, on training tiles and on validation tiles alike.  Every tile's
ground-truth buildings that are not truncated and have a height of at least
INPUT.MIN_HEIGHT are decoded as queries (WISHNet.gtq_sar_readout) with

    real   the tile's SAR;
    mean   the SAR memory replaced by its position-wise mean over the readout's
           training tiles (the SAR's level stays, the per-building content goes);
    zero   a zero SAR input;
    zero_gf3, zero_ins
           (SAR_MEAN_READOUT.PER_MODALITY, with INSAR) GF-3 alone or the InSAR
           input alone zeroed.

Per set, for x = mean, zero (zero_gf3, zero_ins):

    mae        mean |h_real - h|                       (ground-truth-box height MAE)
    dmae       mean |h_x - h| - mae                    (what losing the SAR content costs)
    shift      mean (h_x - h_real)                     (the overall change of the heights)
    dmae_cal   dmae after each set of predictions has its own L1 affine fit to the
               ground truth -- the overall shift removed, the per-building
               information kept

and ratio_cal = train dmae_cal / val dmae_cal: how much more the SAR content is
worth on the tiles the model trained on than on new ones.  Scalars
sar_mean/... and sar_zero/...; every reading is also appended to
<OUTPUT_DIR>/sar_mean_readout.json, which lists the tiles.

The tiles are fixed: every n-th of the dataset's tiles sorted by file name (n
from SAR_MEAN_READOUT.TRAIN_TILES / VAL_TILES, 0 = all), the same for every
run on the same data.  The training tiles are read whole, without jitter or
flips, their annotations clipped and flagged as in training.

The detection record of one image (det_record), from the decoding's matching
queries (they cannot see the ground-truth ones, so these are the image's
inference outputs):

    prop_hit   per ground-truth roof that is not truncated: does one of the
               encoder proposals' masks (the proposal chain) reach IoU 0.5
    scores     per query, as instance_inference scores it: the class
               probability times the mean mask probability inside the mask
    status     per query, matched greedily by score at IoU 0.5 to the roofs:
               1 a hit, 0 a false detection, -1 ignored (it only reaches
               truncated roofs, which count as ignore regions)

all on the stride-4 grid of the mask logits (the roofs area-averaged onto it).
"""
import copy
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.data import DatasetCatalog
from detectron2.evaluation.evaluator import inference_context

from ..data.mapper import WISHDatasetMapper

logger = logging.getLogger(__name__)

# the SAR states whose heights are compared with the real ones: the mean
# replacement, both zeroed, and (SAR_MEAN_READOUT.PER_MODALITY) GF-3 alone and
# InSAR alone zeroed
STATES = ("mean", "zero", "zero_gf3", "zero_ins")
# the validation buildings counted as tall in the fixed-set heights (m)
TALL_M = 50
# FIXED_SET_HEIGHT.BANDS: the height bands of the ground truth, m, [lo, hi)
HEIGHT_BANDS = ((3, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, None))
# the detection readings: IoU of a hit, the metrics per set of tiles
IOU = 0.5
METRICS = ("prop_recall", "f1", "ap50")


# ------------------------------------------------------------ the detections
def iou_matrix(a, b):
    """a (P, N), b (G, N) float masks of 0 / 1 -> (P, G) IoU."""
    inter = a @ b.t()
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return inter / union.clamp(min=1.0)


def to_grid(g, h, w, chunk=64):
    """(n, H, W) bool masks of the input -> (n, h * w) float 0 / 1 on the stride-4
    grid (h, w): a cell is in the mask when at least half of its 4 x 4 pixels
    are; the input padded with zeros to (4h, 4w).  ``chunk`` masks at a time."""
    out = []
    for i in range(0, int(g.shape[0]), chunk):
        x = g[i:i + chunk].to(torch.uint8)[:, :4 * h, :4 * w]
        if x.shape[-2] < 4 * h or x.shape[-1] < 4 * w:
            x = F.pad(x, (0, 4 * w - x.shape[-1], 0, 4 * h - x.shape[-2]))
        s = x.reshape(x.shape[0], h, 4, w, 4).sum(dim=(2, 4), dtype=torch.int16)
        out.append((s >= 8).flatten(1).float())
    return torch.cat(out, 0)


@torch.no_grad()
def det_record(dd, gt_masks, truncated, iou=IOU):
    """dd {"logits" (Q, classes), "masks" (Q, h, w), "prop" (Q, h, w)} of one image
    (WISHDecoder.forward_gt_queries with detection); gt_masks (n, H, W) bool
    roofs of the input; truncated (n,) -> the record (numpy, see the module)."""
    masks = dd["masks"].float()
    h, w = masks.shape[-2:]
    dev = masks.device
    g = gt_masks.to(dev)
    n = int(g.shape[0])
    trunc = truncated.bool().cpu().numpy() if n else np.zeros(0, dtype=bool)
    gb = to_grid(g, h, w) if n else None
    mb = (masks > 0).flatten(1).float()
    mask_score = (masks.sigmoid().flatten(1) * mb).sum(1) / (mb.sum(1) + 1e-6)
    scores = (dd["logits"].float().sigmoid().max(-1).values * mask_score).cpu().numpy()
    Q = len(scores)
    status = np.zeros(Q, dtype=np.int8)
    if not n:
        return {"prop_hit": np.zeros(0, dtype=bool), "scores": scores, "status": status, "n_gt": 0}
    pb = (dd["prop"] > 0).flatten(1).float()
    prop_hit = (iou_matrix(pb, gb).max(0).values >= iou).cpu().numpy()[~trunc]
    ious = iou_matrix(mb, gb).cpu().numpy()
    hit = ious >= iou
    valid = ~trunc
    matched = np.zeros(n, dtype=bool)
    for q in np.argsort(-scores, kind="stable"):
        row = hit[q]
        if not row.any():
            continue
        cand = np.flatnonzero(row & valid & ~matched)
        if len(cand):
            matched[cand[np.argmax(ious[q, cand])]] = True
            status[q] = 1
        elif (row & trunc).any():
            status[q] = -1
    return {"prop_hit": prop_hit, "scores": scores, "status": status, "n_gt": int(valid.sum())}


def det_metrics(recs):
    """The records of a set of tiles, pooled -> {"n_gt", "prop_recall", "f1", "ap50"}."""
    n_gt = int(sum(r["n_gt"] for r in recs))
    hits = int(sum(int(r["prop_hit"].sum()) for r in recs))
    s = np.concatenate([r["scores"] for r in recs]) if recs else np.zeros(0)
    st = np.concatenate([r["status"] for r in recs]) if recs else np.zeros(0, dtype=np.int8)
    keep = st >= 0
    s, tp = s[keep], st[keep] == 1
    tp = tp[np.argsort(-s, kind="mergesort")]
    out = {"n_gt": n_gt, "prop_recall": hits / max(n_gt, 1), "f1": 0.0, "ap50": 0.0}
    if len(tp) and n_gt:
        ctp = np.cumsum(tp)
        cfp = np.cumsum(~tp)
        rec = ctp / n_gt
        prec = ctp / np.maximum(ctp + cfp, 1)
        out["f1"] = float((2 * prec * rec / np.maximum(prec + rec, 1e-12)).max())
        env = np.maximum.accumulate(prec[::-1])[::-1]
        idx = np.searchsorted(rec, np.linspace(0.0, 1.0, 101), side="left")
        out["ap50"] = float(np.mean([env[i] if i < len(env) else 0.0 for i in idx]))
    return out


def fixed_detection(dets):
    """{"train": {state: [records]}, "val": {...}} -> the readings
    <part>_n_gt, <part>_<metric> of "real" and <part>_<metric>_d_<state> =
    state - real for every other state decoded."""
    out = {}
    for part, by in dets.items():
        if not by.get("real"):
            continue
        base = det_metrics(by["real"])
        out["%s_n_gt" % part] = base["n_gt"]
        for m in METRICS:
            out["%s_%s" % (part, m)] = base[m]
        for name, recs in by.items():
            if name == "real" or not recs:
                continue
            mm = det_metrics(recs)
            for m in METRICS:
                out["%s_%s_d_%s" % (part, m, name)] = mm[m] - base[m]
    return out


# --------------------------------------------------------------- the heights
def pick_tiles(dicts, n):
    """Every n-th of ``dicts`` sorted by file name (all for n <= 0 or n >= len)."""
    order = sorted(range(len(dicts)), key=lambda i: dicts[i]["file_name"])
    if n <= 0 or n >= len(order):
        return [dicts[i] for i in order]
    return [dicts[order[i]] for i in np.unique(np.linspace(0, len(order) - 1, n).round().astype(int))]


class WholeTrainingTile:
    """A training tile as a whole: the training mapper without jitter, flips or
    speckle resampling (its window of a 2048 tile with a 2048 window is the
    tile), with the global numpy random state left as it was."""

    def __init__(self, cfg):
        kw = WISHDatasetMapper.from_config(cfg, True)
        kw.update(brightness=0.0, contrast=0.0, geo_aug="", crop_retry=1, speckle=None)
        self.mapper = WISHDatasetMapper(**kw)

    def __call__(self, d):
        state = np.random.get_state()
        try:
            return self.mapper(copy.deepcopy(d))
        finally:
            np.random.set_state(state)


def l1_affine(h, y, iters=50):
    """(a, b) minimising sum |y - (a h + b)|, by iteratively reweighted least squares."""
    A = np.stack([h, np.ones_like(h)], 1)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(iters):
        w = 1.0 / np.maximum(np.abs(y - A @ coef), 1e-3)
        Aw = A * w[:, None]
        coef = np.linalg.solve(A.T @ Aw, Aw.T @ y)
    return coef


def summarize(rec):
    """rec: {"gt", "real", [one of STATES, ...]} arrays of the kept buildings -> readings."""
    y = rec["gt"]
    out = {"n": int(len(y))}
    if len(y) < 10:
        return out
    real = rec["real"]
    out["mae"] = float(np.abs(real - y).mean())
    a, b = l1_affine(real, y)
    mae_cal = float(np.abs(a * real + b - y).mean())
    for k in STATES:
        if k not in rec:
            continue
        h = rec[k]
        a2, b2 = l1_affine(h, y)
        out[k] = {"dmae": float(np.abs(h - y).mean()) - out["mae"],
                  "shift": float((h - real).mean()),
                  "dmae_cal": float(np.abs(a2 * h + b2 - y).mean()) - mae_cal}
    return out


def scalars(res):
    """{"train": summary, "val": summary} -> the event-storage scalars."""
    s = {}
    for part in ("train", "val"):
        r = res.get(part, {})
        s["sar_mean/%s_n" % part] = r.get("n", 0)
        if "mae" in r:
            s["sar_mean/%s_mae" % part] = r["mae"]
        for k in STATES:
            for q, v in r.get(k, {}).items():
                s["sar_%s/%s_%s" % (k, part, q)] = v
    for k in STATES:
        t = res.get("train", {}).get(k, {}).get("dmae_cal")
        v = res.get("val", {}).get(k, {}).get("dmae_cal")
        if t is not None and v is not None and abs(v) > 1e-6:
            s["sar_%s/ratio_cal" % k] = t / v
    return s


def band_readings(y, h, part, suffix=""):
    """Per band of the ground truth ``y`` (HEIGHT_BANDS): the buildings, the MAE
    and the signed bias (mean of h - y) of the heights ``h`` ->
    {<part>_n_b<lo>_<hi>, <part>_mae<suffix>_b<lo>_<hi>, <part>_bias<suffix>_b<lo>_<hi>};
    the count only without ``suffix``, no MAE or bias for an empty band."""
    out = {}
    for lo, hi in HEIGHT_BANDS:
        m = y >= lo
        if hi is not None:
            m &= y < hi
        tag = "b%d_%s" % (lo, "inf" if hi is None else hi)
        if not suffix:
            out["%s_n_%s" % (part, tag)] = int(m.sum())
        if m.any():
            d = h[m] - y[m]
            out["%s_mae%s_%s" % (part, suffix, tag)] = float(np.abs(d).mean())
            out["%s_bias%s_%s" % (part, suffix, tag)] = float(d.mean())
    return out


def fixed_heights(cols, tall=TALL_M, bands=False):
    """{"train": cols, "val": cols} (each {"gt", "real", ...} over the kept
    buildings) -> the fixed-set readings: per set n and the height MAE of the
    ground-truth-box queries, the gap val - train, and the validation buildings
    of ``tall`` m and more; with ``bands`` the same per height band as well
    (band_readings)."""
    out = {}
    for part in ("train", "val"):
        y, h = cols[part]["gt"], cols[part]["real"]
        out["%s_n" % part] = int(len(y))
        if len(y):
            out["%s_mae" % part] = float(np.abs(h - y).mean())
            if "opt" in cols[part]:
                # MODEL.WISHNET.OPTICAL_PASS: the model's own optical part
                out["%s_mae_optpass" % part] = float(np.abs(cols[part]["opt"] - y).mean())
    y, h = cols["val"]["gt"], cols["val"]["real"]
    t = y >= tall
    out["val_n_%d" % tall] = int(t.sum())
    if t.any():
        out["val_mae_%d" % tall] = float(np.abs(h[t] - y[t]).mean())
    if "train_mae" in out and "val_mae" in out:
        out["gap"] = out["val_mae"] - out["train_mae"]
    if bands:
        for part in ("train", "val"):
            c = cols[part]
            out.update(band_readings(c["gt"], c["real"], part))
            if "opt" in c:
                out.update(band_readings(c["gt"], c["opt"], part, "_optpass"))
    return out


class _KeepRandomState:
    """A mapper called with the global numpy and Python random states left as
    they were (the readout must not move the training's random numbers)."""

    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, d):
        np_state, py_state = np.random.get_state(), random.getstate()
        try:
            return self.mapper(copy.deepcopy(d))
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)


class ValidationReadout:
    """The fixed-set heights (FIXED_SET_HEIGHT) and, in mode fusion, the SAR
    readout (SAR_MEAN_READOUT) of a model, on the fixed tiles of
    SAR_MEAN_READOUT.TRAIN_TILES / VAL_TILES, from one decoding: the "real"
    heights of the SAR readout are the fixed-set heights."""

    def __init__(self, cfg, out_dir=None):
        g = cfg.MODEL.WISHNET
        r = g.SAR_MEAN_READOUT
        self.fixed = bool(g.FIXED_SET_HEIGHT.ENABLED)
        self.bands = bool(g.FIXED_SET_HEIGHT.BANDS)
        self.sar_mean = g.MODE == "fusion" and bool(r.ENABLED)
        self.with_zero = self.sar_mean and bool(r.WITH_ZERO)
        # SAR_MEAN_READOUT.PER_MODALITY: GF-3 alone and InSAR alone zeroed as well
        self.per_modality = self.sar_mean and bool(r.PER_MODALITY)
        # MODEL.WISHNET.OPTICAL_PASS: the decoding without SAR steps as well
        self.optpass = g.MODE == "fusion" and float(g.OPTICAL_PASS.WEIGHT) > 0
        # FIXED_SET_HEIGHT.DETECTION: the detection readings of the same decodings,
        # at the validations every DETECTION_PERIOD iterations (0: all) and the last
        self.detection = bool(g.FIXED_SET_HEIGHT.DETECTION)
        self.det_period = int(g.FIXED_SET_HEIGHT.DETECTION_PERIOD)
        self.max_iter = int(cfg.SOLVER.MAX_ITER)
        self.min_height = float(cfg.INPUT.MIN_HEIGHT)
        self.train = pick_tiles(DatasetCatalog.get(cfg.DATASETS.TRAIN[0]), int(r.TRAIN_TILES))
        self.val = pick_tiles(DatasetCatalog.get(cfg.DATASETS.TEST[0]), int(r.VAL_TILES))
        self.train_mapper = WholeTrainingTile(cfg)
        self.val_mapper = _KeepRandomState(WISHDatasetMapper(cfg, False))
        out = out_dir or cfg.OUTPUT_DIR
        self.path = os.path.join(out, "sar_mean_readout.json")
        self.fixed_path = os.path.join(out, "fixed_set_height.json")
        self.det_path = os.path.join(out, "fixed_set_detection.json")
        logger.info("[readout] %d training and %d validation tiles (%s, %s): %s", len(self.train),
                    len(self.val), cfg.DATASETS.TRAIN[0], cfg.DATASETS.TEST[0],
                    " + ".join(n for n, on in (("fixed-set heights", self.fixed),
                                                ("SAR readout", self.sar_mean)) if on))

    def _mean_memory(self, model):
        sums, n = None, 0
        for d in self.train:
            maps = model.sar_memory_maps([self.train_mapper(d)])
            maps = [m.float().sum(0) for m in maps]
            sums = maps if sums is None else [a + b for a, b in zip(sums, maps)]
            n += 1
        return [m / max(n, 1) for m in sums]

    def _read(self, model, tiles, mapper, mean_maps, detection=False):
        """-> the heights per state over the kept buildings, and with
        ``detection`` the detection records per state ({state: [record per tile]})."""
        keys = (["gt", "real"] + (["mean"] if mean_maps is not None else [])
                + (["zero"] if self.with_zero else []) + (["zero_gf3", "zero_ins"] if self.per_modality else [])
                + (["opt"] if self.optpass else []))
        cols = {k: [] for k in keys}
        dets = {k: [] for k in keys[1:]}
        for d in tiles:
            for r in model.gtq_sar_readout([mapper(d)], mean_maps, self.with_zero, self.optpass,
                                           self.per_modality, detection):
                keep = (~r["truncated"]) & (r["gt"] >= self.min_height)
                for k in keys:
                    cols[k].append(r[k][keep].numpy().astype(np.float64))
                    if detection and k != "gt":
                        dets[k].append(r["det_" + k])
        return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}, dets

    def detection_due(self, iteration):
        """FIXED_SET_HEIGHT.DETECTION at the validation after ``iteration``."""
        if not self.detection:
            return False
        return (self.det_period <= 0 or iteration is None or (int(iteration) + 1) % self.det_period == 0
                or int(iteration) + 1 >= self.max_iter)

    @torch.no_grad()
    def run(self, model, iteration=None):
        """The readouts of ``model`` as it is (the caller swaps in the moving
        average) -> the scalars; the records go into fixed_set_height.json,
        sar_mean_readout.json and fixed_set_detection.json."""
        model = getattr(model, "module", model)
        det = self.detection_due(iteration)
        with inference_context(model):
            mean_maps = self._mean_memory(model) if self.sar_mean else None
            cols, dets = {}, {}
            for part, tiles, mapper in (("train", self.train, self.train_mapper), ("val", self.val, self.val_mapper)):
                cols[part], dets[part] = self._read(model, tiles, mapper, mean_maps, det)
        del mean_maps
        torch.cuda.empty_cache()
        values = {}
        if det:
            fd = fixed_detection(dets)
            self._write(self.det_path, iteration, fd)
            values.update({"fixdet/" + k: v for k, v in fd.items()})
        if self.sar_mean:
            res = {part: summarize(c) for part, c in cols.items()}
            self._write(self.path, iteration, res)
            values.update(scalars(res))
        if self.fixed:
            fx = fixed_heights(cols, bands=self.bands)
            self._write(self.fixed_path, iteration, fx)
            values.update({"fixed/" + k: v for k, v in fx.items()})
        return values

    def _write(self, path, iteration, res):
        doc = {"tiles": {"train": [os.path.basename(d["file_name"]) for d in self.train],
                         "val": [os.path.basename(d["file_name"]) for d in self.val]},
               "readings": []}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    old = json.load(f)
                if old.get("tiles") == doc["tiles"]:
                    doc["readings"] = old.get("readings", [])
            except Exception:
                pass
        doc["readings"].append({"iteration": iteration, **res})
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w") as f:
            json.dump(doc, f, indent=1)
        os.replace(path + ".tmp", path)
