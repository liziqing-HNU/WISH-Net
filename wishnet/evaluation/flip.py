# -*- coding: utf-8 -*-
"""Test-time up-down flip averaging of the heights (the paper's final
evaluation).

Every tile goes through the model twice: as it is, and flipped up-down with
the training mapper's transform (optical, GF-3, InSAR and geo_M, so a fusion
model's range reads turn with it, as in training).  The second pass is written
back to the original frame (masks flipped back, boxes y -> H - y, offsets
dy -> -dy, heights unchanged), and its detections are paired with the first
pass's one to one by mask IoU: the first pass's detections in order of score,
each with the free second-pass detection of the highest IoU, at least
``min_iou``; IoU on the masks sampled every 4th px.

The averaged prediction is the first pass with each paired detection's height
the mean of the two passes: detections, masks, boxes, scores and offsets those
of the first pass, so the F1, the mAP and the best-F1 threshold are the first
pass's and only the heights change.
"""
import copy

import numpy as np
import torch
from detectron2.structures import Boxes, Instances


def flipped(mapper, d):
    """The mapped evaluation sample flipped up-down, as a training sample that
    drew flip_ud (``mapper``: a WISHDatasetMapper)."""
    d1 = copy.deepcopy(d)
    mapper._geo(d1, d1["image"].shape[-1], "flip_ud")
    return d1


def write_back(inst, height):
    """Second-pass instances (flipped frame) -> the original frame, as a new
    Instances: masks flipped back, boxes y -> H - y, offsets dy -> -dy."""
    out = Instances(inst.image_size)
    out.pred_masks = inst.pred_masks.flip(-2)
    b = inst.pred_boxes.tensor.clone()
    b[:, 1], b[:, 3] = height - inst.pred_boxes.tensor[:, 3], height - inst.pred_boxes.tensor[:, 1]
    out.pred_boxes = Boxes(b)
    out.scores = inst.scores
    out.pred_classes = inst.pred_classes
    out.pred_heights = inst.pred_heights
    off = inst.pred_offsets.clone()
    off[:, 1] = -off[:, 1]
    out.pred_offsets = off
    return out


def pair(m1, m2, s1, min_iou=0.5, step=4):
    """One-to-one pairs of first-pass (masks m1, scores s1) and second-pass (m2)
    detections by mask IoU on every ``step``-th px -> (the matched second-pass
    index per first-pass detection or -1, its IoU)."""
    n1, n2 = len(m1), len(m2)
    match, best = np.full(n1, -1), np.zeros(n1)
    if n1 == 0 or n2 == 0:
        return match, best
    dt = torch.float16 if m1.is_cuda else torch.float32       # 0 / 1 values, half the memory
    a = m1[:, ::step, ::step].reshape(n1, -1).to(dt)
    b = m2[:, ::step, ::step].reshape(n2, -1).to(dt)
    inter = (a @ b.T).float()
    union = a.float().sum(1)[:, None] + b.float().sum(1)[None, :] - inter
    iou = (inter / union.clamp(min=1.0)).cpu().numpy()
    free = np.ones(n2, dtype=bool)
    for i in np.argsort(-s1.cpu().numpy()):
        cand = np.where(free, iou[i], -1.0)
        j = int(cand.argmax())
        if cand[j] >= min_iou:
            match[i], best[i], free[j] = j, cand[j], False
    return match, best


def averaged(inst1, inst2, match):
    """The first pass with each paired detection's height the mean of the two passes."""
    out = Instances(inst1.image_size)
    for k in ("pred_masks", "pred_boxes", "scores", "pred_classes", "pred_offsets"):
        out.set(k, inst1.get(k))
    h = inst1.pred_heights.clone()
    sel = torch.as_tensor(np.flatnonzero(match >= 0), device=h.device)
    if len(sel):
        other = torch.as_tensor(match[match >= 0], device=h.device)
        h[sel] = 0.5 * (inst1.pred_heights[sel] + inst2.pred_heights[other])
    out.pred_heights = h
    return out


def shallow(inst):
    """A new Instances over the same tensors: BDHEvaluator.process removes the
    masks of the Instances it is handed."""
    out = Instances(inst.image_size)
    for k, v in inst.get_fields().items():
        out.set(k, v)
    return out
