# -*- coding: utf-8 -*-
"""Checks of the up-down flip averaging (wishnet/evaluation/flip.py) on a
fake BDH tile and synthetic detections, CPU only, no model:

    flipped     a mapped validation sample flipped twice is the sample again;
                flipped once, every input and ground-truth mask is flipped up-down;
    write_back  detections of the flipped tile come back to the original frame
                (masks, boxes y -> H - y, offsets dy -> -dy, heights unchanged);
    pair        one-to-one pairs by mask IoU in the order of the first pass's
                scores, none below the threshold;
    averaged    the first pass's detections with the paired heights averaged.

    python tests/test_flip.py
"""
import argparse
import logging
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "equivalence"))

import numpy as np
import torch
from detectron2.data import DatasetCatalog
from detectron2.structures import Boxes, Instances

import train_net as T
from wishnet import WISHDatasetMapper
from wishnet.evaluation.flip import averaged, flipped, pair, shallow, write_back

SIZE = 256


def sample(root):
    """The first validation tile of a fake BDH, through the evaluation mapper."""
    import make_fake_bdh
    sys.argv = ["make_fake_bdh", root, "--size", str(SIZE), "--train", "1", "--val", "1"]
    make_fake_bdh.main()
    opts = ["DATASETS.BDH_ROOT", root, "DATASETS.BDH_PREPARED", os.path.join(root, "prepared_v2"),
            "MODEL.WISHNET.INSAR.ROOT", os.path.join(root, "s1_insar_asc"),
            "INPUT.CROP_SIZE", str(SIZE), "INPUT.EVAL_CROP_SIZE", str(SIZE),
            "MODEL.WISHNET.SAR.INPUT_PX", str(SIZE // 2), "MODEL.WISHNET.INSAR.INPUT_PX", str(SIZE // 16),
            "MODEL.DEVICE", "cpu", "OUTPUT_DIR", os.path.join(root, "out")]
    args = argparse.Namespace(config_file=os.path.join(os.path.dirname(HERE), "configs", "bdh", "wishnet.yaml"),
                              opts=opts, eval_only=True, resume=False, num_gpus=1, machine_rank=0,
                              num_machines=1, dist_url="auto")
    cfg = T.setup(args)
    mapper = WISHDatasetMapper(cfg, False)
    return mapper, mapper(DatasetCatalog.get("bdh_val")[0])


def same(a, b):
    """Equal bit for bit, no-data (NaN) at the same places."""
    if a.is_floating_point():
        return torch.equal(a.isnan(), b.isnan()) and torch.equal(a.nan_to_num(), b.nan_to_num())
    return torch.equal(a, b)


def check_flipped(mapper, d):
    d1 = flipped(mapper, d)
    d2 = flipped(mapper, d1)
    keys = [k for k in ("image", "sar_image", "insar_image") if k in d]
    assert keys == ["image", "sar_image", "insar_image"], keys
    for k in keys:
        assert same(d1[k], d[k].flip(-2)), k
        assert same(d2[k], d[k]), k
    m, m1, m2 = (x["instances"].gt_masks for x in (d, d1, d2))
    m = m.tensor if hasattr(m, "tensor") else m
    m1 = m1.tensor if hasattr(m1, "tensor") else m1
    m2 = m2.tensor if hasattr(m2, "tensor") else m2
    assert len(m) > 0 and torch.equal(m1, m.flip(-2)) and torch.equal(m2, m)
    assert torch.equal(d2["instances"].gt_boxes.tensor, d["instances"].gt_boxes.tensor)
    print("flipped: inputs %s and %d masks flipped up-down; flipped twice = the sample" % (keys, len(m)))


def detections(masks, scores, heights):
    h, w = masks.shape[-2:]
    inst = Instances((h, w))
    inst.pred_masks = masks
    ys, xs = [], []
    for m in masks:
        yy, xx = torch.nonzero(m, as_tuple=True)
        ys.append((int(yy.min()), int(yy.max()) + 1))
        xs.append((int(xx.min()), int(xx.max()) + 1))
    inst.pred_boxes = Boxes(torch.tensor([[x[0], y[0], x[1], y[1]] for x, y in zip(xs, ys)], dtype=torch.float32))
    inst.scores = torch.tensor(scores, dtype=torch.float32)
    inst.pred_classes = torch.zeros(len(masks), dtype=torch.int64)
    inst.pred_heights = torch.tensor(heights, dtype=torch.float32)
    inst.pred_offsets = torch.tensor([[1.0, 2.0 * (i + 1)] for i in range(len(masks))])
    return inst


def rect(y0, y1, x0, x1, size=SIZE):
    m = torch.zeros(size, size, dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def check_pair_average():
    # first pass: A, B, C (C overlaps B); the second pass in the flipped frame: B', A', D
    a, b, c = rect(10, 50, 10, 50), rect(100, 160, 100, 160), rect(110, 170, 104, 164)
    first = detections(torch.stack([a, b, c]), [0.9, 0.8, 0.7], [10.0, 20.0, 30.0])
    b2, a2, d2 = rect(100, 160, 102, 162), rect(10, 50, 10, 50), rect(200, 240, 200, 240)
    second_orig = detections(torch.stack([b2, a2, d2]), [0.85, 0.6, 0.5], [24.0, 12.0, 99.0])
    # the same detections as the model returns them on the flipped tile
    second = detections(second_orig.pred_masks.flip(-2), [0.85, 0.6, 0.5], [24.0, 12.0, 99.0])
    second.pred_offsets = second_orig.pred_offsets * torch.tensor([1.0, -1.0])
    back = write_back(second, float(SIZE))
    assert torch.equal(back.pred_masks, second_orig.pred_masks)
    assert torch.allclose(back.pred_boxes.tensor, second_orig.pred_boxes.tensor)
    assert torch.equal(back.pred_offsets, second_orig.pred_offsets)
    assert torch.equal(back.pred_heights, second_orig.pred_heights)
    match, iou = pair(first.pred_masks, back.pred_masks, first.scores, 0.5)
    # A pairs A' (IoU 1), B the free B' (IoU 0.875 on every 4th px), C finds B' taken, D apart
    assert match.tolist() == [1, 0, -1], match
    assert iou[0] == 1.0 and abs(iou[1] - 0.875) < 1e-6 and iou[2] == 0.0, iou
    avg = averaged(first, back, match)
    assert avg.pred_heights.tolist() == [11.0, 22.0, 30.0], avg.pred_heights
    for k in ("pred_masks", "scores", "pred_offsets"):
        assert torch.equal(avg.get(k), first.get(k)), k
    assert torch.equal(avg.pred_boxes.tensor, first.pred_boxes.tensor)
    # the evaluator removes the masks of what it is handed: shallow keeps the original's
    s = shallow(first)
    s.remove("pred_masks")
    assert first.has("pred_masks")
    print("write_back, pair, averaged: pairs %s, IoU %s, heights %s" % (
        match.tolist(), np.round(iou, 3).tolist(), avg.pred_heights.tolist()))


def close_logs():
    """Close the log files setup() opened in the temporary folder."""
    for lg in [logging.getLogger()] + [logging.getLogger(n) for n in list(logging.root.manager.loggerDict)]:
        for h in list(getattr(lg, "handlers", [])):
            if isinstance(h, logging.StreamHandler) and h.stream not in (sys.stdout, sys.stderr):
                st = h.stream
                lg.removeHandler(h)
                h.close()
                if st is not None:
                    st.close()


def main():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as root:
        try:
            mapper, d = sample(root)
            check_flipped(mapper, d)
        finally:
            close_logs()
    check_pair_average()
    print("PASS")


if __name__ == "__main__":
    main()
