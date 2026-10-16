# -*- coding: utf-8 -*-
"""The evaluation protocol of the paper (BDHEvaluator).

Segmentation, height and offset quality are all read off one COCO instance
matching at IoU 0.5, so the three are always measured over the same true
positives.  Three-dimensional IoU is reported twice, once extruding the roof
mask and once extruding the roof translated by the predicted offset, because a
model can get one right and the other wrong.

Two operating points are swept on validation and then frozen for test: the
threshold with the best F1, and the threshold with the lowest height MAE
subject to a recall floor.  The floor matters -- MAE is measured only over true
positives, so without it the sweep degenerates into keeping the few easiest
buildings.  The paper reports the best-F1 point (``f1op/*``).

Beside it, the literature's height MAE without a score threshold
(``bdhnet/*``, BDH-Net style: every image's MAX_DETS highest-scoring
detections, greedy COCO matching at IoU 0.5) and the Hungarian height MAE over
all predictions (``hungarian/*``).
"""
import contextlib
import io
import itertools
import json
import logging
import multiprocessing as mp
import os
from collections import OrderedDict

import numpy as np
import torch
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils.file_io import PathManager

logger = logging.getLogger(__name__)

POINT_NAMES = ("f1op", "maeop")
POINT_LABELS = {"f1op": "best F1", "maeop": "best MAE"}
BIN_METRICS = ("MAE", "RMSE", "mIoU3D_roof", "mIoU3D_base", "EPE_px")


def _bin_labels(edges):
    labels = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        labels.append("[%g,%g)" % (lo, hi) if hi is not None else "[%g,inf)" % lo)
    return labels


def _bin_index(h, edges):
    idx = 0
    for i, e in enumerate(edges):
        if h >= e:
            idx = i
    return idx


def _rle_counts(s):
    """The run lengths of a compressed COCO RLE string (maskApi.c rleFrString)."""
    if isinstance(s, str):
        s = s.encode("ascii")
    cnts, p, n = [], 0, len(s)
    while p < n:
        x = k = 0
        more = 1
        while more:
            c = s[p] - 48
            x |= (c & 0x1f) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def _rle_window(rle, x0, x1, y0, y1):
    """``mask_util.decode(rle)[y0:y1, x0:x1]``, evaluated at those pixels only:
    a pixel is set when more runs of ones start than end at or before its
    column-major index."""
    h = rle["size"][0]
    c = np.asarray(_rle_counts(rle["counts"]), dtype=np.int64)
    ends = np.cumsum(c)
    starts = ends - c
    f = np.arange(x0, x1)[None, :] * h + np.arange(y0, y1)[:, None]
    return (np.searchsorted(starts[1::2], f, side="right")
            - np.searchsorted(ends[1::2], f, side="right")) > 0


def _shift_window(rle, dx, dy, bbox):
    """Translate an RLE mask by a possibly fractional (dx, dy).

    The annotated offset is fractional (e.g. -1.535, +14.961), and the GT base
    is the rasterisation of the footprint polygon.  Rounding the shift to whole
    pixels therefore costs ~3.6 IoU even for a perfect prediction, which would
    cap ``mIoU3D_base`` below 100 for reasons that have nothing to do with the
    model.  So the sub-pixel part is applied as a bilinear resample thresholded
    at 0.5, which reproduces the rasterised translation closely.

    Work is confined to the mask's bounding box ``bbox`` (plus one pixel of
    margin), read straight from the RLE: a 2048 frame is never decoded.  The
    result is ``(sub, y, x)``: the translated mask is ``sub`` placed at (y, x),
    zero elsewhere; anything leaving the frame is dropped.
    """
    h, w = rle["size"]
    empty = (np.zeros((0, 0), dtype=bool), 0, 0)
    ix, iy = int(np.floor(dx)), int(np.floor(dy))
    fx, fy = float(dx - ix), float(dy - iy)

    bx, by, bw, bh = (int(v) for v in bbox)
    x0s, y0s = max(bx - 1, 0), max(by - 1, 0)
    x1s, y1s = min(bx + bw + 2, w), min(by + bh + 2, h)
    if x1s <= x0s or y1s <= y0s:
        return empty

    patch = _rle_window(rle, x0s, x1s, y0s, y1s)
    if fx < 1e-6 and fy < 1e-6:
        res = patch.astype(bool)
    else:
        patch = patch.astype(np.float32)
        ph, pw = patch.shape
        acc = np.zeros((ph + 1, pw + 1), dtype=np.float32)
        acc[0:ph, 0:pw] += (1.0 - fx) * (1.0 - fy) * patch
        acc[0:ph, 1:pw + 1] += fx * (1.0 - fy) * patch
        acc[1:ph + 1, 0:pw] += (1.0 - fx) * fy * patch
        acc[1:ph + 1, 1:pw + 1] += fx * fy * patch
        res = acc >= 0.5

    rh, rw = res.shape
    dy0, dx0 = y0s + iy, x0s + ix
    ty0, ty1 = max(dy0, 0), min(dy0 + rh, h)
    tx0, tx1 = max(dx0, 0), min(dx0 + rw, w)
    if ty1 <= ty0 or tx1 <= tx0:
        return empty
    return res[ty0 - dy0:ty1 - dy0, tx0 - dx0:tx1 - dx0], ty0, tx0


def _window_overlap(a, b):
    """Pixels set in both of two ``_shift_window`` results."""
    (ma, ya, xa), (mb, yb, xb) = a, b
    y0, x0 = max(ya, yb), max(xa, xb)
    y1 = min(ya + ma.shape[0], yb + mb.shape[0])
    x1 = min(xa + ma.shape[1], xb + mb.shape[1])
    if y1 <= y0 or x1 <= x0:
        return 0.0
    return float(np.logical_and(ma[y0 - ya:y1 - ya, x0 - xa:x1 - xa],
                                mb[y0 - yb:y1 - yb, x0 - xb:x1 - xb]).sum())


# ------------------------------------------------- worker processes for evaluate()
# The per-image COCO matching and the per-detection prisms are independent, so
# they run in forked worker processes; the results are put back in their
# original order, which leaves every number unchanged.  A forked worker reads
# the job from _POOL_JOB (set just before the fork) and never touches CUDA.
_POOL_JOB = None


def _eval_workers():
    """Worker processes for evaluate(): WISHNET_EVAL_WORKERS when set (1 = serial),
    else the CPUs this process may use (affinity and cgroup quota), at most 8."""
    env = os.environ.get("WISHNET_EVAL_WORKERS", "").strip()
    if env:
        return max(int(env), 1)
    if "fork" not in mp.get_all_start_methods():
        return 1
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    for quota, period in (("/sys/fs/cgroup/cpu.max", None),
                          ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
                           "/sys/fs/cgroup/cpu/cpu.cfs_period_us")):
        try:
            with open(quota) as f:
                q = f.read().split()
            if period is not None:
                with open(period) as f:
                    q.append(f.read().strip())
            if q[0] not in ("max", "-1"):
                n = min(n, max(int(int(q[0]) / int(q[1])), 1))
            break
        except (OSError, ValueError, IndexError):
            continue
    return max(min(n, 8), 1)


def _run_slice(bounds):
    return [_POOL_JOB(i) for i in range(*bounds)]


def _pool_map(fn, n, workers):
    """``[fn(i) for i in range(n)]``, spread over ``workers`` forked processes."""
    global _POOL_JOB
    if workers <= 1 or n < 4 * workers:
        return [fn(i) for i in range(n)]
    step = -(-n // (workers * 4))
    bounds = [(i, min(i + step, n)) for i in range(0, n, step)]
    _POOL_JOB = fn
    try:
        with mp.get_context("fork").Pool(workers) as pool:
            return [r for part in pool.map(_run_slice, bounds) for r in part]
    except Exception as exc:          # e.g. a daemonic process may not fork
        logger.warning("[BDHEvaluator] %d worker processes failed (%s); running serially",
                       workers, exc)
        return [fn(i) for i in range(n)]
    finally:
        _POOL_JOB = None


def _coco_evaluate(E, workers):
    """``E.evaluate()`` with its per-image matching spread over worker processes:
    evaluate() itself still lists the (image, category, area, maxDet) jobs in its
    own order, with evaluateImg stubbed out; the jobs then run in the pool."""
    if workers <= 1:
        E.evaluate()
        return
    E.evaluateImg = lambda *job: job
    try:
        E.evaluate()
    finally:
        del E.evaluateImg
    jobs = E.evalImgs
    E.evalImgs = _pool_map(lambda i: E.evaluateImg(*jobs[i]), len(jobs), workers)


def _hungarian_height(coco_gt, preds, min_height=0.0):
    """Height MAE over EVERY valid ground truth, via optimal assignment.

    No score threshold and no IoU gate: each ground truth is assigned the query
    that best overlaps it, and its height error counts whatever that query is.
    The denominator is therefore fixed at the number of valid ground truths, so
    the value is comparable across training iterations -- which the COCO-matched
    MAE is not.

    Returns (mae, rmse, coverage, n) where coverage is the share of ground
    truths whose assigned prediction actually overlaps them (IoU > 0.5).  Watch
    coverage alongside the MAE: a run can improve the MAE by predicting heights
    better OR by covering more buildings, and the two mean different things.
    """
    from scipy.optimize import linear_sum_assignment

    by_img = {}
    for p in preds:
        by_img.setdefault(p["image_id"], []).append(p)

    err, ok, n = [], 0, 0
    for img_id, anns in _gt_by_image(coco_gt, min_height).items():
        if not anns:
            continue
        # a building cut by the window is matched but never scored
        scored = [not a.get("truncated", False) for a in anns]
        n += sum(scored)
        dts = by_img.get(img_id, [])
        if not dts:
            # nothing predicted at all: every GT here is a miss.  Charging the
            # full height is the honest cost -- skipping them would let a model
            # that predicts nothing look perfect.
            err.extend(float(a["bd_height"]) for a, sc in zip(anns, scored) if sc)
            continue
        g_rle = [a["_rle"] for a in anns]
        d_rle = [d["segmentation"] for d in dts]
        iou = mask_util.iou(d_rle, g_rle, [0] * len(g_rle))     # (n_dt, n_gt)
        iou = np.asarray(iou, dtype=np.float64).reshape(len(d_rle), len(g_rle))
        di, gi = linear_sum_assignment(-iou)
        got = {}
        for a_, b_ in zip(di, gi):
            got[b_] = (float(dts[a_]["bd_height"]), float(iou[a_, b_]))
        for k, a in enumerate(anns):
            if not scored[k]:
                continue
            h_gt = float(a["bd_height"])
            if k in got:
                h_dt, v = got[k]
                err.append(h_dt - h_gt)
                ok += int(v > 0.5)
            else:
                err.append(-h_gt)          # more GT than queries: a miss
    if not err:
        return float("nan"), float("nan"), float("nan"), 0
    e = np.asarray(err, dtype=np.float64)
    return (float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean())),
            float(ok) / max(n, 1), n)


def _gt_by_image(coco_gt, min_height):
    """Valid ground truths per image, with their RLE cached on first use."""
    out = {}
    for img_id in coco_gt.getImgIds():
        keep = []
        for a in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id])):
            if float(a.get("bd_height", 0.0)) < min_height:
                continue
            if a.get("bd_ignore_region", False):
                continue          # not a height target (BDH_EVAL.IGNORE_REGIONS)
            if "_rle" not in a:
                a["_rle"] = coco_gt.annToRLE(a)
            keep.append(a)
        out[img_id] = keep
    return out


def _iou3d(inter, a_dt, a_gt, h_dt, h_gt, floor):
    hp, hg = max(h_dt, floor), max(h_gt, floor)
    vi = inter * min(hp, hg)
    vu = a_dt * hp + a_gt * hg - vi
    return float(vi / vu) if vu > 0 else 0.0


class BDHEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        output_dir=None,
        height_bins=(3.0, 10.0, 20.0, 30.0, 50.0, 100.0),
        iou_thresh=0.5,
        score_thresh_f1=-1.0,
        score_thresh_mae=-1.0,
        min_recall=0.30,
        mae_tol=0.05,
        max_dets=150,
        vol_min_height=1.5,
        tag="",
        ignore_truncated=True,
        ignore_regions=True,
        dump_predictions=True,
    ):
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self._edges = list(height_bins)
        self._labels = _bin_labels(self._edges)
        self._iou_thresh = float(iou_thresh)
        self._fixed = {"f1op": float(score_thresh_f1), "maeop": float(score_thresh_mae)}
        self._min_recall = float(min_recall)
        self._mae_tol = float(mae_tol)
        self._max_dets = int(max_dets)
        self._vol_min_h = float(vol_min_height)
        self._tag = tag
        self._dump_preds = bool(dump_predictions)
        self._cpu_device = torch.device("cpu")

        meta = MetadataCatalog.get(dataset_name)
        self._json_file = PathManager.get_local_path(meta.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(self._json_file)
        self._n_ignore = self._apply_ignore(bool(ignore_truncated), bool(ignore_regions))
        self._predictions = []

    def _apply_ignore(self, ignore_truncated, ignore_regions):
        """Ignore regions (COCO iscrowd): a detection on one is neither a true
        nor a false positive and missing one is not a false negative; COCOeval
        does both by itself, and the operating-point metrics below drop the
        detections it marks dtIgnore.  With ``ignore_truncated`` the buildings
        cut by the window or tile edge; with ``ignore_regions`` also the ones
        the eval json leaves out (no recorded floor count, or a visible part
        below the size / fraction gates), from ``<json>_ignore.json``.  The
        Hungarian schedule metric keeps assigning the truncated ones (matched,
        not scored) as before and never sees the added regions."""
        ds = self._coco_api.dataset
        n_tr = n_reg = 0
        if ignore_truncated:
            for a in ds["annotations"]:
                if a.get("truncated", False):
                    a["iscrowd"] = 1
                    n_tr += 1
        if ignore_regions:
            path = os.path.splitext(self._json_file)[0] + "_ignore.json"
            if not os.path.exists(path):
                raise FileNotFoundError(
                    "BDH_EVAL.IGNORE_REGIONS: %s is missing -- run tools/labels/make_ignore_regions.py "
                    "or set BDH_EVAL.IGNORE_REGIONS False" % path)
            with open(path, "r", encoding="utf-8") as f:
                extra = json.load(f)["annotations"]
            nid = max([a["id"] for a in ds["annotations"]] + [0]) + 1
            for a in extra:
                b = dict(a)
                b["id"] = nid
                b["iscrowd"] = 1
                b["bd_ignore_region"] = True
                ds["annotations"].append(b)
                nid += 1
            n_reg = len(extra)
        if n_tr or n_reg:
            with contextlib.redirect_stdout(io.StringIO()):
                self._coco_api.createIndex()
        logger.info("[BDHEvaluator] %s: ignore regions -- %d truncated buildings, %d added "
                    "(no recorded height / below the crop gates)", self._dataset_name, n_tr, n_reg)
        return n_tr, n_reg

    def reset(self):
        self._predictions = []

    # --------------------------------------------------------------- process
    @staticmethod
    def _sparse_rle(m, col_any=None):
        """COCO RLE of one (H, W) bool mask, the same bytes as
        mask_util.encode(np.asfortranarray(m)), from its occupied columns only:
        column-major order puts columns x0..x1 in one run of the flattened mask,
        so the zeros before and after them become two runs.  A building fills a
        few hundred of a 2048 tile's 4.2 M pixels."""
        H, W = m.shape
        cols = np.flatnonzero(m.any(0) if col_any is None else col_any)
        if len(cols) == 0:
            return mask_util.frPyObjects([{"size": [H, W], "counts": [H * W]}], H, W)[0]
        x0, x1 = int(cols[0]), int(cols[-1])
        flat = m[:, x0:x1 + 1].ravel(order="F")
        change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
        runs = np.diff(np.concatenate([[0], change, [flat.size]])).tolist()
        if flat[0]:                       # an RLE starts with a run of zeros
            runs = [0] + runs
        runs[0] += x0 * H
        tail = (W - 1 - x1) * H
        if tail:
            if len(runs) % 2 == 1:        # the last run is zeros
                runs[-1] += tail
            else:
                runs.append(tail)
        return mask_util.frPyObjects([{"size": [H, W], "counts": runs}], H, W)[0]

    @staticmethod
    def _rles_on_device(masks, chunk=32):
        """COCO RLE of each (H, W) mask of an (n, H, W) GPU tensor, the same bytes
        as _sparse_rle: the masks are flattened column-major on the GPU, and only
        the positions where a pixel differs from the one before come back."""
        n, H, W = masks.shape
        N = H * W
        rles = []
        for i in range(0, n, chunk):
            t = masks[i:i + chunk].bool()
            f = t.transpose(1, 2).reshape(t.shape[0], N)
            d = (f[:, 1:] != f[:, :-1]).nonzero().cpu().numpy()     # (mask, position), sorted
            first = f[:, 0].cpu().numpy()
            cut = np.searchsorted(d[:, 0], np.arange(t.shape[0] + 1))
            for j in range(t.shape[0]):
                change = d[cut[j]:cut[j + 1], 1] + 1
                runs = np.diff(np.concatenate([[0], change, [N]])).tolist()
                if first[j]:                  # an RLE starts with a run of zeros
                    runs = [0] + runs
                rles.append(mask_util.frPyObjects([{"size": [H, W], "counts": runs}], H, W)[0])
        return rles

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            inst = out["instances"]
            n = len(inst)
            if n == 0:
                continue
            if inst.pred_masks.is_cuda:
                rles = self._rles_on_device(inst.pred_masks)
                # encoded: the next image's forward need not hold these as well
                inst.remove("pred_masks")
            else:
                masks = inst.pred_masks.numpy().astype(bool, copy=False)
                col_any = masks.any(axis=1)                      # (n, W) occupied columns
                rles = [self._sparse_rle(masks[i], col_any[i]) for i in range(n)]
            scores = inst.scores.cpu().numpy()
            heights = inst.pred_heights.cpu().numpy() if inst.has("pred_heights") else np.zeros(n)
            offsets = (inst.pred_offsets.cpu().numpy() if inst.has("pred_offsets")
                       else np.zeros((n, 2), dtype=np.float32))
            for i in range(n):
                self._predictions.append(
                    {
                        "image_id": inp["image_id"],
                        "category_id": 1,
                        "segmentation": rles[i],   # roof; counts stays bytes on purpose
                        "score": float(scores[i]),
                        "bd_height": float(heights[i]),
                        "bd_offset": [float(offsets[i][0]), float(offsets[i][1])],
                    }
                )

    # -------------------------------------------------------------- evaluate
    def _prisms(self, dt, gt):
        """Areas and overlap of one true positive's two prisms:
        (a_dt, a_gt, inter) of the roofs, by RLE set operations, and
        (ab_dt, ab_gt, inter_b) of the bases, each roof mask translated by its
        own offset so the two sides are constructed identically (ceiling is
        100); inter_b is 0 when either base is empty."""
        rle_dt = dt["segmentation"]
        rle_gt = self._coco_api.annToRLE(gt)
        a_dt = float(mask_util.area(rle_dt))
        a_gt = float(mask_util.area(rle_gt))
        inter = float(mask_util.area(mask_util.merge([rle_dt, rle_gt], intersect=True)))

        op = np.asarray(dt.get("bd_offset", [0.0, 0.0]), dtype=np.float64)
        og = np.asarray(gt.get("bd_offset", [0.0, 0.0]), dtype=np.float64)
        b_dt = _shift_window(rle_dt, op[0], op[1], mask_util.toBbox(rle_dt))
        b_gt = _shift_window(rle_gt, og[0], og[1], mask_util.toBbox(rle_gt))
        ab_dt = float(b_dt[0].sum())
        ab_gt = float(b_gt[0].sum())
        inter_b = _window_overlap(b_dt, b_gt) if ab_dt > 0 and ab_gt > 0 else 0.0
        return a_dt, a_gt, inter, ab_dt, ab_gt, inter_b

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            preds = comm.gather(self._predictions, dst=0)
            preds = list(itertools.chain(*preds))
            if not comm.is_main_process():
                return {}
        else:
            preds = self._predictions

        if len(preds) == 0:
            logger.warning("[BDHEvaluator] no predictions at all")
            return OrderedDict({"bdh": {}})

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
        if self._output_dir and self._dump_preds:
            dump = []
            for p in preds:
                q = dict(p)
                q["segmentation"] = {
                    "size": p["segmentation"]["size"],
                    "counts": p["segmentation"]["counts"].decode("utf-8"),
                }
                dump.append(q)
            with PathManager.open(
                os.path.join(self._output_dir, "bdh_predictions%s.json" % self._tag), "w"
            ) as f:
                json.dump(dump, f)

        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = self._coco_api.loadRes([dict(p) for p in preds])
        workers = _eval_workers()
        logger.info("[BDHEvaluator] %d predictions, %d worker processes", len(preds), workers)

        # ---------------------------------------------------------- COCO AP
        E = COCOeval(self._coco_api, coco_dt, iouType="segm")
        E.params.maxDets = [1, 10, self._max_dets]
        with contextlib.redirect_stdout(io.StringIO()):
            _coco_evaluate(E, workers)
            E.accumulate()
        E.summarize()
        ap_names = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
        res = OrderedDict((("seg" + nm), float(E.stats[i] * 100)) for i, nm in enumerate(ap_names))

        # ------------------------- Hungarian matching, for the SCHEDULE only
        # Threshold-free and with a fixed denominator, so it is comparable
        # across training iterations.  Never reported as a protocol number.
        try:
            # the prepared GT is already filtered to bd_height >= INPUT.MIN_HEIGHT
            # when the crops are generated (verified: its minimum is exactly
            # 3.00 m), so there is no second filter here
            hm, hr, hc, hn = _hungarian_height(self._coco_api, preds)
            res["hungarian/MAE"] = hm
            res["hungarian/RMSE"] = hr
            res["hungarian/coverage"] = hc * 100.0
            res["hungarian/n_gt"] = float(hn)
            logger.info("[BDHEvaluator] Hungarian (schedule metric, all %d GT, "
                        "no threshold): MAE %.4f m  RMSE %.4f m  coverage %.1f%%",
                        hn, hm, hr, hc * 100.0)
        except Exception as exc:
            logger.warning("[BDHEvaluator] Hungarian metric failed: %s", exc)

        # --------------------------------------- COCO matching at one IoU
        M = COCOeval(self._coco_api, coco_dt, iouType="segm")
        M.params.iouThrs = np.array([self._iou_thresh])
        M.params.areaRng = [[0, 1e10]]
        M.params.areaRngLbl = ["all"]
        M.params.maxDets = [self._max_dets]
        with contextlib.redirect_stdout(io.StringIO()):
            _coco_evaluate(M, workers)

        pairs = []          # (score, dt_id, gt_id)  gt_id == 0 -> false positive
        n_gt = 0
        for e in M.evalImgs:
            if e is None:
                continue
            gt_ig = np.asarray(e["gtIgnore"], dtype=bool)
            n_gt += int((~gt_ig).sum())
            dt_ids = e["dtIds"]
            dt_scores = e["dtScores"]
            dt_m = np.asarray(e["dtMatches"])[0]
            dt_ig = np.asarray(e["dtIgnore"])[0].astype(bool)
            for i in range(len(dt_ids)):
                if dt_ig[i]:
                    continue
                pairs.append((float(dt_scores[i]), int(dt_ids[i]), int(dt_m[i])))

        pairs.sort(key=lambda x: -x[0])
        n = len(pairs)
        scores = np.array([p[0] for p in pairs])

        # ---- per-detection quantities, computed ONCE over the whole ranking
        is_tp = np.zeros(n, dtype=bool)
        trunc = np.zeros(n, dtype=bool)   # the matched GT is cut by the window
        serr = np.zeros(n)            # signed height error, metres
        oerr = np.zeros(n)            # offset end-point error, pixels
        gsd = np.zeros(n)             # metres per pixel of the parent tile
        i3_roof = np.zeros(n)
        i3_base = np.zeros(n)
        i2_base = np.zeros(n)         # 2-D IoU of the bases -> footprint positive
        has_base = np.zeros(n, dtype=bool)
        binx = np.full(n, -1, dtype=int)
        tp = [i for i, (_, _, gt_id) in enumerate(pairs) if gt_id > 0]
        prisms = dict(zip(tp, _pool_map(
            lambda j: self._prisms(coco_dt.anns[pairs[tp[j]][1]], self._coco_api.anns[pairs[tp[j]][2]]),
            len(tp), workers)))
        for i, (_, dt_id, gt_id) in enumerate(pairs):
            if gt_id <= 0:
                continue
            dt = coco_dt.anns[dt_id]
            gt = self._coco_api.anns[gt_id]
            img = self._coco_api.imgs[gt["image_id"]]
            hp = float(dt["bd_height"])
            hg = float(gt["bd_height"])
            is_tp[i] = True
            trunc[i] = bool(gt.get("truncated", False))
            serr[i] = hp - hg
            binx[i] = _bin_index(hg, self._edges)
            gsd[i] = float(img.get("gsd_m", 0.0))

            op = np.asarray(dt.get("bd_offset", [0.0, 0.0]), dtype=np.float64)
            og = np.asarray(gt.get("bd_offset", [0.0, 0.0]), dtype=np.float64)
            oerr[i] = float(np.linalg.norm(op - og))

            a_dt, a_gt, inter, ab_dt, ab_gt, inter_b = prisms[i]
            i3_roof[i] = _iou3d(inter, a_dt, a_gt, hp, hg, self._vol_min_h)
            if ab_gt <= 0:
                # the GT base falls outside the crop entirely -- nothing to score
                continue
            has_base[i] = True
            if ab_dt <= 0:
                i3_base[i] = 0.0
                i2_base[i] = 0.0
                continue
            i3_base[i] = _iou3d(inter_b, ab_dt, ab_gt, hp, hg, self._vol_min_h)
            union_b = ab_dt + ab_gt - inter_b
            i2_base[i] = inter_b / union_b if union_b > 0 else 0.0

        tp_cum = np.cumsum(is_tp.astype(np.float64))
        fp_cum = np.cumsum(1.0 - is_tp.astype(np.float64))
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        rec = tp_cum / max(n_gt, 1)
        f1_curve = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        # the MAE sweep, like the metrics below, scores the non-truncated
        # true positives only (a cut building is matched, not measured)
        sc_cum = np.cumsum((is_tp & ~trunc).astype(np.float64))
        mae_curve = np.cumsum(np.abs(serr) * (~trunc)) / np.maximum(sc_cum, 1.0)
        mae_curve[sc_cum == 0] = np.inf

        # ------------------------------------------- pick the two thresholds
        guard_hit = False
        sweep_f1_i = int(np.argmax(f1_curve)) if n else 0
        sweep_f1_tau = float(scores[sweep_f1_i]) if n else 1.0

        if n:
            eligible = rec >= self._min_recall
            if not eligible.any():
                eligible = np.ones(n, dtype=bool)
                guard_hit = True
            cand = np.where(eligible)[0]
            mc = mae_curve[cand]
            # `cand` is ascending in k and recall is non-decreasing in k, so the
            # LAST near-optimal candidate keeps the most detections.  A plain
            # argmin would do the opposite on a flat curve.
            near = cand[mc <= float(np.min(mc)) + self._mae_tol]
            sweep_mae_i = int(near[-1])
            sweep_mae_tau = float(scores[sweep_mae_i])
        else:
            sweep_mae_i, sweep_mae_tau = 0, 1.0

        chosen = OrderedDict()
        for name, sweep_tau in (("f1op", sweep_f1_tau), ("maeop", sweep_mae_tau)):
            fixed = self._fixed[name]
            if fixed >= 0:
                tau = fixed
                k = int(np.searchsorted(-scores, -tau, side="right")) if n else 0
                src = "frozen"
            else:
                tau = sweep_tau
                k = ((sweep_f1_i if name == "f1op" else sweep_mae_i) + 1) if n else 0
                src = "swept"
            chosen[name] = (k, tau, src)

        # -------------------------------------- aggregate at each op point
        res["n_gt"] = float(n_gt)
        res["n_ignore_truncated"] = float(self._n_ignore[0])
        res["n_ignore_regions"] = float(self._n_ignore[1])
        res["n_det"] = float(n)
        per_bin = OrderedDict()

        def _mean(a):
            return float(a.mean()) if a.size else float("nan")

        for name, (k, tau, src) in chosen.items():
            p = name + "/"
            res[p + "score_thresh"] = float(tau)
            res[p + "F1@0.5"] = float(f1_curve[k - 1] * 100) if k > 0 else 0.0
            res[p + "precision"] = float(prec[k - 1] * 100) if k > 0 else 0.0
            res[p + "recall"] = float(rec[k - 1] * 100) if k > 0 else 0.0

            m = is_tp[:k] & ~trunc[:k]          # scored: matched and not cut
            res[p + "n_trunc_excluded"] = float((is_tp[:k] & trunc[:k]).sum())
            e = serr[:k][m]
            oe = oerr[:k][m]
            gs = gsd[:k][m]
            gr = i3_roof[:k][m]
            hb = has_base[:k][m]
            gb = i3_base[:k][m][hb]
            b = binx[:k][m]
            # BONAI's EPE counts an offset only where the footprint prediction
            # is itself a positive
            fpos = hb & (i2_base[:k][m] >= self._iou_thresh)
            oe_p = oe[fpos]
            gs_p = gs[fpos]

            res[p + "MAE"] = _mean(np.abs(e))
            res[p + "RMSE"] = float(np.sqrt((e ** 2).mean())) if e.size else float("nan")
            res[p + "mIoU3D_roof"] = _mean(gr) * 100 if gr.size else float("nan")
            res[p + "mIoU3D_base"] = _mean(gb) * 100 if gb.size else float("nan")
            res[p + "EPE_px"] = _mean(oe_p)
            res[p + "EPE_median_px"] = float(np.median(oe_p)) if oe_p.size else float("nan")
            res[p + "EPE_m"] = _mean(oe_p * gs_p)
            res[p + "EPE_all_px"] = _mean(oe)
            res[p + "n_epe"] = float(oe_p.size)
            res[p + "fp_positive_rate"] = (
                float(fpos.sum()) / e.size * 100 if e.size else float("nan"))
            res[p + "n_matched"] = float(e.size)
            res[p + "n_base"] = float(gb.size)

            pb = OrderedDict()
            for bi, lab in enumerate(self._labels):
                sel = b == bi
                ee, rr = e[sel], gr[sel]
                bb = i3_base[:k][m][sel & hb]
                oo = oe[sel & fpos]
                pb[lab] = {
                    "n": int(ee.size),
                    "n_base": int(bb.size),
                    "n_epe": int(oo.size),
                    "MAE": _mean(np.abs(ee)),
                    "RMSE": float(np.sqrt((ee ** 2).mean())) if ee.size else float("nan"),
                    "mIoU3D_roof": _mean(rr) * 100 if rr.size else float("nan"),
                    "mIoU3D_base": _mean(bb) * 100 if bb.size else float("nan"),
                    "EPE_px": _mean(oo),
                }
                for mk in BIN_METRICS:
                    res[p + mk + "/" + lab] = pb[lab][mk]
                res[p + "n/" + lab] = float(ee.size)
            per_bin[name] = pb

        # -------------- supplementary: the literature's MAE (BDH-Net style)
        # No score threshold: the top MAX_DETS detections of each image, COCO's
        # greedy matching at IoU 0.5 (the matching above), and the MAE over every
        # matched ground truth with a height -- cut buildings and the ignore
        # regions left out as in the main protocol.  Added after every number
        # above, which it leaves as it was.
        m = is_tp & ~trunc
        e, b = serr[m], binx[m]
        res["bdhnet/MAE"] = _mean(np.abs(e))
        res["bdhnet/n_matched"] = float(e.size)
        pb = OrderedDict()
        for bi, lab in enumerate(self._labels):
            ee = e[b == bi]
            pb[lab] = {"n": int(ee.size), "MAE": _mean(np.abs(ee))}
            res["bdhnet/MAE/" + lab] = pb[lab]["MAE"]
            res["bdhnet/n/" + lab] = float(ee.size)
        per_bin["bdhnet"] = pb

        self._log_table(res, per_bin, chosen, guard_hit)

        if self._output_dir:
            payload = {
                "dataset": self._dataset_name,
                "min_recall": self._min_recall,
                "mae_tol": self._mae_tol,
                "min_recall_guard_unreachable": bool(guard_hit),
                "height_bins": self._edges,
                "bin_labels": self._labels,
                "points": {kk: {"k": int(v[0]), "score_thresh": float(v[1]), "source": v[2]}
                           for kk, v in chosen.items()},
                "overall": dict(res),
                "per_bin": per_bin,
                "sweep": {
                    "score": scores.tolist(),
                    "precision": prec.tolist(),
                    "recall": rec.tolist(),
                    "f1": f1_curve.tolist(),
                    "mae": [None if not np.isfinite(v) else float(v) for v in mae_curve],
                },
            }
            with PathManager.open(
                os.path.join(self._output_dir, "bdh_metrics%s.json" % self._tag), "w"
            ) as f:
                json.dump(payload, f, indent=2)

        return OrderedDict({"bdh": res})

    def _log_table(self, res, per_bin, chosen, guard_hit):
        lines = [
            "",
            "=" * 104,
            "BDH evaluation on %s" % self._dataset_name,
            "segAP %.2f   segAP50 %.2f   segAP75 %.2f   (roof masks, threshold-free)"
            "   GT %d, detections %d"
            % (res["segAP"], res["segAP50"], res["segAP75"], int(res["n_gt"]), int(res["n_det"])),
        ]
        if guard_hit:
            lines.append("!! recall floor %.2f was UNREACHABLE -> the MAE sweep ran "
                         "unconstrained; check its recall before quoting its MAE."
                         % self._min_recall)
        for name in POINT_NAMES:
            k, tau, src = chosen[name]
            p = name + "/"
            extra = ""
            if name == "maeop" and src == "swept":
                extra = "  [recall floor %.2f, tie window %.2f m]" % (
                    self._min_recall, self._mae_tol)
            lines += [
                "-" * 104,
                "operating point '%s' = %s  (%s, score >= %.4f)%s"
                % (name, POINT_LABELS[name], src, tau, extra),
                "  F1@IoU0.5 %.2f   P %.2f   R %.2f   matched %d / %d GT   (base scored on %d)"
                % (res[p + "F1@0.5"], res[p + "precision"], res[p + "recall"],
                   int(res[p + "n_matched"]), int(res["n_gt"]), int(res[p + "n_base"])),
                "  offset EPE %.2f px (%.2f m)   median %.2f px   on %d footprint-"
                "positive of %d matched (%.1f%%)   [unconditioned %.2f px]"
                % (res[p + "EPE_px"], res[p + "EPE_m"], res[p + "EPE_median_px"],
                   int(res[p + "n_epe"]), int(res[p + "n_matched"]),
                   res[p + "fp_positive_rate"], res[p + "EPE_all_px"]),
                "  %-12s %7s %10s %10s %11s %11s %8s %9s"
                % ("height bin", "n", "MAE(m)", "RMSE(m)",
                   "IoU3D_roof", "IoU3D_base", "n_epe", "EPE(px)"),
            ]
            for lab, d in per_bin[name].items():
                lines.append("  %-12s %7d %10.3f %10.3f %11.2f %11.2f %8d %9.2f"
                             % (lab, d["n"], d["MAE"], d["RMSE"],
                                d["mIoU3D_roof"], d["mIoU3D_base"],
                                d["n_epe"], d["EPE_px"]))
            lines.append("  %-12s %7d %10.3f %10.3f %11.2f %11.2f %8d %9.2f"
                         % ("Overall", int(res[p + "n_matched"]), res[p + "MAE"],
                            res[p + "RMSE"], res[p + "mIoU3D_roof"],
                            res[p + "mIoU3D_base"], int(res[p + "n_epe"]),
                            res[p + "EPE_px"]))
        if "bdhnet" in per_bin:
            lines += ["-" * 104,
                      "supplementary, BDH-Net style: no score threshold, top %d per image, "
                      "COCO matching at IoU %.1f -- MAE %.3f m over %d matched"
                      % (self._max_dets, self._iou_thresh, res["bdhnet/MAE"],
                         int(res["bdhnet/n_matched"])),
                      "  %-12s %7s %10s" % ("height bin", "n", "MAE(m)")]
            lines += ["  %-12s %7d %10.3f" % (lab, d["n"], d["MAE"])
                      for lab, d in per_bin["bdhnet"].items()]
        lines.append("=" * 104)
        logger.info("\n".join(lines))
