# -*- coding: utf-8 -*-
"""Dataset mapper: the optical window, the GF-3 window of the same ground, the
InSAR tile, the ground-truth instances.

* training: a random CROP_SIZE window of the 2048 tile (with CROP_SIZE 2048 the
  whole tile; retried until it holds a building), +-10 % brightness and
  contrast on the optical image (INPUT.RGB_*); evaluation: the prepared crop
  (crop_box of the evaluation json);
* annotations clipped to the window: roof mask, box = roof union footprint
  (clipped), height in metres, roof -> footprint offset in px; a building is
  ``truncated`` when the window cuts its roof or footprint;
* SAR: the GF-3 tile of the same ground (float32 dB on the tile's grid, NaN ->
  the tile's mean), cut to the optical window.  No radiometric change --
  standardisation happens in the model.  The InSAR tile (MODEL.WISHNET.INSAR)
  over the same window, physical values, NaN = no data.
* INPUT.TILE_SCALE (0 = off): the optical window and the SAR window are
  area-resampled by this factor (the SAR in dB, as the model resamples its
  window), the roof polygons are rasterised at the new size and the boxes and
  offsets scaled; heights stay in metres.  ``height`` / ``width`` keep the
  window's own size, so the model maps its outputs back to it.  With 2048
  windows and 0.5 the whole tile goes in at 1024 (0.83 m/px).
* INPUT.GEO_AUG "all8" (training only): the sample is flipped or rotated as a
  whole -- optical, GF-3, InSAR, masks, boxes and roof -> footprint offsets by
  the same transform, heights unchanged -- drawn uniformly from the eight
  transforms of the square.  ``geo_M`` is its 2 x 2 linear part (image px, x
  right, y down), which the range reads apply to the direction towards the
  sensor and to their learned offsets.
* MODEL.WISHNET.SAR.SPECKLE (training only): with probability PROB the GF-3
  window, before it is resampled, gets new speckle (speckle_resample) -- the
  fine texture a model can memorise a tile by is drawn anew; the structure
  above MEAN_SIGMA_M and the strong scatterers stay.
"""
import copy
import logging
import os

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import box as shp_box

from detectron2.config import configurable
from detectron2.structures import Boxes, Instances

from .register import offset_vector

logger = logging.getLogger(__name__)

# truncated: less than this share of the roof inside the window, or the roof
# within this many px of its edge (INPUT.TRUNC_PX; this is its default), or
# the footprint beyond it
TRUNC_FRAC = 0.995
TRUNC_EDGE_PX = 1.0

# INPUT.GEO_AUG "all8": the transforms a training sample is drawn from, each one
# (transpose, flip x, flip y) applied in that order to a square window
GEO_TRANSFORMS = {
    "identity": (False, False, False),
    "flip_ud": (False, False, True),
    "flip_lr": (False, True, False),
    "rot180": (False, True, True),
    "transpose": (True, False, False),
    "rot90": (True, False, True),           # x -> y, y -> -x: counter-clockwise on screen
    "rot270": (True, True, False),
    "antitranspose": (True, True, True),
}
GEO_AUG_SETS = {"all8": tuple(GEO_TRANSFORMS)}


def _gauss(t, sigma_px):
    r = max(1, int(np.ceil(3.0 * sigma_px)))
    x = torch.arange(-r, r + 1, dtype=t.dtype)
    k = torch.exp(-0.5 * (x / float(sigma_px)) ** 2)
    k = k / k.sum()
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(t, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))


def gauss_blur(a, sigma_px):
    """2-D float array -> its Gaussian blur (separable, reflected edges).  A wide
    blur (sigma >= 6 px) runs on the array area-reduced by sigma // 3 and is
    enlarged back bilinearly: the blur is smooth at that scale, and a 2048
    window costs a few tens of milliseconds instead of seconds."""
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64))[None, None]
    q = int(sigma_px // 3) if sigma_px >= 6 else 1
    if q <= 1:
        return _gauss(t, sigma_px)[0, 0].numpy()
    H, W = t.shape[-2:]
    small = F.interpolate(t, size=(max(1, H // q), max(1, W // q)), mode="area")
    small = _gauss(small, sigma_px / q)
    return F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)[0, 0].numpy()


def speckle_parts(lin, gsd_m, sigma_m=5.0, keep_sigma_m=1.0, bright_db=7.6):
    """The parts speckle_resample keeps: ``keep``, the strong scatterers (the
    intensity blurred with keep_sigma_m above bright_db), and ``mean``, the
    Gaussian local mean with sigma_m over the other pixels (normalised
    convolution, so a kept scatterer leaves no halo around it)."""
    keep = gauss_blur(lin, keep_sigma_m / gsd_m) >= 10.0 ** (float(bright_db) / 10.0)
    free = (~keep).astype(np.float64)
    num = gauss_blur(lin * free, sigma_m / gsd_m)
    den = gauss_blur(free, sigma_m / gsd_m)
    mean = np.where(den > 1e-3, num / np.maximum(den, 1e-3), lin)
    return keep, mean


def speckle_noise(shape, gsd_m, enl, native_m_x, native_m_y, upsample="bilinear", rng=np.random):
    """Mean-1 Gamma(enl, 1/enl) samples every native_m_x along x (range) and
    native_m_y along y (azimuth), at a random phase, enlarged to ``shape``."""
    H, W = shape
    fy, fx = float(native_m_y) / float(gsd_m), float(native_m_x) / float(gsd_m)
    oy, ox = rng.uniform(0.0, fy), rng.uniform(0.0, fx)
    nh, nw = int(np.ceil((H + oy) / fy)) + 2, int(np.ceil((W + ox) / fx)) + 2
    noise = rng.gamma(float(enl), 1.0 / float(enl), size=(nh, nw))
    if upsample == "nearest":
        r = np.floor((np.arange(H) + oy) / fy).astype(int)
        c = np.floor((np.arange(W) + ox) / fx).astype(int)
        return noise[np.ix_(r, c)]
    r = (np.arange(H) + oy) / fy + 0.5                       # bilinear between the native samples
    c = (np.arange(W) + ox) / fx + 0.5
    r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
    wr, wc = (r - r0)[:, None], (c - c0)[None, :]
    return ((1 - wr) * (1 - wc) * noise[np.ix_(r0, c0)] + (1 - wr) * wc * noise[np.ix_(r0, c0 + 1)]
            + wr * (1 - wc) * noise[np.ix_(r0 + 1, c0)] + wr * wc * noise[np.ix_(r0 + 1, c0 + 1)])


def speckle_resample(sw, gsd_m, enl=0.38, native_m_x=2.69, native_m_y=3.10, sigma_m=5.0, keep_sigma_m=1.0,
                     bright_db=7.6, upsample="bilinear", floor_db=-35.0, rng=np.random):
    """MODEL.WISHNET.SAR.SPECKLE: a GF-3 window (1, H, W) in dB on the tile's grid
    (``gsd_m`` m/px) with new speckle: in the linear domain the Gaussian local
    mean with ``sigma_m`` times new speckle (speckle_noise) divided by its own
    Gaussian local mean with ``sigma_m``, so that the new speckle varies only
    below that scale and the structure above it stays the original's; the
    strong scatterers keep their own values, bit for bit (speckle_parts).  Back
    in dB, floored at ``floor_db`` as the product is."""
    db = sw[0].astype(np.float64)
    lin = np.power(10.0, db / 10.0)
    keep, mean = speckle_parts(lin, gsd_m, sigma_m, keep_sigma_m, bright_db)
    up = speckle_noise(lin.shape, gsd_m, enl, native_m_x, native_m_y, upsample, rng)
    up = up / np.maximum(gauss_blur(up, sigma_m / gsd_m), 1e-6)
    out = 10.0 * np.log10(np.maximum(mean * up, 10.0 ** (float(floor_db) / 10.0)))
    out = np.where(keep, db, out)                               # kept pixels bit for bit
    return out.astype(sw.dtype)[None]


def _speckle_args(cfg):
    sp = cfg.MODEL.WISHNET.SAR.SPECKLE
    if not sp.ENABLED or cfg.MODEL.WISHNET.MODE != "fusion":
        return None
    return {"prob": float(sp.PROB), "gsd_m": float(cfg.MODEL.WISHNET.RANGE.GSD_M), "enl": float(sp.ENL),
            "native_m_x": float(sp.NATIVE_M_X), "native_m_y": float(sp.NATIVE_M_Y),
            "sigma_m": float(sp.MEAN_SIGMA_M), "keep_sigma_m": float(sp.KEEP_SIGMA_M),
            "bright_db": float(sp.BRIGHT_DB), "upsample": str(sp.UPSAMPLE)}


def geo_matrix(tr):
    """The 2 x 2 linear part of a transform (transpose, flip x, flip y) in image
    px (x right, y down): a vector v of the original goes to M v."""
    m = np.eye(2)
    if tr[0]:
        m = np.array([[0.0, 1.0], [1.0, 0.0]]) @ m
    if tr[1]:
        m = np.diag([-1.0, 1.0]) @ m
    if tr[2]:
        m = np.diag([1.0, -1.0]) @ m
    return m


def geo_tensor(t, tr):
    """(..., H, W) tensor -> the transformed one (H == W when it transposes)."""
    if tr[0]:
        if t.shape[-1] != t.shape[-2]:
            raise ValueError("INPUT.GEO_AUG transposes square windows only, not %s" % (tuple(t.shape[-2:]),))
        t = t.transpose(-1, -2)
    if tr[1]:
        t = t.flip(-1)
    if tr[2]:
        t = t.flip(-2)
    return t.contiguous()


def geo_points(xy, tr, size):
    """(N, 2) points (px, continuous, in a window of side ``size``) -> transformed."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2).copy()
    if tr[0]:
        xy = xy[:, ::-1].copy()
    if tr[1]:
        xy[:, 0] = size - xy[:, 0]
    if tr[2]:
        xy[:, 1] = size - xy[:, 1]
    return xy


def _to_shapely(poly_flat):
    pts = np.asarray(poly_flat, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return None
    g = Polygon(pts)
    if not g.is_valid:
        g = g.buffer(0)
    return None if g.is_empty else g


def _geom_to_coco_polys(geom):
    if isinstance(geom, Polygon):
        geoms = [geom]
    elif isinstance(geom, MultiPolygon):
        geoms = list(geom.geoms)
    else:
        geoms = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    polys, area = [], 0.0
    for g in geoms:
        if g.is_empty or g.area <= 0:
            continue
        coords = np.asarray(g.exterior.coords, dtype=np.float64)[:-1]
        if len(coords) < 3:
            continue
        polys.append(coords.reshape(-1).tolist())
        area += g.area
    return polys, area


def _clip_poly_list(polys, rect, x0, y0, size):
    """Clip flat polygons to ``rect`` and shift them into window pixels."""
    geoms = [g for g in (_to_shapely(q) for q in polys) if g is not None]
    if not geoms:
        return None, 0.0, 0.0
    geom = geoms[0] if len(geoms) == 1 else MultiPolygon(geoms).buffer(0)
    orig_area = geom.area
    if orig_area <= 0:
        return None, 0.0, 0.0
    try:
        inter = geom.intersection(rect)
    except Exception:
        return None, 0.0, orig_area
    if inter.is_empty:
        return None, 0.0, orig_area
    out, area = _geom_to_coco_polys(inter)
    if not out:
        return None, 0.0, orig_area
    shifted = []
    for q in out:
        arr = np.asarray(q, dtype=np.float64).reshape(-1, 2)
        arr[:, 0] = np.clip(arr[:, 0] - x0, 0.0, size)
        arr[:, 1] = np.clip(arr[:, 1] - y0, 0.0, size)
        shifted.append(arr.reshape(-1).tolist())
    return shifted, area, orig_area


def clip_annotations(annos, x0, y0, size, min_area=32.0, min_frac=0.1, edge_px=TRUNC_EDGE_PX):
    """Annotations in window coordinates, with the roof-union-footprint box,
    ``clip_frac`` and ``truncated`` (a roof within ``edge_px`` of the window's
    edge is one of the cut)."""
    rect = shp_box(x0, y0, x0 + size, y0 + size)
    out = []
    for a in annos:
        segm = a.get("segmentation")
        if not isinstance(segm, list) or len(segm) == 0:
            continue
        shifted, area, orig_area = _clip_poly_list(segm, rect, x0, y0, size)
        if shifted is None or area < min_area or area < min_frac * orig_area:
            continue
        xs = np.concatenate([np.asarray(q).reshape(-1, 2)[:, 0] for q in shifted])
        ys = np.concatenate([np.asarray(q).reshape(-1, 2)[:, 1] for q in shifted])
        bx0, by0, bx1, by1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        frac = float(area / orig_area) if orig_area > 0 else 1.0
        cut = (frac < TRUNC_FRAC or bx0 <= edge_px or by0 <= edge_px
               or bx1 >= size - edge_px or by1 >= size - edge_px)
        # the box spans roof and footprint (the footprint is the roof moved by
        # the offset), clipped to the window; a footprint beyond it is a cut
        dx, dy = (a.get("bd_offset") or offset_vector(a))
        bx0, bx1 = min(bx0, bx0 + dx), max(bx1, bx1 + dx)
        by0, by1 = min(by0, by0 + dy), max(by1, by1 + dy)
        cut = cut or bx0 < 0.0 or by0 < 0.0 or bx1 > float(size) or by1 > float(size)
        bx0, by0 = max(0.0, bx0), max(0.0, by0)
        bx1, by1 = min(float(size), bx1), min(float(size), by1)
        nb = copy.copy(a)
        nb["segmentation"] = shifted
        nb["bbox"] = [bx0, by0, max(bx1 - bx0, 1e-3), max(by1 - by0, 1e-3)]
        nb["area"] = float(area)
        nb["clip_frac"] = frac
        nb["truncated"] = bool(cut)
        nb["bd_offset"] = a.get("bd_offset") or offset_vector(a)
        out.append(nb)
    return out


def polygons_to_bitmask(polys, h, w):
    rles = mask_util.frPyObjects(polys, h, w)
    return mask_util.decode(mask_util.merge(rles)).astype(bool)


def _area_resize(chw, out):
    """(C, H, W) float array or tensor -> (C, out, out) tensor, area-averaged."""
    t = torch.as_tensor(np.ascontiguousarray(chw) if isinstance(chw, np.ndarray) else chw, dtype=torch.float32)
    return F.interpolate(t[None], size=(out, out), mode="area")[0]


def _empty_instances(size):
    inst = Instances((size, size))
    inst.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
    inst.gt_classes = torch.zeros((0,), dtype=torch.int64)
    inst.gt_masks = torch.zeros((0, size, size), dtype=torch.bool)
    inst.gt_heights = torch.zeros((0,), dtype=torch.float32)
    inst.gt_offsets = torch.zeros((0, 2), dtype=torch.float32)
    inst.gt_truncated = torch.zeros((0,), dtype=torch.bool)
    return inst


def _crop_window(arr, x0, y0, w):
    """arr (C, H, W) -> its square window [x0, x0 + w) x [y0, y0 + w) in arr's px at
    round(w) px: a slice when the window is on arr's grid (NaN outside arr), else
    bilinear."""
    n = max(1, int(round(w)))
    if all(abs(v - round(v)) < 1e-6 for v in (x0, y0, w)):
        a, b = int(round(y0)), int(round(x0))
        out = np.full((arr.shape[0], n, n), np.nan, np.float32)
        ys, xs = max(a, 0), max(b, 0)
        ye, xe = min(a + n, arr.shape[1]), min(b + n, arr.shape[2])
        if ye > ys and xe > xs:
            out[:, ys - a:ye - a, xs - b:xe - b] = arr[:, ys:ye, xs:xe]
        return out
    c = (torch.arange(n, dtype=torch.float32) + 0.5) * (w / n)
    gx = (x0 + c) / arr.shape[2] * 2 - 1
    gy = (y0 + c) / arr.shape[1] * 2 - 1
    gyy, gxx = torch.meshgrid(gy, gx, indexing="ij")
    grid = torch.stack([gxx, gyy], -1)[None]
    t = torch.as_tensor(np.ascontiguousarray(arr), dtype=torch.float32)[None]
    return F.grid_sample(t, grid, mode="bilinear", align_corners=False, padding_mode="border")[0].numpy()


class WISHDatasetMapper:
    @configurable
    def __init__(self, is_train, *, crop_size, crop_retry, min_inst_area, min_inst_frac,
                 brightness, contrast, read_sar, tile_scale=0.0, insar_root="", geo_aug="",
                 speckle=None, trunc_px=TRUNC_EDGE_PX):
        self.is_train = is_train
        # INPUT.TRUNC_PX: a roof within this many px of the training window's edge is cut
        self.trunc_px = float(trunc_px)
        self.crop_size = int(crop_size)
        self.tile_scale = float(tile_scale)
        self.crop_retry = int(crop_retry)
        self.min_inst_area = float(min_inst_area)
        self.min_inst_frac = float(min_inst_frac)
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.read_sar = bool(read_sar)
        # InSAR tiles (MODEL.WISHNET.INSAR): <insar_root>/<split>/<GF-3 file name>
        self.insar_root = str(insar_root) if self.read_sar else ""
        # INPUT.GEO_AUG: the transforms a training sample is drawn from (() = off)
        if geo_aug and geo_aug not in GEO_AUG_SETS:
            raise ValueError("INPUT.GEO_AUG must be '' or one of %s, got %r" % (sorted(GEO_AUG_SETS), geo_aug))
        self.geo_names = GEO_AUG_SETS[geo_aug] if (geo_aug and is_train) else ()
        # MODEL.WISHNET.SAR.SPECKLE: the arguments of speckle_resample plus "prob" (None = off)
        self.speckle = dict(speckle) if (speckle and is_train and self.read_sar) else None
        logger.info("[WISHDatasetMapper] train=%s crop=%d sar=%s insar=%s scale=%g geo=%s", is_train,
                    self.crop_size, self.read_sar, bool(self.insar_root), self.tile_scale,
                    ",".join(self.geo_names) or "off")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        return {
            "is_train": is_train,
            "crop_size": cfg.INPUT.CROP_SIZE if is_train else (cfg.INPUT.EVAL_CROP_SIZE
                                                               or cfg.INPUT.CROP_SIZE),
            "crop_retry": cfg.INPUT.CROP_RETRY,
            "min_inst_area": cfg.INPUT.MIN_INST_AREA,
            "min_inst_frac": cfg.INPUT.MIN_INST_FRAC,
            "brightness": cfg.INPUT.RGB_BRIGHTNESS if is_train else 0.0,
            "contrast": cfg.INPUT.RGB_CONTRAST if is_train else 0.0,
            "read_sar": cfg.MODEL.WISHNET.MODE == "fusion",
            "tile_scale": cfg.INPUT.TILE_SCALE,
            "insar_root": ((cfg.MODEL.WISHNET.INSAR.ROOT or os.path.join(cfg.DATASETS.BDH_ROOT, "s1_insar_asc"))
                           if cfg.MODEL.WISHNET.INSAR.ENABLED else ""),
            "geo_aug": cfg.INPUT.GEO_AUG if is_train else "",
            "speckle": _speckle_args(cfg) if is_train else None,
            "trunc_px": cfg.INPUT.TRUNC_PX,
        }

    def _pick_crop(self, d):
        """(x0, y0, annotations in window px)."""
        size = self.crop_size
        annos = d.get("annotations", [])
        if not self.is_train:
            cb = d["crop_box"]
            return (0, 0, annos) if cb is None else (int(cb[0]), int(cb[1]), annos)
        best = None
        for _ in range(max(1, self.crop_retry)):
            x0 = int(np.random.randint(0, max(d["parent_width"] - size, 0) + 1))
            y0 = int(np.random.randint(0, max(d["parent_height"] - size, 0) + 1))
            kept = clip_annotations(annos, x0, y0, size, self.min_inst_area, self.min_inst_frac, self.trunc_px)
            if kept:
                return x0, y0, kept
            if best is None:
                best = (x0, y0, kept)
        return best

    def _jitter_rgb(self, img):
        if self.brightness > 0:
            img = img * (1.0 + np.random.uniform(-self.brightness, self.brightness))
        if self.contrast > 0:
            m = img.mean()
            img = (img - m) * (1.0 + np.random.uniform(-self.contrast, self.contrast)) + m
        return np.clip(img, 0.0, 255.0)

    def __call__(self, dataset_dict):
        d = copy.deepcopy(dataset_dict)
        size = self.crop_size
        rgb = tifffile.imread(d["file_name"])
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        rgb = rgb[..., :3]
        x0, y0, annos = self._pick_crop(d)
        win = rgb[y0:y0 + size, x0:x0 + size, :].astype(np.float32)
        if win.shape[0] != size or win.shape[1] != size:          # tile smaller than the window
            pad = np.zeros((size, size, 3), dtype=np.float32)
            pad[:win.shape[0], :win.shape[1]] = win
            win = pad
        if self.is_train:
            win = self._jitter_rgb(win)
        # TILE_SCALE: what goes in is the window resampled to size_in
        size_in = int(round(size * self.tile_scale)) if self.tile_scale > 0 else size
        k = size_in / float(size)
        if size_in != size:
            d["image"] = _area_resize(win.transpose(2, 0, 1), size_in)
        else:
            d["image"] = torch.as_tensor(np.ascontiguousarray(win.transpose(2, 0, 1)))

        if self.read_sar:
            sar = tifffile.imread(d["sar_file_name"]).astype(np.float32)
            if sar.ndim == 3:
                sar = sar[..., 0]
            if not np.isfinite(sar).all():
                sar = np.where(np.isfinite(sar), sar, np.nanmean(sar))
            ph, pw = sar.shape
            # the SAR window is the optical one
            sx, sy, swin = x0, y0, size
            ys = np.clip(np.arange(sy, sy + swin), 0, ph - 1)
            xs = np.clip(np.arange(sx, sx + swin), 0, pw - 1)
            sw = np.ascontiguousarray(sar[np.ix_(ys, xs)])[None]
            if self.speckle is not None and np.random.rand() < self.speckle["prob"]:
                sw = speckle_resample(sw, **{k: v for k, v in self.speckle.items() if k != "prob"})
            if size_in != size:          # the SAR window resampled by the same factor
                d["sar_image"] = _area_resize(sw, int(round(swin * k)))
            else:
                d["sar_image"] = torch.as_tensor(sw)
            if self.insar_root:
                # the InSAR tile over the same window (it covers the whole tile:
                # Sentinel-1 at 128 px, 1/16 of the tile's px), NaN = no data
                sp = d["sar_file_name"]
                ins = tifffile.imread(os.path.join(self.insar_root, os.path.basename(os.path.dirname(sp)),
                                                   os.path.basename(sp))).astype(np.float32)
                if ins.ndim == 3 and ins.shape[-1] < ins.shape[0]:
                    ins = ins.transpose(2, 0, 1)                      # (H, W, C) -> (C, H, W)
                f = ins.shape[-1] / float(pw)
                d["insar_image"] = torch.as_tensor(np.ascontiguousarray(_crop_window(ins, sx * f, sy * f, swin * f)))

        d["height"], d["width"] = size, size
        d.pop("annotations", None)
        masks, boxes, classes, heights, offsets, truncs = [], [], [], [], [], []
        for a in annos:
            segs = a["segmentation"] if k == 1.0 else [
                (np.asarray(q, dtype=np.float64) * k).tolist() for q in a["segmentation"]]
            m = polygons_to_bitmask(segs, size_in, size_in)
            if m.sum() < 1:
                continue
            ys_, xs_ = np.nonzero(m)
            bx0, by0, bw, bh = (float(v) * k for v in a["bbox"])
            # the box contains the roof whatever the annotation said
            boxes.append([min(bx0, float(xs_.min())), min(by0, float(ys_.min())),
                          max(bx0 + bw, float(xs_.max() + 1)), max(by0 + bh, float(ys_.max() + 1))])
            masks.append(m)
            classes.append(a["category_id"])
            heights.append(a["bd_height"])
            o = a.get("bd_offset") or offset_vector(a)
            offsets.append([float(o[0]) * k, float(o[1]) * k])
            truncs.append(bool(a.get("truncated", False)) or not a.get("bd_height_valid", True))
        if not masks:
            d["instances"] = _empty_instances(size_in)
        else:
            inst = Instances((size_in, size_in))
            inst.gt_boxes = Boxes(torch.as_tensor(np.array(boxes), dtype=torch.float32))
            inst.gt_classes = torch.as_tensor(classes, dtype=torch.int64)
            inst.gt_masks = torch.as_tensor(np.stack(masks), dtype=torch.bool)
            inst.gt_heights = torch.as_tensor(heights, dtype=torch.float32)
            inst.gt_offsets = torch.as_tensor(np.asarray(offsets, dtype=np.float32))
            inst.gt_truncated = torch.as_tensor(truncs, dtype=torch.bool)
            d["instances"] = inst
        if self.geo_names:
            self._geo(d, size_in)
        return d

    def _geo(self, d, size_in, name=None):
        """INPUT.GEO_AUG: one transform, drawn uniformly (or ``name``), for the
        whole sample: the optical input, the SAR window and the InSAR tile each
        in its own frame, the masks, boxes and offsets of the instances;
        ``geo_M`` its linear part."""
        if name is None:
            name = self.geo_names[np.random.randint(len(self.geo_names))]
        tr = GEO_TRANSFORMS[name]
        m = geo_matrix(tr)
        d["geo_M"] = torch.as_tensor(m, dtype=torch.float32)
        if not any(tr):
            return
        d["image"] = geo_tensor(d["image"], tr)
        if "sar_image" in d:
            d["sar_image"] = geo_tensor(d["sar_image"], tr)
        if "insar_image" in d:
            d["insar_image"] = geo_tensor(d["insar_image"], tr)
        inst = d["instances"]
        if len(inst) == 0:
            inst.gt_masks = geo_tensor(inst.gt_masks, tr)
            return
        inst.gt_masks = geo_tensor(inst.gt_masks, tr)
        b = inst.gt_boxes.tensor.numpy().astype(np.float64)
        c0, c1 = geo_points(b[:, :2], tr, size_in), geo_points(b[:, 2:], tr, size_in)
        inst.gt_boxes = Boxes(torch.as_tensor(np.concatenate([np.minimum(c0, c1), np.maximum(c0, c1)], 1),
                                              dtype=torch.float32))
        inst.gt_offsets = (inst.gt_offsets.double() @ torch.as_tensor(m).T).float()
