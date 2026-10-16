# -*- coding: utf-8 -*-
"""Cut a GF-3 3-channel uint8 mosaic into the BDH tile grid.

The resampling is make_tiles_desc.py's, line for line (one bilinear
cv2.warpAffine per tile from the mosaic grid onto the tile's own
geotransform_3857, pixel-centre term included, validity mask warped alongside
and any pixel that picked up nodata written back to DN 0), so a GF-3 tile covers
exactly the ground of the S1 tile with the same name.  Unlike that script, the
output folder is separate from the dataset root and nothing in the dataset is
written: tiles go to <out>/{split}/<sar_file_name>, the product json is copied
next to them.

``--min-valid`` skips tiles whose valid fraction is below it (for the
single-scene sets that only feed measure_rd.py).  ``--interp nearest`` is for
label rasters: gf3_source.tif (1 = 20240406, 2 = 20240401) is cut that way
into per-tile source maps, so a model can take the imaging geometry per pixel.
A float mosaic (the product: sigma0 in dB, NaN = nodata) gives float32 tiles,
DEFLATE with the floating-point predictor, NaN = nodata.

    python tools/gf3/make_tiles_gf3.py --mosaic <gf3>/gf3_product/gf3_3ch_uint8.tif \
        --meta <gf3>/gf3_product/SAR_BANDS_gf3.json --root <BDH-RGB-SAR> \
        --out <tiles>/sar_gf3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import cv2
import numpy as np
import tifffile

SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mosaic", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--root", required=True, help="dataset root (read only)")
    ap.add_argument("--coco", default=None, help="default <root>/annotations_fixed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    ap.add_argument("--min-valid", type=float, default=0.0)
    ap.add_argument("--interp", choices=("linear", "nearest"), default="linear")
    a = ap.parse_args()
    coco_dir = a.coco or os.path.join(a.root, "annotations_fixed")
    assert os.path.abspath(a.out) != os.path.abspath(a.root), "tiles do not go into the dataset root"

    meta = json.load(open(a.meta, encoding="utf-8"))
    g = meta["grid"]["geotransform_3857"]
    ox, tr, oy = g[0], g[1], g[3]
    mos = tifffile.imread(a.mosaic)
    if mos.ndim == 3 and mos.shape[0] == 3:
        mos = np.transpose(mos, (1, 2, 0))
    single = mos.ndim == 2                          # the one-channel product
    if single:
        mos = mos[..., None]
    H, W = mos.shape[:2]
    assert (W, H) == (meta["grid"]["width"], meta["grid"]["height"]), (W, H)
    is_float = np.issubdtype(mos.dtype, np.floating)  # physical values (dB), NaN = nodata
    if is_float:
        valid = np.isfinite(mos[:, :, 0]).astype(np.uint8) * 255
        mos = np.nan_to_num(mos, nan=0.0)             # never interpolated into a valid pixel, see below
    else:
        valid = (mos[:, :, 0] > 0).astype(np.uint8) * 255
    print("mosaic %d x %d x %d, valid %.2f%%" % (W, H, mos.shape[2], 100 * (valid > 0).mean()), flush=True)

    rows = []
    for sp in a.splits:
        coco = json.load(open(os.path.join(coco_dir, "instances_%s.json" % sp)))
        d = os.path.join(a.out, sp)
        os.makedirs(d, exist_ok=True)
        n_w = 0
        for img in coco["images"]:
            gt = img["geotransform_3857"]
            tw, th = img["width"], img["height"]
            s = gt[1] / tr
            u0 = (gt[0] - ox) / tr
            v0 = (oy - gt[3]) / tr
            cx0 = max(0, int(np.floor(u0)) - 2)
            cy0 = max(0, int(np.floor(v0)) - 2)
            cx1 = min(W, int(np.ceil(u0 + tw * s)) + 2)
            cy1 = min(H, int(np.ceil(v0 + th * s)) + 2)
            if cx1 <= cx0 or cy1 <= cy0:
                rows.append((sp, img["sar_file_name"], 0.0, False))
                continue
            off = 0.5 * s - 0.5
            M = np.array([[s, 0.0, u0 - cx0 + off], [0.0, s, v0 - cy0 + off]], np.float32)
            flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
            dflags = (cv2.INTER_NEAREST if a.interp == "nearest" else cv2.INTER_LINEAR) | cv2.WARP_INVERSE_MAP
            mv = cv2.warpAffine(valid[cy0:cy1, cx0:cx1], M, (tw, th), flags=flags,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            frac = float((mv >= 250).mean())
            if frac < a.min_valid or frac == 0:
                rows.append((sp, img["sar_file_name"], frac, False))
                continue
            out = cv2.warpAffine(mos[cy0:cy1, cx0:cx1], M, (tw, th), flags=dflags,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if out.ndim == 2:                       # cv2 drops a single channel axis
                out = out[..., None]
            if is_float:
                # a pixel is valid only where all four source samples were
                # (validity warped with the same kernel), so the 0 fill of
                # nodata never leaks into a value
                out = np.where((mv >= 250)[..., None], out, np.nan).astype(np.float32)
                tifffile.imwrite(os.path.join(d, img["sar_file_name"]), out[..., 0] if single else out,
                                 compression="deflate", predictor=3)
            else:
                out = np.where((mv >= 250)[..., None], out, 0).astype(np.uint8)
                out = np.where((mv >= 250)[..., None] & (out == 0), 1, out).astype(np.uint8)
                tifffile.imwrite(os.path.join(d, img["sar_file_name"]), out[..., 0] if single else out,
                                 compression="deflate")
            rows.append((sp, img["sar_file_name"], frac, True))
            n_w += 1
        print("%s: %d / %d tiles written -> %s" % (sp, n_w, len(coco["images"]), d), flush=True)

    cov = np.array([r[2] for r in rows])
    wr = np.array([r[3] for r in rows])
    print("written %d of %d; valid fraction of written tiles: min %.4f mean %.4f, below 0.99: %d"
          % (wr.sum(), len(rows), cov[wr].min(), cov[wr].mean(), int((cov[wr] < 0.99).sum())))
    with open(os.path.join(a.out, "tile_coverage.csv"), "w") as f:
        f.write("split,sar_file_name,valid_fraction,written\n")
        for r in rows:
            f.write("%s,%s,%.5f,%d\n" % (r[0], r[1], r[2], int(r[3])))
    shutil.copyfile(a.meta, os.path.join(a.out, os.path.basename(a.meta)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
