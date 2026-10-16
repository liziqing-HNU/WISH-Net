# -*- coding: utf-8 -*-
"""Cut a Sentinel-1 InSAR mosaic (make_s1insar.py) into WISH-Net tiles.

For every image of the BDH-RGB-SAR geographic split (annotations_fixed/instances_*.json)
the mosaic is averaged (gdal average, NaN = no data) onto a 128 x 128 grid over the
tile's own extent -- 16 dataset pixels, 8.64 EPSG:3857 units, 6.6 m on the ground; S1's
resolution after the 5 x 2 looks is about 17 x 28 m -- and written as float32 bands (17
ascending, 15 descending) to <out>/<split>/<sar_file_name>.  The file name is the GF-3
tile's, so tools/labels/make_orig_split.py --sar-dir s1_insar_<track> links the original
split as well.

Normalisation statistics (mean and std of every band over valid pixels, no clipping)
come from training tiles only, for both splits: geographic (instances_train) and BDH's
original (metadata/tiles_index.csv, split bdh_train).  They are written with the
mosaic's metadata to <out>/S1_INSAR_<TRACK>.json (S1_INSAR_ASC.json, S1_INSAR_DESC.json);
per-tile coverage goes to <out>/tile_quality.csv.

    python tools/s1insar/tile_s1insar.py --mosaic <s1insar>/s1insar_asc_mosaic.tif --root <BDH-RGB-SAR>
    (--track desc --mosaic <s1insar_desc>/s1insar_desc_mosaic.tif for the descending stack)
"""
import argparse
import csv
import json
import os

import numpy as np
from osgeo import gdal

gdal.UseExceptions()
TRACKS = ("asc", "desc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=TRACKS, default="asc")
    ap.add_argument("--mosaic", required=True,
                    help="s1insar_<track>_mosaic.tif of make_s1insar.py (its .json next to it)")
    ap.add_argument("--root", required=True, help="BDH-RGB-SAR: annotations_fixed/, metadata/tiles_index.csv")
    ap.add_argument("--out", default="", help="default: <root>/s1_insar_<track>")
    ap.add_argument("--px", type=int, default=128)
    a = ap.parse_args()
    a.out = a.out or os.path.join(a.root, "s1_insar_" + a.track)
    meta = json.load(open(a.mosaic[:-4] + ".json"))
    nb = len(meta["channels"])
    orig = {}
    with open(os.path.join(a.root, "metadata", "tiles_index.csv")) as f:
        for row in csv.DictReader(f):
            orig[os.path.splitext(row["file_name"])[0]] = row["split"]
    src = gdal.Open(a.mosaic)
    acc = {k: [np.zeros(nb), np.zeros(nb), 0] for k in ("train_geo", "train_orig")}
    rows, count = [], {}
    for split in ("train", "val", "test"):
        d = json.load(open(os.path.join(a.root, "annotations_fixed", "instances_%s.json" % split)))
        os.makedirs(os.path.join(a.out, split), exist_ok=True)
        for im in d["images"]:
            gt, w, h = im["geotransform_3857"], im["width"], im["height"]
            x0, y1 = gt[0], gt[3]
            x1, y0 = x0 + w * gt[1], y1 + h * gt[5]
            dst = os.path.join(a.out, split, im["sar_file_name"])
            gdal.Warp(dst, src, outputBounds=(x0, y0, x1, y1), width=a.px, height=a.px,
                      resampleAlg="average", srcNodata=float("nan"), dstNodata=float("nan"),
                      outputType=gdal.GDT_Float32,
                      creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=3"])
            arr = gdal.Open(dst).ReadAsArray().astype(np.float64)
            ok = np.isfinite(arr).all(0)
            stem = os.path.splitext(im["file_name"])[0]
            for key, use in (("train_geo", split == "train"),
                             ("train_orig", orig.get(stem) == "bdh_train")):
                if use and ok.any():
                    v = arr[:, ok]
                    acc[key][0] += v.sum(1)
                    acc[key][1] += (v ** 2).sum(1)
                    acc[key][2] += v.shape[1]
            coh = float(np.nanmean(arr[meta["channels"].index("coh_mean")])) if ok.any() else float("nan")
            rows.append([split, im["sar_file_name"], "%.4f" % ok.mean(), "%.3f" % coh])
            count[split] = count.get(split, 0) + 1
        print("%s: %d tiles" % (split, count[split]), flush=True)
    norm = {}
    for key, (s1, s2, n) in acc.items():
        mean = s1 / n
        norm[key] = {"mean": mean.tolist(), "std": np.sqrt(np.maximum(s2 / n - mean ** 2, 0)).tolist(),
                     "pixels": int(n)}
    with open(os.path.join(a.out, "tile_quality.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "file", "valid_fraction", "coh_mean"])
        wr.writerows(rows)
    meta["tiles"] = {"px": a.px, "pixel_ground_m": None, "count": count,
                     "extent": "each tile's own extent (images[].geotransform_3857)",
                     "file_name": "<split>/<GF-3 sar_file_name>", "nodata": "NaN"}
    gsd = [im.get("gsd_m") for im in d["images"] if im.get("gsd_m")]
    meta["tiles"]["pixel_ground_m"] = float(np.median(gsd) * 2048 / a.px) if gsd else None
    meta["normalisation"] = norm
    with open(os.path.join(a.out, "S1_INSAR_%s.json" % a.track.upper()), "w") as f:
        json.dump(meta, f, indent=1)
    bad = [r for r in rows if float(r[2]) < 0.99]
    print("tiles with < 99%% valid pixels: %d" % len(bad))
    for r in bad[:20]:
        print("  ", r)
    for key in norm:
        print(key, "mean", np.round(norm[key]["mean"], 3).tolist())
        print(key, "std ", np.round(norm[key]["std"], 3).tolist())


if __name__ == "__main__":
    main()
