# -*- coding: utf-8 -*-
"""Per-tile summary of the GF-3 tile set: coverage, source scene, value statistics.

Reads <tiles>/sar_gf3/{split}/ (the one-channel product: float32 sigma0 in dB,
NaN = nodata; a uint8 product is decoded with the json's lo/hi) and
<tiles>/sar_gf3_source/{split}/ (1 = 20240406, 2 = 20240401), <tiles> = --tiles,
and writes <tiles>/tile_summary_gf3.json:

  * per tile: valid fraction, fraction from each scene;
  * per split: tiles fully from one scene, tiles crossing the seam, tiles with nodata;
  * per split, in dB over the valid pixels: mean, std and the percentiles
    0.1 / 2 / 50 / 98 / 99.9 -- the TRAIN ones are what input normalisation uses.

The train statistics are also written into the product json next to the tiles
(``train_statistics``; --meta, default <tiles>/sar_gf3/SAR_BANDS_gf3.json).

    python tools/gf3/tile_summary_gf3.py --tiles <tiles>
"""
import argparse
import glob
import json
import os

import numpy as np
import tifffile

PCT = (0.1, 2, 50, 98, 99.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, help="the folder holding sar_gf3/ and sar_gf3_source/")
    ap.add_argument("--meta", default=None, help="default <tiles>/sar_gf3/SAR_BANDS_gf3.json")
    a = ap.parse_args()
    a.meta = a.meta or os.path.join(a.tiles, "sar_gf3", "SAR_BANDS_gf3.json")
    meta = json.load(open(a.meta, encoding="utf-8"))
    band = meta["bands"][0]["name"]
    enc = meta["encoding"]
    out = {"band": band, "units": "dB", "splits": {}, "tiles": {}}
    rng = np.random.default_rng(0)
    for sp in ("train", "val", "test"):
        files = sorted(glob.glob(os.path.join(a.tiles, "sar_gf3", sp, "*.tif")))
        s1 = s2 = 0.0
        n = 0
        sample = []
        full = {"20240406": 0, "20240401": 0}
        seam, partial = [], []
        for f in files:
            name = os.path.basename(f)
            x = tifffile.imread(f)
            if np.issubdtype(x.dtype, np.floating):
                v = np.isfinite(x)
                db = x[v].astype(np.float64)
            else:
                sc = enc["scales"][band]
                v = x > 0
                db = sc["lo"] + (x[v].astype(np.float64) - 1) / 254.0 * (sc["hi"] - sc["lo"])
            src = tifffile.imread(os.path.join(a.tiles, "sar_gf3_source", sp, name))
            fr = {"valid": float(v.mean()), "20240406": float((src == 1).mean()),
                  "20240401": float((src == 2).mean())}
            out["tiles"]["%s/%s" % (sp, name)] = fr
            if fr["valid"] < 1.0:
                partial.append(name)
            if fr["20240406"] > 0.999:
                full["20240406"] += 1
            elif fr["20240401"] > 0.999:
                full["20240401"] += 1
            else:
                seam.append({"tile": name, "frac_20240401": round(fr["20240401"], 4)})
            s1 += db.sum()
            s2 += (db * db).sum()
            n += db.size
            sample.append(rng.choice(db, 20000, replace=False))
        m = s1 / n
        p = np.percentile(np.concatenate(sample), PCT)
        out["splits"][sp] = {
            "tiles": len(files), "tiles_with_nodata": partial,
            "tiles_fully_from": full, "tiles_crossing_seam": seam,
            "mean_db": m, "std_db": float(np.sqrt(s2 / n - m * m)),
            "percentiles_db": dict(zip(["p0.1", "p2", "p50", "p98", "p99.9"], map(float, p)))}
        print("%-5s %3d tiles  fully 04-06 %3d  fully 04-01 %2d  crossing the seam %2d  nodata tiles %d  "
              "mean %.2f std %.2f dB  p0.1 %.2f p2 %.2f p50 %.2f p98 %.2f p99.9 %.2f"
              % (sp, len(files), full["20240406"], full["20240401"], len(seam), len(partial), m,
                 out["splits"][sp]["std_db"], *p))
    with open(os.path.join(a.tiles, "tile_summary_gf3.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    tr = out["splits"]["train"]
    meta["train_statistics"] = {"mean_db": tr["mean_db"], "std_db": tr["std_db"],
                                "percentiles_db": tr["percentiles_db"]}
    with open(a.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
