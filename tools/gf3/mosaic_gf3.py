# -*- coding: utf-8 -*-
"""Two geocoded GF-3 scenes -> one mosaic on the BDH SAR grid + the one-channel product.

Only one acquisition time exists (two scenes five days apart that together
cover the AOI), so there is no temporal averaging; speckle is reduced
spatially instead, and the single-look intensity is kept as its own band.

Scene priority: 20240406 first (93.7 % of the grid, incidence 36-38 deg, the
same geometry as the S1 descending product), 20240401 fills the rest (the NW
corner, incidence 28-29 deg).  There is no feathering: the two geometries lay a
building over by different lengths (k_rd 3.2 vs 4.4 px/m), so a blended seam
would show every building on it twice.  Each pixel comes from one scene and
``gf3_source.tif`` says which (1 = 20240406, 2 = 20240401).

The product has ONE channel (the data are single-polarisation HH, one date;
the user asked for one channel rather than three derived from the same HH).
Three candidate processings exist and are written together, as a 3-band
evaluation file, so measure_rd.py can compare them on the annotations:

  0  hh_sl_db     single-look sigma0 HH, dB (zeros, i.e. quantised dark
                  pixels, and anything below -35 dB are set to -35 dB)
  1  hh_lee7_db   Lee filter, 7 x 7, single-look multiplicative noise model
  2  hh_box3_db   3 x 3 boxcar mean of sigma0 (5 m)

``--single NAME`` writes a one-band product instead, as uint8 (gf3_hh_uint8.tif,
the S1 products' encoding: DN 0 = nodata, DN 1..255 linear over [lo, hi] =
p0.1 / p99.9 of the valid pixels) or, with ``--float``, as the physical value
itself: gf3_<NAME>.tif, float32 dB, NaN = nodata.

THE PRODUCT (user's choice, 2026-09-28): ``--single hh_sl_db --float``, the
single-look backscatter coefficient in dB, stored as float32.  The 3 x 3 mean
looked visibly softer (5 m against 3 m), measure_rd.py shows the same building
signal for both and the fuller layover extent for single-look, and the user
wants the network to see the calibrated sigma0 values, not an image encoding.

Radiometry is calibrated sigma0 (unlike the S1 products); the step between the
two scenes in their overlap is measured and reported, not removed.

    python tools/gf3/mosaic_gf3.py --geo <gf3>/geo --grid <grid json> --out <gf3>/tmp/eval_all  # 3 candidates
    python tools/gf3/mosaic_gf3.py --geo <gf3>/geo --grid <grid json> --only 20240401 \
        --scales-from .../SAR_BANDS_gf3.json --no-float --out <gf3>/tmp/scene_0401          # one scene
    python tools/gf3/mosaic_gf3.py --geo <gf3>/geo --grid <grid json> --single hh_sl_db --float \
        --out <gf3>/gf3_product
"""
import argparse
import json
import os
import time

import numpy as np
from osgeo import gdal, osr
from scipy.ndimage import uniform_filter

gdal.UseExceptions()

ORDER = ["20240406", "20240401"]
FLOOR_DB = -35.0
PCT = (0.1, 1.0, 50.0, 99.0, 99.9)
BANDS = [("hh_sl_db", "single-look sigma0 HH, dB, floor -35 dB"),
         ("hh_lee7_db", "Lee filter 7x7 (single-look noise model) of sigma0 HH, dB"),
         ("hh_box3_db", "3x3 boxcar mean of sigma0 HH (5 m), dB")]


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def chunked(fn, a, valid, halo, rows=2048):
    """Apply a neighbourhood filter row-chunk by row-chunk with a halo."""
    out = np.full(a.shape, np.nan, np.float32)
    for r0 in range(0, a.shape[0], rows):
        r1 = min(a.shape[0], r0 + rows)
        h0, h1 = max(0, r0 - halo), min(a.shape[0], r1 + halo)
        res = fn(a[h0:h1], valid[h0:h1])
        out[r0:r1] = res[r0 - h0:r0 - h0 + (r1 - r0)]
    return out


def box(a, valid, size):
    num = uniform_filter(np.where(valid, a, 0).astype(np.float32), size, mode="nearest")
    den = uniform_filter(valid.astype(np.float32), size, mode="nearest")
    return np.where(valid, num / np.maximum(den, 1e-6), np.nan)


def lee(a, valid, size=7, looks=1.0):
    x = np.where(valid, a, 0).astype(np.float32)
    den = np.maximum(uniform_filter(valid.astype(np.float32), size, mode="nearest"), 1e-6)
    m = uniform_filter(x, size, mode="nearest") / den
    m2 = uniform_filter(x * x, size, mode="nearest") / den
    v = np.maximum(m2 - m * m, 0)
    cu2 = 1.0 / looks
    varx = np.maximum((v - m * m * cu2) / (1 + cu2), 0)
    k = varx / np.maximum(v, 1e-20)
    return np.where(valid, m + k * (x - m), np.nan)


def to_db(a):
    return (10 * np.log10(np.maximum(a, 10 ** (FLOOR_DB / 10)))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", required=True, help="geocode_gf3.py's --out: <date>_sigma0.tif, <date>_geocode.json")
    ap.add_argument("--grid", required=True,
                    help="json of the SAR grid: grid.width, grid.height, grid.geotransform_3857")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="one scene only (for per-scene checks)")
    ap.add_argument("--scales-from", default=None, help="reuse lo/hi of an existing product json")
    ap.add_argument("--no-float", action="store_true", help="skip the float32 sigma0 mosaic")
    ap.add_argument("--single", default=None, choices=[b[0] for b in BANDS],
                    help="write the one-channel product with this processing")
    ap.add_argument("--float", action="store_true",
                    help="with --single: write float32 dB (NaN = nodata) instead of uint8")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    order = [a.only] if a.only else ORDER
    bands = [b for b in BANDS if a.single is None or b[0] == a.single]

    g = json.load(open(a.grid, encoding="utf-8"))["grid"]
    gt = g["geotransform_3857"]
    nx, ny = int(g["width"]), int(g["height"])
    s0 = np.full((ny, nx), np.nan, np.float32)
    src = np.zeros((ny, nx), np.uint8)
    scenes = {}
    for k, d in enumerate(order, 1):
        gj = json.load(open(os.path.join(a.geo, "%s_geocode.json" % d), encoding="utf-8"))
        w = gj["grid_window"]
        ds = gdal.Open(os.path.join(a.geo, "%s_sigma0.tif" % d))
        arr = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        sl = (slice(w["iy0"], w["iy0"] + w["height"]), slice(w["ix0"], w["ix0"] + w["width"]))
        take = np.isfinite(arr) & (src[sl] == 0)
        s0[sl][take] = arr[take]
        src[sl][take] = k
        scenes[d] = {"source_id": k, "scene": gj["scene"], "centre_time_utc": gj["centre_time_utc"],
                     "geometry": gj["geometry"], "calibration": gj["calibration"],
                     "echo_saturation_pct": gj["echo_saturation_pct"],
                     "doppler_convention": gj["doppler_convention"],
                     "shift_az_m": gj["shift_az_m"], "shift_rg_m": gj["shift_rg_m"],
                     "valid_fraction_of_grid": gj["valid_fraction_of_grid"],
                     "pixels_in_mosaic_fraction": float((src == k).mean())}
        log("%s: %.2f%% of the grid covered, %.2f%% taken into the mosaic"
            % (d, 100 * gj["valid_fraction_of_grid"], 100 * (src == k).mean()))
        # seam: radiometric step against what earlier scenes already hold
        if k > 1:
            both = np.isfinite(arr) & (src[sl] != k) & (src[sl] > 0)
            if both.sum() > 10000:
                va = np.isfinite(arr)
                m_new = box(np.where(va, arr, 0), va, 15)
                vb = np.isfinite(s0[sl])
                m_old = box(np.where(vb, s0[sl], 0), vb, 15)
                dd = (to_db(m_new) - to_db(m_old))[both][::37]
                scenes[d]["step_db_vs_earlier_15x15_median"] = float(np.median(dd))
                scenes[d]["step_db_vs_earlier_15x15_p25_p75"] = np.percentile(dd, [25, 75]).tolist()
                scenes[d]["overlap_fraction_of_grid"] = float(both.sum() / float(nx * ny))
                log("%s vs earlier scenes in the overlap (%.1f%% of the grid): 15x15-mean step "
                    "median %+.2f dB [p25 %+.2f, p75 %+.2f]"
                    % (d, 100 * both.sum() / float(nx * ny), np.median(dd), *np.percentile(dd, [25, 75])))
                del m_new, m_old
        del arr
    valid = np.isfinite(s0)
    log("mosaic valid %.3f%% of the grid" % (100 * valid.mean()))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    drv = gdal.GetDriverByName("GTiff")
    geot = (gt[0], gt[1], 0.0, gt[3], 0.0, gt[5])

    def write(path, arrays, dtype, nodata, descs, opts):
        d = drv.Create(path, nx, ny, len(arrays), dtype, options=opts)
        d.SetProjection(srs.ExportToWkt())
        d.SetGeoTransform(geot)
        for i, (arr, desc) in enumerate(zip(arrays, descs), 1):
            b = d.GetRasterBand(i)
            b.WriteArray(arr)
            if nodata is not None:
                b.SetNoDataValue(nodata)
            b.SetDescription(desc)
        d = None

    if not a.no_float:
        write(os.path.join(a.out, "gf3_sigma0_mosaic.tif"), [s0], gdal.GDT_Float32, float("nan"),
              ["sigma0_linear_single_look_HH"],
              ["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=3", "BIGTIFF=IF_SAFER"])
        write(os.path.join(a.out, "gf3_source.tif"), [src], gdal.GDT_Byte, 0,
              ["source scene: " + ", ".join("%d=%s" % (v["source_id"], d) for d, v in scenes.items())],
              ["TILED=YES", "COMPRESS=DEFLATE"])

    # derived bands
    t0 = time.time()
    make = {"hh_sl_db": lambda: to_db(np.where(valid, s0, 0)),
            "hh_lee7_db": lambda: to_db(chunked(lambda x, v: lee(x, v, 7), s0, valid, 4)),
            "hh_box3_db": lambda: to_db(chunked(lambda x, v: box(x, v, 3), s0, valid, 2))}
    planes = [make[b[0]]() for b in bands]
    del s0
    for p in planes:
        p[~valid] = np.nan
    log("derived bands, %.0f s" % (time.time() - t0))

    if a.float:
        assert a.single, "--float needs --single"
        v = planes[0]
        rng = np.random.default_rng(0)
        sample = v[valid][rng.integers(0, int(valid.sum()), 20_000_000)]
        pv = np.percentile(sample, (0.1, 2, 50, 98, 99.9))
        out_name = "gf3_%s.tif" % a.single
        write(os.path.join(a.out, out_name), [v], gdal.GDT_Float32, float("nan"), [a.single],
              ["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=3", "BIGTIFF=IF_SAFER"])
        log("%s float32: p0.1 %.2f  p2 %.2f  median %.2f  p98 %.2f  p99.9 %.2f dB (all splits)" % ((a.single,) + tuple(pv)))
        del planes, v
        main_scene = scenes[order[0]]
        meta = {
            "product": "GF-3 UFS single-look HH backscatter coefficient sigma0 in dB, single acquisition time",
            "grid": g, "scenes": scenes, "priority": order, "file": out_name,
            "bands": [{"index": 0, "name": a.single, "units": "dB", "desc": dict(BANDS)[a.single]}],
            "encoding": {"dtype": "float32", "units": "dB", "value": "10*log10(sigma0), calibrated",
                         "nodata": "NaN", "floor_db": FLOOR_DB,
                         "floor_note": "quantised zeros of the SLC (dark pixels) and anything below the floor",
                         "percentiles_all_splits_db": dict(zip(["p0.1", "p2", "p50", "p98", "p99.9"],
                                                               map(float, pv)))},
            "normalisation": "use the TRAIN statistics (tile_summary_gf3.json / train_statistics below): "
                             "z-score, or (x - p2) / (p98 - p2) WITHOUT clipping the upper tail -- "
                             "3/4 of the pixels above the train p98 lie on building footprints and "
                             "their layover",
            "radiometry": "calibrated sigma0 (GF-3 L1A: QualifyValue and CalibrationConst); no incidence "
                          "normalisation; the step between the scenes is reported under scenes, not removed",
            "geometry": main_scene["geometry"],
            "valid_fraction": float(valid.mean()),
        }
        with open(os.path.join(a.out, "SAR_BANDS_gf3.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        log("wrote %s" % a.out)
        return

    if a.scales_from:
        prev = json.load(open(a.scales_from, encoding="utf-8"))["encoding"]["scales"]
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(valid.ravel())
    sample = idx if idx.size <= 20_000_000 else rng.choice(idx, 20_000_000, replace=False)
    q8 = np.zeros((len(bands), ny, nx), np.uint8)
    scales = {}
    for k, (name, _) in enumerate(bands):
        v = planes[k]
        planes[k] = None
        pv = np.percentile(v.ravel()[sample], PCT)
        if a.scales_from:
            lo, hi = prev[name]["lo"], prev[name]["hi"]
        else:
            lo, hi = float(pv[0]), float(pv[-1])
        v -= np.float32(lo)
        v *= np.float32(254.0 / (hi - lo))
        np.round(v, out=v)
        np.clip(v, 0, 254, out=v)                  # NaN stays NaN
        q8[k] = np.where(valid, np.nan_to_num(v, nan=0) + 1, 0).astype(np.uint8)
        del v
        vq = q8[k].ravel()[sample]
        scales[name] = {"lo": lo, "hi": hi, "p1": float(pv[1]), "median": float(pv[2]), "p99": float(pv[3]),
                        "clipped_low_pct": float(100 * (vq == 1).mean()),
                        "clipped_high_pct": float(100 * (vq == 255).mean())}
        log("band %d %-11s lo %7.2f  p1 %7.2f  median %7.2f  p99 %7.2f  hi %7.2f dB"
            % (k, name, lo, pv[1], pv[2], pv[3], hi))
    del planes
    out_name = "gf3_hh_uint8.tif" if a.single else "gf3_3ch_uint8.tif"
    write(os.path.join(a.out, out_name), list(q8), gdal.GDT_Byte, 0, [b[0] for b in bands],
          ["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2", "INTERLEAVE=PIXEL", "BIGTIFF=IF_SAFER"])
    del q8

    main_scene = scenes[order[0]]
    meta = {
        "product": "GF-3 UFS single-look HH, calibrated sigma0, single acquisition time",
        "grid": g,
        "scenes": scenes,
        "priority": order,
        "file": out_name,
        "bands": [{"index": i, "name": n, "scale_key": n, "desc": dsc} for i, (n, dsc) in enumerate(bands)],
        "encoding": {"dn0": "nodata", "formula": "value = lo + (DN - 1) / 254 * (hi - lo)",
                     "lo_hi": "percentiles 0.1 / 99.9 of the valid pixels of the %s product"
                              % ("two-scene" if not a.scales_from else "reference"),
                     "floor_db": FLOOR_DB, "scales": scales},
        "radiometry": "calibrated sigma0 (GF-3 L1A: QualifyValue and CalibrationConst); no incidence "
                      "normalisation; the step between the scenes is reported under scenes, not removed",
        # the geometry block measure_rd.py reads: the priority scene's (it covers most tiles)
        "geometry": main_scene["geometry"],
        "valid_fraction": float(valid.mean()),
    }
    with open(os.path.join(a.out, "SAR_BANDS_gf3.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log("wrote %s" % a.out)


if __name__ == "__main__":
    main()
