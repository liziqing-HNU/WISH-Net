# -*- coding: utf-8 -*-
"""Relative registration of a geocoded GF-3 scene against the S1 mosaics.

Both live on the same grid, so the offset is measured directly: 1024 x 1024
blocks (1.7 km), both images in dB, GF-3 boxcar-averaged 3 x 3 and both
area-downsampled by 2 (3.3 m) so the comparison is not dominated by detail
only GF-3 resolves, then normalised cross-correlation (FFT, Hann window) with
a parabolic sub-pixel peak.  The offset is GF-3 minus S1: a feature at p in
S1 sits at p + d in GF-3.  It is reported in metres and split into the
along-track (heading) and towards-sensor components of the GF-3 geometry.

The S1 descending product has nearly the same geometry as GF-3 (descending,
right-looking, towards-sensor bearing 100 deg, incidence 36.9 deg against
29.5 / 37.0 deg), so its layover lies on the same side and the comparison is
clean; against S1 ascending the layover points the other way and a
height-dependent bias in the towards-sensor component is expected.

    python tools/gf3/check_registration.py --gf3 <gf3>/geo/20240406_sigma0.tif \
        --geocode-json <gf3>/geo/20240406_geocode.json \
        --s1-desc <s1 desc>/sar_desc_3ch_uint8.tif <s1 desc>/SAR_BANDS_desc.json \
        --s1-asc <s1 asc>/sar_3ch_uint8.tif <s1 asc>/sar_3ch_scales.json
"""
import argparse
import json

import numpy as np
from osgeo import gdal
from scipy.ndimage import uniform_filter

gdal.UseExceptions()

S1 = None           # {"desc": (mosaic, json), "asc": (mosaic, json)}: --s1-desc, --s1-asc, set in main()
PIX_M = 1.657


def s1_db(path, meta_json, win):
    """VV mean dB (band index 2) of an S1 mosaic over a grid window."""
    j = json.load(open(meta_json, encoding="utf-8"))
    sc = j["encoding"]["scales"]["vv_db"]
    x0, y0, w, h = win
    ds = gdal.Open(path)
    dn = ds.GetRasterBand(3).ReadAsArray(x0, y0, w, h).astype(np.float32)
    v = sc["lo"] + (dn - 1) / 254.0 * (sc["hi"] - sc["lo"])
    return np.where(dn > 0, v, np.nan)


def down2(a):
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    return a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))


def ncc_shift(a, b, max_lag):
    """Peak of the normalised cross-correlation c(d) = sum a(x) b(x + d)."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    a = (a - a.mean()) / (a.std() + 1e-9) * win
    b = (b - b.mean()) / (b.std() + 1e-9) * win
    c = np.fft.ifft2(np.conj(np.fft.fft2(a)) * np.fft.fft2(b)).real / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9)
    c = np.fft.fftshift(c)
    cy, cx = np.array(c.shape) // 2
    sub = c[cy - max_lag:cy + max_lag + 1, cx - max_lag:cx + max_lag + 1]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    peak = sub[iy, ix]

    def parab(m, z, p):
        d = m - 2 * z + p
        return 0.0 if d == 0 else 0.5 * (m - p) / d

    fy = parab(sub[iy - 1, ix], peak, sub[iy + 1, ix]) if 0 < iy < sub.shape[0] - 1 else 0.0
    fx = parab(sub[iy, ix - 1], peak, sub[iy, ix + 1]) if 0 < ix < sub.shape[1] - 1 else 0.0
    # second-highest peak outside the main lobe: a sharpness / ambiguity check
    mask = np.ones_like(sub, bool)
    mask[max(0, iy - 5):iy + 6, max(0, ix - 5):ix + 6] = False
    second = sub[mask].max()
    return ix - max_lag + fx, iy - max_lag + fy, float(peak), float(second)


def main():
    global S1
    ap = argparse.ArgumentParser()
    ap.add_argument("--gf3", required=True)
    ap.add_argument("--geocode-json", required=True)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--max-lag-m", type=float, default=80.0)
    ap.add_argument("--min-valid", type=float, default=0.98)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--s1-desc", nargs=2, required=True, metavar=("MOSAIC", "JSON"),
                    help="S1 descending 3-channel uint8 mosaic on the grid (band 3: VV dB) and its json "
                         "(encoding.scales.vv_db)")
    ap.add_argument("--s1-asc", nargs=2, required=True, metavar=("MOSAIC", "JSON"),
                    help="the same for S1 ascending")
    a = ap.parse_args()
    S1 = {"desc": tuple(a.s1_desc), "asc": tuple(a.s1_asc)}

    gj = json.load(open(a.geocode_json, encoding="utf-8"))
    gw = gj["grid_window"]
    u_s = np.array(gj["geometry"]["unit_px_towards_sensor"])          # image px (x east, y south)
    u_a = np.array([-u_s[1], u_s[0]])                                  # along track, 90 deg to it
    # orient along-track with the flight direction (descending: heading ~190 deg = +y)
    if gj["pass"] == "DEC" and u_a[1] < 0:
        u_a = -u_a
    ds = gdal.Open(a.gf3)
    g = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    valid_g = np.isfinite(g)
    gm = uniform_filter(np.where(valid_g, g, 0), 3) / np.maximum(uniform_filter(valid_g.astype(np.float32), 3), 1e-6)
    gdb = np.where(valid_g, 10 * np.log10(np.maximum(gm, 1e-8)), np.nan)
    del g, gm
    H, W = gdb.shape
    win = (gw["ix0"], gw["iy0"], W, H)
    max_lag = int(round(a.max_lag_m / PIX_M / 2))
    res = {}
    for name, (tif, js) in S1.items():
        s = s1_db(tif, js, win)
        rows = []
        B = a.block
        for y in range(0, H - B + 1, B // 2):
            for x in range(0, W - B + 1, B // 2):
                gb, sb = gdb[y:y + B, x:x + B], s[y:y + B, x:x + B]
                ok = np.isfinite(gb) & np.isfinite(sb)
                if ok.mean() < a.min_valid:
                    continue
                gb = np.where(ok, gb, np.nanmedian(gb))
                sb = np.where(ok, sb, np.nanmedian(sb))
                dx, dy, peak, second = ncc_shift(down2(sb), down2(gb), max_lag)
                dx, dy = 2 * dx, 2 * dy                                 # back to grid px
                rows.append((x + gw["ix0"], y + gw["iy0"], dx, dy, peak, second))
        rows = np.array(rows)
        if rows.size == 0:
            print("%s: no block with both valid" % name)
            continue
        good = (rows[:, 4] > 0.15) & (rows[:, 4] > 1.3 * rows[:, 5])
        d = rows[good][:, 2:4]
        med = np.median(d, 0)
        mad = 1.4826 * np.median(np.abs(d - med), 0)
        along = d @ u_a * PIX_M
        across = d @ u_s * PIX_M
        res[name] = {"blocks": int(len(rows)), "blocks_used": int(good.sum()),
                     "median_dx_dy_px": med.tolist(), "robust_sd_px": mad.tolist(),
                     "median_along_track_m": float(np.median(along)),
                     "median_towards_sensor_m": float(np.median(across)),
                     "p25_p75_along_m": np.percentile(along, [25, 75]).tolist(),
                     "p25_p75_towards_sensor_m": np.percentile(across, [25, 75]).tolist(),
                     "median_peak_ncc": float(np.median(rows[good][:, 4])),
                     "rows": rows.round(3).tolist()}
        print("%-5s %3d/%3d blocks  offset GF3-S1 median (%+.2f, %+.2f) px = %.1f m  "
              "along-track %+.1f m [%+.1f, %+.1f]  towards sensor %+.1f m [%+.1f, %+.1f]  NCC %.2f"
              % (name, good.sum(), len(rows), med[0], med[1], np.hypot(*med) * PIX_M,
                 res[name]["median_along_track_m"], *res[name]["p25_p75_along_m"],
                 res[name]["median_towards_sensor_m"], *res[name]["p25_p75_towards_sensor_m"],
                 res[name]["median_peak_ncc"]))
    if a.out_json:
        with open(a.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
