# -*- coding: utf-8 -*-
"""One GF-3 L1A SLC scene -> calibrated sigma0 on the BDH SAR grid.

The grid is the Sentinel-1 products' grid (EPSG:3857, pixel 2.15993024 map
units = 1.657 m on the ground, origin and size from the --grid json, the S1
products' sar_3ch_scales.json), so a pixel index is the same ground point in the
GF-3, S1-ascending and S1-descending mosaics, and the existing tile cutter works
unchanged.

Processing, all from the product's own metadata:

  * sigma0 = (I^2 + Q^2) * (QualifyValue / 32767)^2 / 10^(CalibrationConst / 10),
    single look, in radar geometry;
  * backward range-Doppler geocoding on the --dem height model (ellipsoidal
    heights: Copernicus GLO-30, the file the S1 products used, or the ground
    model make_pseudo_dtm.py derives from it): every grid pixel
    centre (lon, lat, h_DEM) is solved for azimuth time and slant range
    against a polynomial fit of the state vectors, on nodes every STEP grid
    pixels, and the radar coordinates are interpolated bilinearly in between;
  * the intensity is sampled bilinearly.  The SLCs have no fill: every zero
    sample is a dark pixel quantised to I = Q = 0 (isolated, at most 17
    connected pixels; 0.16 % of the 04-01 scene, 1.36 % of the 04-06 scene,
    whose QualifyValue 7227 makes one DN step -31.2 dB), so zeros are data and
    only the image extent limits validity.  Nodata in the output is NaN.

Azimuth convention (``--doppler``): the product's RPC puts every point about
23-25 lines (40 m) later than the zero-Doppler solution, and within about one
line of the Doppler-centroid solution with the opposite sign of the XML's
DopplerCentroidCoefficients.  ``rpc`` (default) uses that convention,
``zero`` the zero-Doppler one; which of the two the image follows is measured
against the S1 products afterwards, not assumed.

Outputs in --out, named by acquisition date:
  <date>_sigma0.tif     float32 linear sigma0 on the grid window covering the
                        scene, NaN = nodata (0 is a valid, quantised value)
  <date>_geometry.tif   coarse (1 px per 32 grid px): incidence on the
                        ellipsoid, unit vector towards the sensor in image
                        pixels (x right, y down), line, sample
  <date>_geocode.json   parameters, checks, radar-domain statistics

Needs GDAL and SciPy:

    python tools/gf3/geocode_gf3.py --scene <raw>/GF3_..._L10006917544 --grid <grid json> \
        --dem <dem>/beijing.pdtm.wgs84.vrt --out <gf3>/geo
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from osgeo import gdal, osr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gf3_common import C_LIGHT, GF3Meta, Orbit, RPC, geo2rdr, incidence_at, llh2ecef  # noqa: E402

gdal.UseExceptions()

R_MERC = 6378137.0
STEP = 4            # geo2rdr on nodes every STEP grid pixels
COARSE = 32         # geometry raster
DATASET_GSD_M = 0.4142


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def unmerc(x, y):
    return (np.degrees(x / R_MERC),
            np.degrees(2 * np.arctan(np.exp(y / R_MERC)) - np.pi / 2))


def merc(lon, lat):
    return (R_MERC * np.radians(lon),
            R_MERC * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)))


class DEM:
    def __init__(self, path):
        ds = gdal.Open(path)
        self.gt = ds.GetGeoTransform()
        self.z = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)

    def __call__(self, lon, lat):
        """Bilinear, pixel-is-area geotransform (sample centres at +0.5)."""
        c = (lon - self.gt[0]) / self.gt[1] - 0.5
        r = (lat - self.gt[3]) / self.gt[5] - 0.5
        c0 = np.clip(np.floor(c).astype(np.int64), 0, self.z.shape[1] - 2)
        r0 = np.clip(np.floor(r).astype(np.int64), 0, self.z.shape[0] - 2)
        fc, fr = c - c0, r - r0
        z = self.z
        return ((1 - fr) * ((1 - fc) * z[r0, c0] + fc * z[r0, c0 + 1]) +
                fr * ((1 - fc) * z[r0 + 1, c0] + fc * z[r0 + 1, c0 + 1]))


def doppler_of(meta, R, mode):
    if mode == "zero":
        return 0.0
    tau = R / C_LIGHT * 1e6 - meta.fdc_ref_us          # one-way time, us
    f = sum(c * tau ** k for k, c in enumerate(meta.fdc))
    return -f


def read_sigma0(meta, tif, block=1024):
    """Single-look sigma0 in radar geometry, plus radar-domain statistics."""
    ds = gdal.Open(tif)
    bi, bq = ds.GetRasterBand(1), ds.GetRasterBand(2)
    H, W = meta.height, meta.width
    out = np.empty((H, W), np.float32)
    gain = np.float32(meta.sigma0_gain())
    n_zero = n_clip = 0
    amp_max = 0
    for r0 in range(0, H, block):
        n = min(block, H - r0)
        i = bi.ReadAsArray(0, r0, W, n).astype(np.float32)
        q = bq.ReadAsArray(0, r0, W, n).astype(np.float32)
        p = i * i + q * q
        n_zero += int((p == 0).sum())
        n_clip += int(((np.abs(i) >= 32767) | (np.abs(q) >= 32767)).sum())
        amp_max = max(amp_max, int(np.abs(i).max()), int(np.abs(q).max()))
        out[r0:r0 + n] = p * gain
    return out, {"quantised_zero_fraction": n_zero / float(H * W),
                 "int16_clipped_fraction": n_clip / float(H * W), "max_abs_iq": amp_max,
                 "one_dn_step_db": float(10 * np.log10(meta.sigma0_gain()))}


def radar_stats(s0, meta):
    """Robust statistics of the single-look scene in radar geometry.  Zeros
    (quantised dark pixels) enter the dB percentiles at half a DN step."""
    v = s0[::7, ::7]
    db = 10 * np.log10(np.maximum(v, 0.5 * meta.sigma0_gain()))
    pct = np.percentile(db, [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9])
    # dark floor: the darkest 7x7-averaged cells approach the noise floor
    from scipy.ndimage import uniform_filter
    sub = s0[::2, ::2]
    m = uniform_filter(sub, 7)[::7, ::7]
    m = m[m > 0]
    floor = 10 * np.log10(np.percentile(m, [0.05, 0.2, 1.0]))
    # speckle: the ratio std/mean of the single-look intensity over the whole
    # scene is ~1 for fully developed speckle on homogeneous cells; report the
    # median over 15x15 cells
    s1 = uniform_filter(sub, 15)
    s2 = uniform_filter(sub * sub, 15)
    cv = np.sqrt(np.maximum(s2 - s1 * s1, 0))[::15, ::15] / np.maximum(s1[::15, ::15], 1e-12)
    # range/azimuth mean profiles, to see stripes / antenna pattern residuals
    col_mean = 10 * np.log10(np.maximum(s0[::16].mean(0), 1e-12))
    row_mean = 10 * np.log10(np.maximum(s0[:, ::16].mean(1), 1e-12))
    # dark floor across the swath: per 512-column range bin, the 0.5th
    # percentile of the 7x7 means (water, radar shadow, smooth ground): an
    # upper bound of the noise-equivalent sigma0 and its elevation pattern
    mf = uniform_filter(sub, 7)[::7, ::7]
    per = 256 // 7                                     # columns of mf per 512 radar columns
    floor_rg = [float(10 * np.log10(max(np.percentile(mf[:, k:k + per], 0.5), 1e-12)))
                for k in range(0, mf.shape[1] - per + 1, per)]
    return {"sigma0_db_percentiles": dict(zip(["p0.1", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99.9"],
                                              map(float, pct))),
            "dark_floor_db_7x7_p0.05_p0.2_p1": list(map(float, floor)),
            "dark_floor_db_per_512_columns": floor_rg,
            "cv_15x15_median": float(np.median(cv)),
            "cv_15x15_p10": float(np.percentile(cv, 10))}, col_mean, row_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="product path without extension")
    ap.add_argument("--grid", required=True,
                    help="json of the SAR grid: grid.width, grid.height, grid.geotransform_3857")
    ap.add_argument("--dem", required=True, help="height model, ellipsoidal, GDAL-readable (e.g. a .vrt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--doppler", choices=("rpc", "zero"), default="rpc")
    ap.add_argument("--shift-az-m", type=float, default=0.0,
                    help="constant along-track correction (m, + = later lines)")
    ap.add_argument("--shift-rg-m", type=float, default=0.0,
                    help="constant slant-range correction (m, + = farther)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    meta = GF3Meta(a.scene + ".meta.xml")
    orbit = Orbit(meta)
    rpc = RPC(a.scene + ".rpc")
    date = meta.t_start.strftime("%Y%m%d")
    log("%s %s %s look %s, %d x %d, inc %.2f-%.2f deg, orbit fit %d vectors rms %.3f m"
        % (date, meta.mode, meta.direction, meta.look, meta.width, meta.height,
           meta.inc_near, meta.inc_far, orbit.n_used, orbit.pos_rms))

    g = json.load(open(a.grid, encoding="utf-8"))["grid"]
    gt = g["geotransform_3857"]
    x0, px, y1 = gt[0], gt[1], gt[3]
    nx, ny = int(g["width"]), int(g["height"])

    # grid window around the scene footprint (metadata corners + margin)
    cl = np.array([v for v in meta.corners.values()])
    cx, cy = merc(cl[:, 1], cl[:, 0])
    ix0 = max(0, int(np.floor((cx.min() - x0) / px)) - 64)
    ix1 = min(nx, int(np.ceil((cx.max() - x0) / px)) + 64)
    iy0 = max(0, int(np.floor((y1 - cy.max()) / px)) - 64)
    iy1 = min(ny, int(np.ceil((y1 - cy.min()) / px)) + 64)
    wx, wy = ix1 - ix0, iy1 - iy0
    log("grid window x %d..%d y %d..%d (%d x %d)" % (ix0, ix1, iy0, iy1, wx, wy))

    dem = DEM(a.dem)

    # --- geo2rdr on nodes
    t0 = time.time()
    nxn, nyn = wx // STEP + 2, wy // STEP + 2
    jn = np.arange(nxn) * STEP
    in_ = np.arange(nyn) * STEP
    X = x0 + (ix0 + jn + 0.5) * px
    Y = y1 - (iy0 + in_ + 0.5) * px
    lon_n, _ = unmerc(X, np.zeros_like(X))
    _, lat_n = unmerc(np.zeros_like(Y), Y)
    LON, LAT = np.meshgrid(lon_n, lat_n)
    HGT = dem(LON, LAT)
    line_n = np.empty(LON.shape)
    samp_n = np.empty(LON.shape)
    ux_n = np.empty(LON.shape, np.float32)
    uy_n = np.empty(LON.shape, np.float32)
    inc_n = np.empty(LON.shape, np.float32)
    for r in range(0, nyn, 64):
        sl = slice(r, min(nyn, r + 64))
        la, lo, hh = LAT[sl], LON[sl], HGT[sl]
        # zero-Doppler first gives the range for the Doppler polynomial
        l, s, look, S, V = geo2rdr(meta, orbit, la, lo, hh, 0.0)
        if a.doppler != "zero":
            fd = doppler_of(meta, meta.sample_range(s), a.doppler)
            l, s, look, S, V = geo2rdr(meta, orbit, la, lo, hh, fd)
        line_n[sl] = l + a.shift_az_m / meta.daz
        samp_n[sl] = s + a.shift_rg_m / meta.dr
        P = llh2ecef(la, lo, hh)
        inc_n[sl] = incidence_at(P, look)
        # towards the sensor, horizontal, in image pixels (x east, y south)
        latr, lonr = np.radians(la), np.radians(lo)
        e = -look[..., 0] * -np.sin(lonr) + -look[..., 1] * np.cos(lonr)
        n = (-look[..., 0] * -np.sin(latr) * np.cos(lonr) + -look[..., 1] * -np.sin(latr) * np.sin(lonr)
             + -look[..., 2] * np.cos(latr))
        nn = np.hypot(e, n)
        ux_n[sl], uy_n[sl] = e / nn, -n / nn
    log("geo2rdr on %d x %d nodes, %.0f s" % (nxn, nyn, time.time() - t0))

    # the RPC as an independent check of the same nodes (inside the image)
    inside_n = (line_n >= 0) & (line_n <= meta.height - 1) & (samp_n >= 0) & (samp_n <= meta.width - 1)
    rl, rs = rpc.project(LAT[inside_n][::50], LON[inside_n][::50], HGT[inside_n][::50])
    d_line = line_n[inside_n][::50] - rl
    d_samp = samp_n[inside_n][::50] - rs
    log("model - RPC over %d nodes: line %+.2f +- %.2f (%.1f m), sample %+.2f +- %.2f (%.1f m)"
        % (d_line.size, d_line.mean(), d_line.std(), d_line.mean() * meta.daz,
           d_samp.mean(), d_samp.std(), d_samp.mean() * meta.dr))

    # --- radar-geometry sigma0
    t0 = time.time()
    s0, io_stats = read_sigma0(meta, a.scene + ".tiff")
    log("sigma0 read, %.0f s; quantised zeros %.4f%%, int16-clipped %.5f%%, max |I|,|Q| %d, "
        "one DN step %.2f dB"
        % (time.time() - t0, 100 * io_stats["quantised_zero_fraction"],
           100 * io_stats["int16_clipped_fraction"], io_stats["max_abs_iq"], io_stats["one_dn_step_db"]))
    stats, col_mean, row_mean = radar_stats(s0, meta)
    log("single-look sigma0 dB percentiles %s" % {k: round(v, 2) for k, v in stats["sigma0_db_percentiles"].items()})
    log("dark floor (7x7 means, p0.05/p0.2/p1): %s dB; CV 15x15 median %.3f p10 %.3f"
        % (["%.2f" % v for v in stats["dark_floor_db_7x7_p0.05_p0.2_p1"]], stats["cv_15x15_median"],
           stats["cv_15x15_p10"]))
    # --- full-resolution sampling, in row chunks
    t0 = time.time()
    out = np.full((wy, wx), np.nan, np.float32)
    from scipy.ndimage import map_coordinates
    H, W = meta.height, meta.width
    jj = np.arange(wx) / STEP
    for r0 in range(0, wy, 256):
        r1 = min(wy, r0 + 256)
        ii = np.arange(r0, r1) / STEP
        I_, J_ = np.meshgrid(ii, jj, indexing="ij")
        L = map_coordinates(line_n, [I_, J_], order=1, mode="nearest")
        S = map_coordinates(samp_n, [I_, J_], order=1, mode="nearest")
        ok = (L >= 0) & (L <= H - 1) & (S >= 0) & (S <= W - 1)
        if not ok.any():
            continue
        l0 = np.clip(np.floor(L).astype(np.int64), 0, H - 2)
        c0 = np.clip(np.floor(S).astype(np.int64), 0, W - 2)
        fl, fc = (L - l0).astype(np.float32), (S - c0).astype(np.float32)
        v = ((1 - fl) * ((1 - fc) * s0[l0, c0] + fc * s0[l0, c0 + 1]) +
             fl * ((1 - fc) * s0[l0 + 1, c0] + fc * s0[l0 + 1, c0 + 1]))
        out[r0:r1] = np.where(ok, v, np.nan)
    del s0
    cover = np.isfinite(out)
    log("sampled %d x %d, %.0f s; valid %.2f%% of the window, %.2f%% of the full grid"
        % (wx, wy, time.time() - t0, 100 * cover.mean(), 100 * cover.sum() / float(nx * ny)))

    # --- write
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3857)
    drv = gdal.GetDriverByName("GTiff")
    f_s0 = os.path.join(a.out, "%s_sigma0.tif" % date)
    d = drv.Create(f_s0, wx, wy, 1, gdal.GDT_Float32,
                   options=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=3", "BIGTIFF=IF_SAFER"])
    d.SetProjection(srs.ExportToWkt())
    d.SetGeoTransform((x0 + ix0 * px, px, 0.0, y1 - iy0 * px, 0.0, -px))
    b = d.GetRasterBand(1)
    b.WriteArray(out)
    b.SetNoDataValue(float("nan"))
    b.SetDescription("sigma0_linear_single_look_%s" % meta.pol)
    d = None

    # coarse geometry raster: nodes every STEP px -> every COARSE px
    k = COARSE // STEP
    f_geo = os.path.join(a.out, "%s_geometry.tif" % date)
    gshape = (inc_n[::k, ::k].shape[1], inc_n[::k, ::k].shape[0])
    g_ = drv.Create(f_geo, gshape[0], gshape[1], 5, gdal.GDT_Float32, options=["COMPRESS=DEFLATE"])
    g_.SetProjection(srs.ExportToWkt())
    # node (i, j) sits at the centre of grid pixel (iy0 + i*STEP, ix0 + j*STEP)
    g_.SetGeoTransform((x0 + (ix0 + 0.5) * px - 0.5 * COARSE * px, COARSE * px, 0.0,
                        y1 - (iy0 + 0.5) * px + 0.5 * COARSE * px, 0.0, -COARSE * px))
    for bi_, (arr, name) in enumerate(((inc_n, "incidence_deg"), (ux_n, "u_sensor_x"),
                                       (uy_n, "u_sensor_y"), (line_n, "line"), (samp_n, "sample")), 1):
        band = g_.GetRasterBand(bi_)
        band.WriteArray(np.asarray(arr[::k, ::k], np.float32))
        band.SetDescription(name)
    g_ = None

    th = inc_n[inside_n]
    u = np.array([ux_n[inside_n].mean(), uy_n[inside_n].mean()])
    u /= np.linalg.norm(u)
    theta = float(th.mean())
    rec = {
        "scene": os.path.basename(a.scene), "date": date, "satellite": meta.satellite,
        "mode": meta.mode, "polarisation": meta.pol, "pass": meta.direction, "look": meta.look,
        "orbit_id": meta.orbit_id, "ephemeris": meta.ephemeris,
        "centre_time_utc": (meta.t_start + (meta.t_end - meta.t_start) / 2).isoformat(),
        "size": [meta.width, meta.height], "spacing_m": {"slant_range": meta.dr, "azimuth": meta.daz},
        "range_bandwidth_mhz": meta.range_bandwidth_mhz, "az_processed_bandwidth_hz": meta.az_bandwidth_hz,
        "calibration": {"qualify_value": meta.qualify, "calibration_const_db": meta.cal_const_db,
                        "gain_db": 10 * np.log10(meta.sigma0_gain())},
        "echo_saturation_pct": meta.echo_saturation, "mgc_db": meta.mgc_db,
        "doppler_convention": a.doppler, "shift_az_m": a.shift_az_m, "shift_rg_m": a.shift_rg_m,
        "orbit_fit": {"vectors": orbit.n_used, "pos_rms_m": orbit.pos_rms, "vel_rms_mps": orbit.vel_rms},
        "model_minus_rpc": {"line_mean": float(d_line.mean()), "line_std": float(d_line.std()),
                            "sample_mean": float(d_samp.mean()), "sample_std": float(d_samp.std())},
        "radar_io": io_stats, "radar_stats": stats,
        "grid_window": {"ix0": ix0, "iy0": iy0, "width": wx, "height": wy},
        "valid_fraction_of_grid": float(cover.sum() / float(nx * ny)),
        "geometry": {
            "incidence_deg": {"mean": theta, "min": float(th.min()), "max": float(th.max())},
            "unit_px_towards_sensor": [float(u[0]), float(u[1])],
            "towards_sensor_bearing_deg": float(np.degrees(np.arctan2(u[0], -u[1])) % 360),
            "ground_range_m_per_sample": float(meta.dr / np.sin(np.radians(theta))),
            "theory_at_dataset_pixel": {"gsd_m": DATASET_GSD_M,
                                        "k_rd_px_per_m": float(1 / np.tan(np.radians(theta)) / DATASET_GSD_M),
                                        "k_sh_px_per_m": float(np.tan(np.radians(theta)) / DATASET_GSD_M)}},
        "profiles": {"range_mean_db_every_64": col_mean[::64].round(3).tolist(),
                     "azimuth_mean_db_every_64": row_mean[::64].round(3).tolist(),
                     "incidence_deg_per_512_columns":
                         meta.incidence_profile(a.scene + ".incidence.xml")[256::512].round(3).tolist()},
    }
    with open(os.path.join(a.out, "%s_geocode.json" % date), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    log("incidence %.2f deg (%.2f-%.2f), towards sensor (%.3f, %.3f) bearing %.1f, k_rd %.2f k_sh %.2f px/m"
        % (theta, th.min(), th.max(), u[0], u[1], rec["geometry"]["towards_sensor_bearing_deg"],
           rec["geometry"]["theory_at_dataset_pixel"]["k_rd_px_per_m"],
           rec["geometry"]["theory_at_dataset_pixel"]["k_sh_px_per_m"]))
    log("wrote %s, %s" % (f_s0, f_geo))


if __name__ == "__main__":
    main()
