# -*- coding: utf-8 -*-
"""Sentinel-1 InSAR features for WISH-Net: radar geometry -> AOI mosaic.

Two stacks, chosen by --track (TRACKS below):

  asc   S1A relative orbit 142 (ascending), VV, 8 dates 2023-06-06 ... 08-29, coregistered
        by ISCE2 topsStack onto 20230606 (--work: reference, coreg_secondarys,
        geom_reference; run_02/04/05, no ESD), swaths IW2 and IW3.
  desc  S1B relative orbit 47 (descending), VV, coregistered onto 20210615
        (its own --work), swaths IW1 and IW2; 7 of its 8 dates, 2021-06-15
        ... 09-07 (0814 was not acquired, so 0802_0826 is a 24-day pair; 0919 is a wet
        scene and is left out).  Its baselines are short: |K| about 1/3.4 of the
        ascending stack's, height of ambiguity >= 315 m.

resamp_withCarrier flattened the secondaries (its default) against the DEM of the topo
step, Copernicus GLO-30 in ellipsoidal heights (dem_isce for asc, beijing.dem for desc)
-- a DSM that still holds part of every building -- and topo put every pixel on that DSM
surface.

Per burst that touches the AOI, in radar geometry (window = AOI part + margin):

1. Interferogram of each sequential pair (i, j) (asc 7, desc 6), s_i conj(s_j),
   re-referenced from the DSM to the ground DEM the GF-3 product uses (beijing.pdtm,
   750 m local minimum, smoothed):  phi_ij = K_ij (h - h_DSM) + noise after ISCE's
   flattening, so it is multiplied by exp(j K_ij (h_DSM - h_pdtm)) to give
   K_ij (h - h_pdtm), the scatterer's height above the ground.
       K_ij = 4 pi (Bp_i - Bp_j) / (lambda R sin theta)
   Bp = (S_date - S_ref) . n, n perpendicular to the line of sight, pointing up; from
   the orbit state vectors at the window centre: each SAFE's annotation for asc, the
   POEORB precise orbits the stack was coregistered with for desc (the S1B 2021
   annotation orbits put Bp 1.4-2.5 m off, 16% on the 0615_0627 pair, against geo2rdr's
   range offsets; for the S1A 2023 stack annotation and POEORB agree to 2 cm).  (Sign:
   ISCE flattens s_ref conj(s_sec) with exp(-j 4 pi dR / lambda), i.e. the SLC phase is
   -4 pi R / lambda, and dR/dh = -Bp / (R sin theta); the second fact is checked here
   against geo2rdr's range offsets, the first by the building-height check afterwards.)
2. Complex coherence over a 5 (range) x 2 (azimuth) window (about 17 x 28 m):
       z = <s_i s_j*> / sqrt(<|s_i|^2> <|s_j|^2>)
3. Local high-pass: z <- z conj(LP) / |LP|, LP = Gaussian (sigma 400 m on the ground) of
   z, normalised by the valid mask.  Atmosphere, orbit residuals and Beijing's subsidence
   bowls vary over kilometres; a building's phase over tens of metres.
4. Mean coherence |z| of the pairs; calibrated sigma0 (VV, the annotation's sigmaNought
   LUT, no noise removal) averaged over the dates as power; amplitude dispersion
   std/mean of the calibrated single-look amplitude over the dates, then windowed.
5. Geolocation moved from the DSM surface to the ground DEM: on a lower surface a pixel's
   ground point lies (h_DSM - h_pdtm) cot(theta) closer to the sensor.  GF-3 is encoded
   on the same ground DEM, so in both products a building's returns are laid over toward
   its own sensor by h cot(theta).

Every burst is then warped (gdalwarp -geoloc, PIXEL_CENTER, bilinear) onto one AOI
lattice of 8 dataset pixels (4.32 EPSG:3857 units, 3.3 m on the ground) together with
its valid mask; each band is divided by the warped mask and the bursts are averaged
where they overlap.  sigma0 is carried as power and turned into dB only at the end.

Output (--out): s1insar_<track>_mosaic.tif (2 x pairs + 3 bands: 17 asc, 15 desc;
float32, NaN = no data) and s1insar_<track>_mosaic.json (channels, dates, baselines,
geometry, checks).  Tiles are cut by tile_s1insar.py.

In an ISCE2 environment (GDAL, SciPy; s1a_isce_utils from --topsstack, default
$ISCE_STACK/topsStack):

    python tools/s1insar/make_s1insar.py --work <stack> --zips <zips> [--old-zips <dir>/] \
        --pdtm <dem>/beijing.pdtm.wgs84.vrt --out <s1insar>
    python tools/s1insar/make_s1insar.py --track desc --work <desc stack> --zips <desc zips> \
        --orbits <POEORB> --pdtm <dem>/beijing.pdtm.wgs84.vrt --out <s1insar_desc>
    (--bursts IW3_03 --sign -1 for a test of one burst under the opposite sign;
     --dry-run prints the track's constants and the bursts on the AOI)
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime

import numpy as np
from osgeo import gdal
from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter

gdal.UseExceptions()

# the path fields (work, zips, old_zips, orbits) are set in main() from --work, --zips,
# --old-zips and --orbits; s1a_isce_utils (ISCE2 topsStack) is imported there as ut
TRACKS = {
    "asc": dict(
        work=None, zips=None,
        old_zips=None,                   # where the stack's VRTs still point (--old-zips)
        orbits=None,                     # SAFE annotation orbits
        mission="S1A", swaths=("IW2", "IW3"),
        dates=["20230606", "20230618", "20230630", "20230712",
               "20230724", "20230805", "20230817", "20230829"],
        product="Sentinel-1A ascending (relative orbit 142) IW VV InSAR features, 8 dates "
                "2023-06-06..08-29, 7 sequential 12-day pairs, on the BDH EPSG:3857 lattice",
        dem="dem_isce.dem.wgs84 (GLO-30 DSM)"),
    "desc": dict(
        work=None, zips=None,
        old_zips=None,                             # the VRTs point at the zips in place
        orbits=None,                               # POEORB, as in the coregistration (--orbits)
        mission="S1B", swaths=("IW1", "IW2"),
        dates=["20210615", "20210627", "20210709", "20210721",
               "20210802", "20210826", "20210907"],
        product="Sentinel-1B descending (relative orbit 47) IW VV InSAR features, 7 dates "
                "2021-06-15..09-07 (0814 not acquired, 0919 wet scene left out), 6 sequential "
                "pairs (12 days, 0802_0826 24 days), on the BDH EPSG:3857 lattice",
        dem="beijing.dem.wgs84 (GLO-30 DSM)")}
PDTM = None                              # ground DEM (tools/gf3/make_pseudo_dtm.py), --pdtm
AOI = (116.19, 39.75, 116.55, 40.03)     # W S E N, all splits plus a margin (geocode_bursts.py)
RGB_PIXEL_3857 = 0.53998256              # BDH dataset pixel, EPSG:3857 units
R_MERC = 6378137.0
WIN = (2, 5)                             # azimuth lines, range samples
SIGMA_M = 400.0
MARGIN_M = 1300.0                        # > 3 sigma of the high-pass
AZ_SPACING_M = 13.95                     # IW azimuth pixel spacing on the ground
WGS84_A, WGS84_E2 = 6378137.0, 6.69437999014e-3


def use_track(name):
    """Bind the stack constants (WORK, ZIPS, DATES, PAIRS, CHANNELS, ...) to one track.
    Channels: re, im of every sequential pair, then coh_mean, sigma0, dispersion; the
    burst rasters carry the valid mask as one more band (2 x pairs + 3)."""
    global TRACK, WORK, ZIPS, OLD_ZIPS, ORBITS, MISSION, SWATHS, DATES, REF, PAIRS, CHANNELS
    t = TRACKS[name]
    TRACK, WORK, ZIPS, OLD_ZIPS, ORBITS = name, t["work"], t["zips"], t["old_zips"], t["orbits"]
    MISSION, SWATHS, DATES = t["mission"], t["swaths"], t["dates"]
    REF = DATES[0]
    PAIRS = list(zip(DATES[:-1], DATES[1:]))
    CHANNELS = ["%s_%s_%s" % (c, i[4:], j[4:]) for i, j in PAIRS for c in ("re", "im")] + [
        "coh_mean", "sigma0_vv_db", "amp_dispersion_vv"]


use_track("asc")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ SAFE annotation
def zip_of(date):
    z = glob.glob(os.path.join(ZIPS, "%s_IW_SLC__1SDV_%sT*.zip" % (MISSION, date)))
    if len(z) != 1:
        raise FileNotFoundError("SAFE zip of %s: %s" % (date, z))
    return z[0]


def _xml(date, pattern):
    with zipfile.ZipFile(zip_of(date)) as z:
        name = [n for n in z.namelist() if re.search(pattern, n)]
        if len(name) != 1:
            raise FileNotFoundError("%s in %s: %s" % (pattern, date, name))
        return ET.fromstring(z.read(name[0]))


def _t(s):
    return datetime.strptime(s[:26], "%Y-%m-%dT%H:%M:%S.%f").timestamp()


_ORB, _LUT = {}, {}


def orbit(date, sw):
    k = (date, sw)
    if k not in _ORB:
        if ORBITS:
            _ORB[k] = poeorb(date)
            return _ORB[k]
        root = _xml(date, r"annotation/%s-%s-slc-vv-.*\.xml$" % (MISSION.lower(), sw.lower()))
        sv = root.findall(".//orbitList/orbit")
        t = np.array([_t(o.find("time").text) for o in sv])
        p = np.array([[float(o.find("position/" + c).text) for c in "xyz"] for o in sv])
        _ORB[k] = (t, p)
    return _ORB[k]


def poeorb(date):
    """State vectors (UTC, ECEF) of the POEORB file in ORBITS whose validity holds the date."""
    for f in sorted(glob.glob(os.path.join(ORBITS, "%s_OPER_AUX_POEORB_*.EOF" % MISSION))):
        v = re.search(r"_V(\d{8})T\d{6}_(\d{8})T", f)
        if v.group(1) < date < v.group(2):
            txt = open(f).read()
            t = np.array([_t(u) for u in re.findall(r"<UTC>UTC=([^<]+)</UTC>", txt)])
            p = np.array([[float(x) for x in re.findall(r'<%s unit="m">([^<]+)</%s>' % (c, c), txt)]
                          for c in "XYZ"]).T
            return t, p
    raise FileNotFoundError("POEORB of %s in %s" % (date, ORBITS))


def lut_sigma(date, sw, width):
    """sigmaNought LUT along range (pixel index of the swath tiff), median over the
    calibration vectors; it changes with range, not along the swath."""
    k = (date, sw)
    if k not in _LUT:
        root = _xml(date, r"annotation/calibration/calibration-%s-%s-slc-vv-.*\.xml$"
                    % (MISSION.lower(), sw.lower()))
        cols = np.arange(width, dtype=np.float64)
        rows = []
        for v in root.findall(".//calibrationVectorList/calibrationVector"):
            px = np.array(v.find("pixel").text.split(), np.float64)
            s0 = np.array(v.find("sigmaNought").text.split(), np.float64)
            rows.append(np.interp(cols, px, s0))
        _LUT[k] = np.median(np.array(rows), 0).astype(np.float32)
    return _LUT[k]


def ecef(lat, lon, h):
    la, lo = np.radians(lat), np.radians(lon)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(la) ** 2)
    return np.array([(n + h) * np.cos(la) * np.cos(lo), (n + h) * np.cos(la) * np.sin(lo),
                     (n * (1 - WGS84_E2) + h) * np.sin(la)])


def zero_doppler_pos(t, p, T):
    i = int(np.argmin(np.linalg.norm(p - T, axis=1)))
    sl = slice(max(0, i - 4), i + 5)
    t0 = t[sl].mean()
    cs = [np.polyfit(t[sl] - t0, p[sl, k], 5) for k in range(3)]
    pos = lambda s: np.array([np.polyval(c, s) for c in cs])
    vel = lambda s: np.array([np.polyval(np.polyder(c), s) for c in cs])
    s = t[i] - t0
    for _ in range(30):
        s -= ((pos(s) - T) @ vel(s)) / (vel(s) @ vel(s))
    return pos(s)


def bperp(sw, lat, lon, h):
    """Perpendicular baseline of every date to REF at one ground point (n up)."""
    T = ecef(lat, lon, h)
    P = {d: zero_doppler_pos(*orbit(d, sw), T) for d in DATES}
    los = (T - P[REF]) / np.linalg.norm(T - P[REF])
    up = T / np.linalg.norm(T)
    n = np.cross(np.cross(los, up), los)
    n /= np.linalg.norm(n)
    if n @ up < 0:
        n = -n
    return {d: float((P[d] - P[REF]) @ n) for d in DATES}, float(np.linalg.norm(T - P[REF]))


# ------------------------------------------------------------------ rasters
def open_vrt(path):
    xml = open(path).read().replace(OLD_ZIPS, ZIPS + "/")
    xml = xml.replace('relativeToVRT="1">burst', 'relativeToVRT="0">%s/burst' % os.path.dirname(path))
    return gdal.Open(xml)


def read_win(path, c0, r0, w, h, band=None):
    ds = open_vrt(path) if path.endswith(".slc.vrt") else gdal.Open(path)
    if band is None:
        return ds.ReadAsArray(c0, r0, w, h)
    return ds.GetRasterBand(band).ReadAsArray(c0, r0, w, h)


def pdtm_at(lat, lon):
    ds = gdal.Open(PDTM)
    gt = ds.GetGeoTransform()
    x = (lon - gt[0]) / gt[1] - 0.5
    y = (lat - gt[3]) / gt[5] - 0.5
    x0, x1 = int(np.floor(x.min())) - 2, int(np.ceil(x.max())) + 3
    y0, y1 = int(np.floor(y.min())) - 2, int(np.ceil(y.max())) + 3
    dem = ds.ReadAsArray(x0, y0, x1 - x0, y1 - y0).astype(np.float64)
    return map_coordinates(dem, [y - y0, x - x0], order=1, mode="nearest")


def boxcar(a):
    return uniform_filter(a, size=WIN, mode="nearest")


def lowpass(a, m, sig):
    num = gaussian_filter(a * m, sig, mode="nearest", truncate=3.0)
    den = gaussian_filter(m, sig, mode="nearest", truncate=3.0)
    return num / np.maximum(den, 1e-3)


def highpass(a, m):
    """Ramp-free first-order height structure: high-pass of hgt, for the checks."""
    return a - lowpass(a, m, (8, 30))


# ------------------------------------------------------------------ one burst
def burst_window(sw, n, b):
    g = os.path.join(WORK, "geom_reference", sw)
    lat = gdal.Open(os.path.join(g, "lat_%02d.rdr.vrt" % n)).ReadAsArray()
    lon = gdal.Open(os.path.join(g, "lon_%02d.rdr.vrt" % n)).ReadAsArray()
    v = np.zeros(lat.shape, bool)
    v[b.firstValidLine:b.firstValidLine + b.numValidLines,
      b.firstValidSample:b.firstValidSample + b.numValidSamples] = True
    ins = v & (lon > AOI[0]) & (lon < AOI[2]) & (lat > AOI[1]) & (lat < AOI[3])
    if not ins.any():
        return None
    rows, cols = np.where(ins.any(1))[0], np.where(ins.any(0))[0]
    inc = np.radians(40.0)
    mr = int(MARGIN_M / (b.rangePixelSize / np.sin(inc)))
    ma = int(MARGIN_M / AZ_SPACING_M)
    r0 = max(b.firstValidLine, rows.min() - ma)
    r1 = min(b.firstValidLine + b.numValidLines, rows.max() + 1 + ma)
    c0 = max(b.firstValidSample, cols.min() - mr)
    c1 = min(b.firstValidSample + b.numValidSamples, cols.max() + 1 + mr)
    return int(r0), int(r1), int(c0), int(c1), float(ins.mean())


def process_burst(sw, n, b, win, sign, tmp):
    r0, r1, c0, c1 = win[:4]
    L, W = r1 - r0, c1 - c0
    g = os.path.join(WORK, "geom_reference", sw)
    lat = read_win(os.path.join(g, "lat_%02d.rdr.vrt" % n), c0, r0, W, L)
    lon = read_win(os.path.join(g, "lon_%02d.rdr.vrt" % n), c0, r0, W, L)
    hgt = read_win(os.path.join(g, "hgt_%02d.rdr.vrt" % n), c0, r0, W, L).astype(np.float64)
    inc = read_win(os.path.join(g, "los_%02d.rdr.vrt" % n), c0, r0, W, L, band=1).astype(np.float64)
    azl = read_win(os.path.join(g, "los_%02d.rdr.vrt" % n), c0, r0, W, L, band=2).astype(np.float64)
    gnd = pdtm_at(lat, lon)
    dh = hgt - gnd                                    # DSM above the ground DEM
    rng = b.startingRange + (c0 + np.arange(W)) * b.rangePixelSize
    inv = 1.0 / (rng[None, :] * np.sin(np.radians(inc)))
    lam = b.radarWavelength
    cl, cc = L // 2, W // 2
    bp, R = bperp(sw, float(lat[cl, cc]), float(lon[cl, cc]), float(hgt[cl, cc]))
    gr_m = b.rangePixelSize / np.sin(np.radians(np.median(inc)))
    sig = (SIGMA_M / AZ_SPACING_M, SIGMA_M / gr_m)
    info = {"window": [int(r0), int(r1), int(c0), int(c1)], "aoi_fraction": win[4],
            "centre": [float(lat[cl, cc]), float(lon[cl, cc])], "slant_range_m": R,
            "bperp_m": bp, "incidence_deg": float(np.median(inc)),
            "los_azimuth_deg": float(np.median(azl)),
            "dsm_minus_ground_m": [float(np.percentile(dh, p)) for p in (5, 50, 95)],
            "dsm_highpass_std_m": float(np.std(highpass(hgt, np.ones_like(hgt))))}

    slc = {}
    for d in DATES:
        p = os.path.join(WORK, "reference" if d == REF else "coreg_secondarys/" + d, sw,
                         "burst_%02d.slc.vrt" % n)
        slc[d] = read_win(p, c0, r0, W, L).astype(np.complex64)
    valid = np.zeros((L, W), bool)
    valid[max(0, b.firstValidLine - r0):b.firstValidLine + b.numValidLines - r0,
          max(0, b.firstValidSample - c0):b.firstValidSample + b.numValidSamples - c0] = True
    for d in DATES:
        valid &= slc[d] != 0
    m = valid.astype(np.float32)

    # check of the baseline sign: geo2rdr's range offset grows with DEM height at
    # -Bp / (R sin theta) / dr pixels per metre
    hp_h = highpass(hgt, m)
    ratio = {}
    for d in DATES[1:]:
        off = read_win(os.path.join(WORK, "coreg_secondarys", d, sw, "range_%02d.off.vrt" % n),
                       c0, r0, W, L).astype(np.float64)
        hp_o = highpass(off, m)
        sel = valid & (np.abs(hp_h) > 3)
        slope = np.sum(hp_o[sel] * hp_h[sel]) / np.sum(hp_h[sel] ** 2)
        expect = np.median(-bp[d] * inv[sel] / b.rangePixelSize)
        ratio[d] = float(slope / expect) if abs(bp[d]) > 5 else None
    info["range_offset_slope_ratio"] = ratio

    out = np.zeros((len(CHANNELS) + 1, L, W), np.float32)
    coh = np.zeros((L, W), np.float32)
    pw = {d: boxcar(np.abs(slc[d]) ** 2) for d in DATES}           # windowed power
    gden = np.maximum(gaussian_filter(m, sig, mode="nearest", truncate=3.0), 1e-3)
    glp = lambda x: gaussian_filter(x * m, sig, mode="nearest", truncate=3.0) / gden
    for k, (di, dj) in enumerate(PAIRS):
        K = sign * 4 * np.pi * (bp[di] - bp[dj]) / lam * inv
        ifg = slc[di] * np.conj(slc[dj]) * np.exp(1j * K * dh).astype(np.complex64)
        num = boxcar(ifg.real) + 1j * boxcar(ifg.imag)
        den = np.sqrt(pw[di] * pw[dj])
        z = np.where(valid, num / np.maximum(den, 1e-12), 0).astype(np.complex64)
        lp = glp(z.real) + 1j * glp(z.imag)
        zt = z * np.conj(lp) / np.maximum(np.abs(lp), 1e-6)
        out[2 * k], out[2 * k + 1] = zt.real, zt.imag
        coh += np.abs(z)
        info.setdefault("pair_K_rad_per_m", {})["%s_%s" % (di, dj)] = float(np.median(K))
        info.setdefault("pair_coherence_median", {})["%s_%s" % (di, dj)] = float(np.median(np.abs(z)[valid]))
    c = 2 * len(PAIRS)                                # coh_mean, sigma0, dispersion, mask
    out[c] = coh / len(PAIRS)
    s0 = np.zeros((L, W), np.float64)
    amp = []
    for d in DATES:
        lut = lut_sigma(d, sw, b.numberOfSamples)[c0:c1][None, :]
        s0 += pw[d] / lut ** 2
        amp.append(np.abs(slc[d]) / lut)
    out[c + 1] = s0 / len(DATES)                      # linear power, dB in the mosaic
    amp = np.array(amp)
    disp = amp.std(0, ddof=1) / np.maximum(amp.mean(0), 1e-12)
    out[c + 2] = boxcar(disp)
    del amp, slc, pw
    out[:, ~valid] = 0
    out[c + 3] = m

    # geolocation moved from the DSM surface onto the ground DEM, toward the sensor
    d_m = dh / np.tan(np.radians(inc))
    a = np.radians(azl)
    phi = np.radians(lat)
    mlat = 111132.954 - 559.822 * np.cos(2 * phi) + 1.175 * np.cos(4 * phi)
    mlon = np.pi / 180 * WGS84_A * np.cos(phi) / np.sqrt(1 - WGS84_E2 * np.sin(phi) ** 2)
    lat2 = lat + np.cos(a) * d_m / mlat
    lon2 = lon - np.sin(a) * d_m / mlon

    tag = "%s_%02d" % (sw, n)
    drv = gdal.GetDriverByName("GTiff")
    co = ["TILED=YES", "BIGTIFF=YES"]
    rad = os.path.join(tmp, "radar_%s.tif" % tag)
    ds = drv.Create(rad, W, L, out.shape[0], gdal.GDT_Float32, co)
    for k in range(out.shape[0]):
        ds.GetRasterBand(k + 1).WriteArray(out[k])
    ds = None
    for name, arr in (("lat", lat2), ("lon", lon2)):
        ds = drv.Create(os.path.join(tmp, "%s_%s.tif" % (name, tag)), W, L, 1, gdal.GDT_Float64, co)
        ds.GetRasterBand(1).WriteArray(arr)
        ds = None
    info["valid_fraction"] = float(valid.mean())
    return rad, info, (float(lon2[valid].min()), float(lat2[valid].min()),
                       float(lon2[valid].max()), float(lat2[valid].max()))


GEOLOC = """<VRTDataset rasterXSize="{nx}" rasterYSize="{ny}">
  <Metadata domain="GEOLOCATION">
    <MDI key="SRS">EPSG:4326</MDI>
    <MDI key="X_DATASET">{lon}</MDI><MDI key="X_BAND">1</MDI>
    <MDI key="Y_DATASET">{lat}</MDI><MDI key="Y_BAND">1</MDI>
    <MDI key="PIXEL_OFFSET">0</MDI><MDI key="LINE_OFFSET">0</MDI>
    <MDI key="PIXEL_STEP">1</MDI><MDI key="LINE_STEP">1</MDI>
    <MDI key="GEOREFERENCING_CONVENTION">PIXEL_CENTER</MDI>
  </Metadata>
{bands}</VRTDataset>
"""
BAND = """  <VRTRasterBand dataType="Float32" band="{k}"><SimpleSource>
    <SourceFilename relativeToVRT="0">{src}</SourceFilename><SourceBand>{k}</SourceBand>
  </SimpleSource></VRTRasterBand>
"""


def merc(lon, lat):
    return np.radians(lon) * R_MERC, np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R_MERC


def warp(rad, tag, box, lattice, tmp):
    ox, oy, tr = lattice
    ds = gdal.Open(rad)
    nb, nx, ny = ds.RasterCount, ds.RasterXSize, ds.RasterYSize
    ds = None
    vrt = os.path.join(tmp, "geoloc_%s.vrt" % tag)
    with open(vrt, "w") as f:
        f.write(GEOLOC.format(nx=nx, ny=ny, lon=os.path.join(tmp, "lon_%s.tif" % tag),
                              lat=os.path.join(tmp, "lat_%s.tif" % tag),
                              bands="".join(BAND.format(k=k + 1, src=rad) for k in range(nb))))
    x0, y1 = merc(max(box[0], AOI[0]), min(box[3], AOI[3]))
    x1, y0 = merc(min(box[2], AOI[2]), max(box[1], AOI[1]))
    px, py = int(np.floor((x0 - ox) / tr)), int(np.floor((oy - y1) / tr))
    ex, ey = int(np.ceil((x1 - ox) / tr)), int(np.ceil((oy - y0) / tr))
    out = os.path.join(tmp, "geo_%s.tif" % tag)
    cmd = ["gdalwarp", "-overwrite", "-geoloc", "-t_srs", "EPSG:3857",
           "-te", ox + px * tr, oy - ey * tr, ox + ex * tr, oy - py * tr, "-tr", tr, tr,
           "-r", "bilinear", "-wo", "NUM_THREADS=8", "-wm", "2048",
           "-co", "TILED=YES", "-co", "BIGTIFF=YES", vrt, out]
    r = subprocess.run([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(r.stdout.decode("utf-8", "replace")[-2000:])
    return out, px, py


def mosaic(pieces, lattice, path):
    ox, oy, tr, W, H = lattice
    wsum = np.zeros((H, W), np.float32)
    for p, px, py in pieces:
        src = gdal.Open(p)                    # keep the dataset alive while its band is read
        v = src.GetRasterBand(len(CHANNELS) + 1).ReadAsArray()
        wsum[py:py + v.shape[0], px:px + v.shape[1]] += v[:H - py, :W - px]
    ok = wsum >= 0.5
    drv = gdal.GetDriverByName("GTiff")
    # band-interleaved: written band by band, a pixel-interleaved compressed file
    # rewrites every block once per band and grows to ~3x its size
    ds = drv.Create(path, W, H, len(CHANNELS), gdal.GDT_Float32,
                    ["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES", "BIGTIFF=YES", "INTERLEAVE=BAND"])
    ds.SetGeoTransform((ox, tr, 0.0, oy, 0.0, -tr))
    ds.SetProjection(gdal.Open(pieces[0][0]).GetProjection())
    stats = {}
    for k, name in enumerate(CHANNELS):
        acc = np.zeros((H, W), np.float32)
        for p, px, py in pieces:
            src = gdal.Open(p)
            v = src.GetRasterBand(k + 1).ReadAsArray()
            acc[py:py + v.shape[0], px:px + v.shape[1]] += v[:H - py, :W - px]
        acc = np.where(ok, acc / np.maximum(wsum, 1e-6), np.nan).astype(np.float32)
        if name == "sigma0_vv_db":
            acc = (10 * np.log10(np.maximum(acc, 1e-10))).astype(np.float32)
        band = ds.GetRasterBand(k + 1)
        band.WriteArray(acc)
        band.SetNoDataValue(float("nan"))
        band.SetDescription(name)
        v = acc[ok]
        stats[name] = [float(np.percentile(v, q)) for q in (1, 50, 99)]
        log("  band %2d %-18s p1 %8.3f  p50 %8.3f  p99 %8.3f" % (k, name, *stats[name]))
    ds = None
    return float(ok.mean()), stats


def main():
    global PDTM
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=sorted(TRACKS), default="asc")
    ap.add_argument("--work", required=True,
                    help="the track's topsStack folder: reference/, coreg_secondarys/, geom_reference/")
    ap.add_argument("--zips", required=True, help="the folder of the track's SAFE zips")
    ap.add_argument("--old-zips", default=None,
                    help="the zip folder the stack's VRTs name, replaced by --zips (default --zips/: in place)")
    ap.add_argument("--orbits", default=None,
                    help="POEORB folder for Bp (desc: required, as in its coregistration; asc: not given, "
                         "the SAFE annotation orbits)")
    ap.add_argument("--pdtm", required=True, help="the ground DEM, GDAL-readable (tools/gf3/make_pseudo_dtm.py)")
    ap.add_argument("--topsstack", default=os.path.join(os.environ["ISCE_STACK"], "topsStack")
                    if os.environ.get("ISCE_STACK") else None,
                    help="ISCE2 topsStack folder holding s1a_isce_utils.py (default $ISCE_STACK/topsStack)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bursts", default="", help="comma list like IW3_03 (default: all on the AOI)")
    ap.add_argument("--sign", type=float, default=1.0, help="sign of K (see the docstring)")
    ap.add_argument("--keep-tmp", action="store_true")
    ap.add_argument("--resume", action="store_true", help="reuse bursts already geocoded in <out>/_tmp")
    ap.add_argument("--dry-run", action="store_true", help="print the constants and the AOI bursts, then stop")
    a = ap.parse_args()
    if a.track == "desc" and not a.orbits:
        ap.error("--track desc needs --orbits (POEORB, as in its coregistration)")
    TRACKS[a.track].update(work=a.work, zips=a.zips, old_zips=a.old_zips or a.zips + "/", orbits=a.orbits)
    PDTM = a.pdtm
    if a.topsstack:
        sys.path.insert(0, a.topsstack)
    try:
        import s1a_isce_utils as ut
    except ImportError as e:
        ap.error("%s: pass --topsstack (ISCE2's topsStack folder) or set ISCE_STACK" % e)
    use_track(a.track)
    t0 = time.time()
    tmp = os.path.join(a.out, "_tmp")
    if a.dry_run:
        log("track %s: stack %s, zips %s/%s_*.zip, orbits %s, swaths %s, out %s"
            % (TRACK, WORK, ZIPS, MISSION, ORBITS or "SAFE annotation", ",".join(SWATHS), a.out))
        log("dates %s (reference %s)" % (" ".join(DATES), REF))
        log("pairs %s" % " ".join("%s_%s" % p for p in PAIRS))
        log("%d channels %s" % (len(CHANNELS), " ".join(CHANNELS)))
    else:
        os.makedirs(tmp, exist_ok=True)
    tr = 8 * RGB_PIXEL_3857
    x0, y1 = merc(AOI[0], AOI[3])
    x1, y0 = merc(AOI[2], AOI[1])
    ox, oy = np.floor(x0 / tr) * tr, np.ceil(y1 / tr) * tr
    W, H = int(np.ceil((x1 - ox) / tr)), int(np.ceil((oy - y0) / tr))
    log("AOI lattice %d x %d at %.5f EPSG:3857 units (%.2f m on the ground)"
        % (W, H, tr, tr * np.cos(np.radians(39.9))))
    want = set(x for x in a.bursts.split(",") if x)
    pieces, meta = [], {}
    for sw in SWATHS:
        ref = ut.loadProduct(os.path.join(WORK, "reference", sw + ".xml"))
        for i, b in enumerate(ref.bursts):
            n, tag = i + 1, "%s_%02d" % (sw, i + 1)
            if want and tag not in want:
                continue
            win = burst_window(sw, n, b)
            if win is None:
                continue
            if a.dry_run:
                log("%s window rows %d-%d cols %d-%d (AOI %.1f%% of the burst)"
                    % (tag, win[0], win[1], win[2], win[3], 100 * win[4]))
                continue
            cache = os.path.join(tmp, "info_%s.json" % tag)
            if a.resume and os.path.exists(cache):
                c = json.load(open(cache))
                pieces.append(tuple(c["piece"]))
                meta[tag] = c["info"]
                log("%s reused from %s" % (tag, tmp))
                continue
            log("%s window rows %d-%d cols %d-%d (AOI %.1f%% of the burst)"
                % (tag, win[0], win[1], win[2], win[3], 100 * win[4]))
            rad, info, box = process_burst(sw, n, b, win, a.sign, tmp)
            log("  incidence %.2f, LOS azimuth %.1f deg, DSM-ground p5/50/95 %s m, DSM high-pass std %.2f m"
                % (info["incidence_deg"], info["los_azimuth_deg"],
                   "/".join("%.1f" % v for v in info["dsm_minus_ground_m"]), info["dsm_highpass_std_m"]))
            log("  Bperp %s" % " ".join("%s:%+.1f" % (d[4:], v) for d, v in info["bperp_m"].items()))
            log("  range-offset slope / expected %s" % " ".join(
                "%s:%s" % (d[4:], "-" if v is None else "%.3f" % v)
                for d, v in info["range_offset_slope_ratio"].items()))
            log("  pair coherence median %s" % " ".join(
                "%.2f" % v for v in info["pair_coherence_median"].values()))
            geo = warp(rad, tag, box, (ox, oy, tr), tmp)
            pieces.append(geo)
            meta[tag] = info
            with open(cache, "w") as f:
                json.dump({"piece": list(geo), "info": info}, f)
            log("  geocoded at (%d, %d)" % (geo[1], geo[2]))
    if a.dry_run:
        return
    path = os.path.join(a.out, "s1insar_%s_mosaic.tif" % TRACK)
    cover, stats = mosaic(pieces, (ox, oy, tr, W, H), path)
    lam = 299792458.0 / 5.405e9
    hoa = {}
    for tag, info in meta.items():
        k = 4 * np.pi / (lam * info["slant_range_m"] * np.sin(np.radians(info["incidence_deg"])))
        hoa[tag] = {"%s_%s" % (i, j): float(2 * np.pi / max(abs(k * (info["bperp_m"][i] - info["bperp_m"][j])), 1e-9))
                    for i, j in PAIRS}
    doc = {
        "product": TRACKS[TRACK]["product"],
        "channels": CHANNELS,
        "channel_notes": {
            "re/im_<i>_<j>": "complex coherence of s_i conj(s_j) over 5x2 looks, flattened against the "
                             "ground DEM (pdtm), locally high-passed (Gaussian sigma 400 m)",
            "coh_mean": "mean |coherence| of the %d pairs" % len(PAIRS),
            "sigma0_vv_db": "calibrated sigma0 VV (sigmaNought LUT, no noise removal), power mean of %d dates, dB"
                            % len(DATES),
            "amp_dispersion_vv": "std/mean of the calibrated single-look amplitude over %d dates, 5x2 window"
                                 % len(DATES)},
        "dates": DATES, "pairs": ["%s_%s" % p for p in PAIRS], "reference": REF,
        "phase_model": "phi_ij = K_ij (h - h_ground), K_ij = sign * 4 pi (Bp_i - Bp_j) / (lambda R sin theta)",
        "sign": a.sign,
        "height_of_ambiguity_m": hoa,
        "geometry": {"towards_sensor": "horizontal LOS from target to platform; azimuth anticlockwise from "
                                        "north (ISCE los.rdr band 2), per burst in 'bursts'",
                     "geocoding_surface": "ground DEM beijing.pdtm (as GF-3)"},
        "grid": {"epsg": 3857, "geotransform": [ox, tr, 0.0, oy, 0.0, -tr], "width": W, "height": H,
                 "pixel_ground_m": tr * np.cos(np.radians(39.9))},
        "window_looks": {"azimuth": WIN[0], "range": WIN[1]}, "highpass_sigma_m": SIGMA_M,
        "coverage_of_aoi_lattice": cover, "band_percentiles_1_50_99": stats, "bursts": meta,
        "source": {"stack": WORK, "dem_of_flattening": TRACKS[TRACK]["dem"],
                   "ground_dem": PDTM},
        "elapsed_s": time.time() - t0}
    if ORBITS:                                        # asc: SAFE annotation orbits, no key
        doc["source"]["orbits"] = "POEORB in %s (Bp, K)" % ORBITS
    with open(os.path.join(a.out, "s1insar_%s_mosaic.json" % TRACK), "w") as f:
        json.dump(doc, f, indent=1)
    log("wrote %s (coverage %.2f%%), %.0f s" % (path, 100 * cover, time.time() - t0))
    if not a.keep_tmp:
        for f in glob.glob(os.path.join(tmp, "*")):
            os.remove(f)
        os.rmdir(tmp)


if __name__ == "__main__":
    main()
