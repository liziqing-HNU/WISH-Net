# -*- coding: utf-8 -*-
"""Gaofen-3 L1A SLC: metadata, orbit, range-Doppler geometry, RPC and calibration.

Shared by the GF-3 scripts in this folder.  Everything here is read from the
product itself (``*.meta.xml``, ``*.incidence.xml``, ``*.rpc``); nothing is
assumed about the mode beyond what the XML states.

Radar coordinates follow the L1A file: ``line`` is the image row (azimuth time
increases with it, ``eqvPRF`` lines per second from ``imagingTime/start``) and
``sample`` the column (slant range ``nearRange + sample * widthspace``).

Calibration (GF-3 user manual, L1A complex product):

    sigma0_dB = 10 log10[(I^2 + Q^2) * (QualifyValue / 32767)^2] - CalibrationConst
"""
import datetime as dt
import re
import xml.etree.ElementTree as ET

import numpy as np

C_LIGHT = 299792458.0
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def parse_time(s):
    return dt.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S.%f")


def _f(node, path):
    v = node.findtext(path)
    return None if v is None or v.strip() in ("", "NULL") else float(v)


class GF3Meta:
    """The fields of ``*.meta.xml`` that geometry and calibration need."""

    def __init__(self, xml_path, pol="HH"):
        r = ET.parse(xml_path).getroot()
        self.path = xml_path
        self.pol = pol
        self.satellite = r.findtext("satellite")
        self.orbit_id = int(r.findtext("orbitID"))
        self.direction = r.findtext("Direction")                     # DEC / ASC
        self.look = r.findtext("sensor/lookDirection")               # R / L
        self.mode = r.findtext("sensor/imagingMode")
        self.lam = _f(r, "sensor/lamda")
        self.look_angle_centre = _f(r, "sensor/waveParams/wave/centerLookAngle")
        self.mgc_db = _f(r, "sensor/waveParams/wave/valueMGC/%s" % pol)
        self.ground_velocity = _f(r, "sensor/waveParams/wave/groundVelocity")
        self.average_altitude = _f(r, "sensor/waveParams/wave/averageAltitude")
        self.range_bandwidth_mhz = _f(r, "sensor/waveParams/wave/bandWidth")
        self.az_bandwidth_hz = _f(r, "sensor/waveParams/wave/proBandwidth")
        self.t_start = parse_time(r.findtext("imageinfo/imagingTime/start"))
        self.t_end = parse_time(r.findtext("imageinfo/imagingTime/end"))
        self.near_range = _f(r, "imageinfo/nearRange")
        self.fs_mhz = _f(r, "imageinfo/eqvFs")
        self.prf = _f(r, "imageinfo/eqvPRF")
        self.width = int(r.findtext("imageinfo/width"))
        self.height = int(r.findtext("imageinfo/height"))
        self.dr = _f(r, "imageinfo/widthspace")                      # slant-range spacing, m
        self.daz = _f(r, "imageinfo/heightspace")                    # azimuth spacing, m
        self.qualify = _f(r, "imageinfo/QualifyValue/%s" % pol)
        self.echo_saturation = _f(r, "imageinfo/echoSaturation/%s" % pol)
        self.cal_const_db = _f(r, "processinfo/CalibrationConst/%s" % pol)
        self.ephemeris = r.findtext("processinfo/EphemerisData")
        self.inc_near = _f(r, "processinfo/incidenceAngleNearRange")
        self.inc_far = _f(r, "processinfo/incidenceAngleFarRange")
        self.fdc = [_f(r, "processinfo/DopplerCentroidCoefficients/d%d" % k) for k in range(5)]
        self.fdc_ref_us = _f(r, "processinfo/DopplerParametersReferenceTime")
        self.corners = {k: (_f(r, "imageinfo/corner/%s/latitude" % k),
                            _f(r, "imageinfo/corner/%s/longitude" % k))
                        for k in ("topLeft", "topRight", "bottomLeft", "bottomRight")}
        sv = r.findall("GPS/GPSParam")
        self.sv_t = np.array([(parse_time(p.findtext("TimeStamp")) - self.t_start).total_seconds()
                              for p in sv])
        self.sv_pos = np.array([[float(p.findtext(k)) for k in ("xPosition", "yPosition", "zPosition")]
                                for p in sv])
        self.sv_vel = np.array([[float(p.findtext(k)) for k in ("xVelocity", "yVelocity", "zVelocity")]
                                for p in sv])
        # sanity: the pixel spacing the XML states follows from fs and PRF
        assert abs(C_LIGHT / (2 * self.fs_mhz * 1e6) - self.dr) < 1e-3, (self.dr, self.fs_mhz)

    @property
    def duration(self):
        return (self.t_end - self.t_start).total_seconds()

    def line_time(self, line):
        """Azimuth time (s after imagingTime/start) of an image row."""
        return np.asarray(line, np.float64) / self.prf

    def sample_range(self, sample):
        return self.near_range + np.asarray(sample, np.float64) * self.dr

    def incidence_profile(self, inc_xml):
        """Per-column incidence angle (deg) from ``*.incidence.xml``."""
        txt = open(inc_xml, encoding="utf-8").read()
        v = np.array([float(x) for x in re.findall(r"<incidenceValue>([^<]+)</incidenceValue>", txt)])
        step = int(re.search(r"<stepSize>(\d+)</stepSize>", txt).group(1))
        cols = np.arange(v.size) * step
        return np.interp(np.arange(self.width), cols, v)

    def sigma0_gain(self):
        """Linear factor turning I^2+Q^2 into sigma0."""
        return (self.qualify / 32767.0) ** 2 / 10 ** (self.cal_const_db / 10.0)


class Orbit:
    """Polynomial fit of the state vectors around the scene.

    ``deg`` 7 over +-``pad`` s: an orbit arc of a minute is smooth to far below
    a centimetre at that order, so the fit residual measures the noise of the
    state vectors themselves (the XML gives positions to 0.1 m)."""

    def __init__(self, meta, pad=40.0, deg=7):
        t = meta.sv_t
        sel = (t >= -pad) & (t <= meta.duration + pad)
        self.t0 = 0.5 * meta.duration
        self.ts = 0.5 * meta.duration + pad
        tn = (t[sel] - self.t0) / self.ts
        self.cp = [np.polyfit(tn, meta.sv_pos[sel, k], deg) for k in range(3)]
        self.cv = [np.polyder(c) for c in self.cp]
        self.ca = [np.polyder(c, 2) for c in self.cp]
        fit_p = np.stack([np.polyval(c, tn) for c in self.cp], 1)
        fit_v = np.stack([np.polyval(c, tn) for c in self.cv], 1) / self.ts
        self.pos_rms = float(np.sqrt(((fit_p - meta.sv_pos[sel]) ** 2).sum(1).mean()))
        self.vel_rms = float(np.sqrt(((fit_v - meta.sv_vel[sel]) ** 2).sum(1).mean()))
        self.n_used = int(sel.sum())

    def pva(self, t):
        tn = (np.asarray(t, np.float64) - self.t0) / self.ts
        p = np.stack([np.polyval(c, tn) for c in self.cp], -1)
        v = np.stack([np.polyval(c, tn) for c in self.cv], -1) / self.ts
        a = np.stack([np.polyval(c, tn) for c in self.ca], -1) / self.ts ** 2
        return p, v, a


def llh2ecef(lat, lon, h):
    lat, lon = np.radians(lat), np.radians(lon)
    sl, cl = np.sin(lat), np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
    return np.stack([(n + h) * cl * np.cos(lon), (n + h) * cl * np.sin(lon),
                     (n * (1.0 - WGS84_E2) + h) * sl], -1)


def geo2rdr(meta, orbit, lat, lon, h, doppler_hz=0.0, iters=8):
    """Range-Doppler inverse: ground points -> (line, sample, look vector).

    ``doppler_hz`` 0 is the zero-Doppler geometry; a number (or array) solves
    the Doppler-centroid geometry instead, so the two conventions can be
    compared on the same points."""
    P = llh2ecef(lat, lon, h)
    t = np.full(P.shape[:-1], orbit.t0)
    half_lam = 0.5 * meta.lam
    for _ in range(iters):
        S, V, A = orbit.pva(t)
        d = P - S
        R = np.sqrt((d * d).sum(-1))
        vd = (V * d).sum(-1)
        f = vd - half_lam * doppler_hz * R
        fp = -(V * V).sum(-1) + (A * d).sum(-1) + half_lam * doppler_hz * vd / R
        t = t - f / fp
    S, V, _ = orbit.pva(t)
    d = P - S
    R = np.sqrt((d * d).sum(-1))
    line = t * meta.prf
    sample = (R - meta.near_range) / meta.dr
    return line, sample, d / R[..., None], S, V


def incidence_at(P, look):
    """Incidence angle (deg) on the ellipsoid normal at ECEF point(s) P."""
    x, y, z = P[..., 0], P[..., 1], P[..., 2]
    nrm = np.stack([x / WGS84_A ** 2, y / WGS84_A ** 2, z / (WGS84_A ** 2 * (1 - WGS84_E2))], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(-(look * nrm).sum(-1), -1, 1)))


class RPC:
    """The ``*.rpc`` file of a GF-3 product (DigitalGlobe RPB layout)."""

    KEYS = ("lineOffset", "sampOffset", "latOffset", "longOffset", "heightOffset",
            "lineScale", "sampScale", "latScale", "longScale", "heightScale")

    def __init__(self, path):
        txt = open(path, encoding="utf-8").read()
        for k in self.KEYS:
            setattr(self, k, float(re.search(r"\b%s\s*=\s*([-+0-9.eE]+)" % k, txt).group(1)))
        for k in ("lineNumCoef", "lineDenCoef", "sampNumCoef", "sampDenCoef"):
            body = re.search(r"%s\s*=\s*\(([^)]*)\)" % k, txt).group(1)
            setattr(self, k, np.array([float(x) for x in body.replace(",", " ").split()]))

    @staticmethod
    def _terms(P, L, H):
        # RPC00B term order
        return [np.ones_like(P), L, P, H, L * P, L * H, P * H, L * L, P * P, H * H,
                P * L * H, L ** 3, L * P * P, L * H * H, L * L * P, P ** 3, P * H * H,
                L * L * H, P * P * H, H ** 3]

    def project(self, lat, lon, h):
        P = (np.asarray(lat, np.float64) - self.latOffset) / self.latScale
        L = (np.asarray(lon, np.float64) - self.longOffset) / self.longScale
        H = (np.asarray(h, np.float64) - self.heightOffset) / self.heightScale
        T = self._terms(P, L, H)
        ev = lambda c: sum(ci * ti for ci, ti in zip(c, T))
        line = ev(self.lineNumCoef) / ev(self.lineDenCoef) * self.lineScale + self.lineOffset
        samp = ev(self.sampNumCoef) / ev(self.sampDenCoef) * self.sampScale + self.sampOffset
        return line, samp

    def gdal_metadata(self):
        """The same model as GDAL's RPC metadata domain (for gdalwarp -rpc)."""
        fmt = lambda a: " ".join("%.15e" % x for x in a)
        return {"LINE_OFF": repr(self.lineOffset), "SAMP_OFF": repr(self.sampOffset),
                "LAT_OFF": repr(self.latOffset), "LONG_OFF": repr(self.longOffset),
                "HEIGHT_OFF": repr(self.heightOffset), "LINE_SCALE": repr(self.lineScale),
                "SAMP_SCALE": repr(self.sampScale), "LAT_SCALE": repr(self.latScale),
                "LONG_SCALE": repr(self.longScale), "HEIGHT_SCALE": repr(self.heightScale),
                "LINE_NUM_COEFF": fmt(self.lineNumCoef), "LINE_DEN_COEFF": fmt(self.lineDenCoef),
                "SAMP_NUM_COEFF": fmt(self.sampNumCoef), "SAMP_DEN_COEFF": fmt(self.sampDenCoef)}
