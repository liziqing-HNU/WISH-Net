# -*- coding: utf-8 -*-
"""Is a SAR product registered to the BDH optical tiles on the BUILDINGS?

profile  (no GDAL)
    Stacked wall-base profile.  Every annotated footprint edge that faces the
    sensor (outward normal within 25 deg of the unit vector towards the sensor)
    is sampled along its normal from -40 to +40 m (+ = towards the sensor), in
    power, straight from a full-grid mosaic; each wall is normalised by its own
    median and the walls are averaged.  The wall-ground dihedral does not depend
    on the building's height, so for a registered product the stacked profile
    peaks at 0 m, with the facade layover as a shoulder on the + side.
    (The measure_rd.py intercept is not a registration measure on its own: it
    correlates FILLED footprints with the SAR, so the layover pulls it towards
    the sensor.  It reported ~0 m on the DSM-geocoded GF-3 product whose wall
    bases sit 11 m away from the sensor.)

profile-map  (no GDAL)
    The same peak per block of tiles (default 2 x 2, ~1.7 km), mapped: is the
    registration uniform over the city or biased locally (dense old town, CBD)?

compare  (no GDAL)
    Image pairs for chosen tiles (optical | each SAR version, footprints and the
    sensor-facing edges drawn, full tile and a 300 m zoom), the profile
    comparison figure, and viewer.html: one self-contained page that swipes or
    blends the optical tile with any SAR version.

export  (GDAL)
    For QGIS, per chosen tile: the optical GeoTIFF, every SAR version as a
    float32 dB GeoTIFF on the tile's own grid, and the footprints (GeoJSON, WGS84).

The SAR versions are full-grid float32 sigma0 mosaics (linear, NaN = nodata) on
the common SAR grid (EPSG:3857, 2.15993024 m), e.g. gf3_product/gf3_sigma0_mosaic.tif.

    python tools/gf3/registration_check_gf3.py profile --layers "GF-3 DSM=<tif>" "GF-3 DTM=<tif>" \
        --u 0.9836,0.1801 --root <BDH-RGB-SAR> --grid <grid json> --summary <tiles>/tile_summary_gf3.json \
        --product-json <gf3>/gf3_product/SAR_BANDS_gf3.json --out <gf3>/registration_check
    python tools/gf3/registration_check_gf3.py profile-map --layers ... --block 2      # peak per ~1.7 km block
    python tools/gf3/registration_check_gf3.py compare --layers ... --tiles val:112 train:431 ...
    python tools/gf3/registration_check_gf3.py export  --layers ... --tiles val:112 train:431 ...
    (each with the --root, --grid, --summary, --product-json and --out of the first)
"""
import argparse
import base64
import glob
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# set in main() from the command line
ROOT = None             # --root: BDH-RGB-SAR (annotations_fixed/, images/)
GRID_JSON = None        # --grid: json of the SAR grid (grid.geotransform_3857)
SUMMARY = None          # --summary: tile_summary_gf3.json
PRODUCT_JSON = None     # --product-json: SAR_BANDS_gf3.json
CACHE = None            # --cache: the mosaics as .npy, default <out>/_cache
GSD = 0.4142
T_M = np.arange(-40.0, 40.01, 0.5)


def grid():
    return json.load(open(GRID_JSON, encoding="utf-8"))["grid"]["geotransform_3857"]


def layers_arg(items):
    return [tuple(it.split("=", 1)) for it in items]


def as_npy(path):
    """A full-grid mosaic as a memory-mappable .npy (cached)."""
    if path.endswith(".npy"):
        return path
    os.makedirs(CACHE, exist_ok=True)
    npy = os.path.join(CACHE, os.path.basename(os.path.dirname(path)) + "_" +
                       os.path.splitext(os.path.basename(path))[0] + ".npy")
    if not os.path.exists(npy) or os.path.getmtime(npy) < os.path.getmtime(path):
        import tifffile
        np.save(npy, tifffile.imread(path).astype(np.float32))
    return npy


def buildings(anns):
    out = []
    for a in anns:
        if not a.get("segmentation"):
            continue
        roof = np.asarray(a["segmentation"][0], float).reshape(-1, 2)
        off = a.get("offset")
        foot = (np.asarray(off[0], float).reshape(-1, 2) if off and len(off[0]) == roof.size else roof)
        out.append((foot, roof, float(a.get("bd_height", 0) or 0)))
    return out


def edges_facing(poly, u, cos_min=0.3):
    """Edges whose outward normal (away from the centroid) has normal . u > cos_min."""
    c = poly.mean(0)
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        e = q - p
        L = np.hypot(*e)
        if L < 1e-6:
            continue
        nrm = np.array([e[1], -e[0]]) / L
        if np.dot((p + q) / 2 - c, nrm) < 0:
            nrm = -nrm
        if np.dot(nrm, u) > cos_min:
            out.append((p, q, nrm, L))
    return out


def tiles_by_split(scene=None):
    summary = json.load(open(SUMMARY, encoding="utf-8"))
    out = []
    for sp in ("train", "val", "test"):
        coco = json.load(open(os.path.join(ROOT, "annotations_fixed", "instances_%s.json" % sp)))
        by = {}
        for a in coco["annotations"]:
            by.setdefault(a["image_id"], []).append(a)
        for img in coco["images"]:
            if img["id"] not in by:
                continue
            fr = summary["tiles"]["%s/%s" % (sp, img["sar_file_name"])]
            sc = "20240401" if fr["20240401"] > 0.5 else "20240406"
            if scene and sc != scene:
                continue
            out.append((sp, img, by[img["id"]], sc))
    return out


# ---------------------------------------------------------------- profile

def _walls(task):
    from scipy.ndimage import map_coordinates
    M = np.load(task["npy"], mmap_mode="r")
    u = np.array(task["u"])
    t_px = T_M / GSD
    u0, v0 = task["uv0"]
    walls = []
    for foot, _, h in task["blds"]:
        if h < 3:
            continue
        for p, q, nrm, L in edges_facing(foot, u, math.cos(math.radians(25))):
            if L * GSD < 8:
                continue
            s_ = np.linspace(0.1, 0.9, max(3, int(L / 5)))
            pts = p[None] + s_[:, None] * (q - p)[None]
            walls.append((pts[:, None, :] + t_px[None, :, None] * nrm[None, None, :]).reshape(-1, 2))
    if not walls:
        return None
    xy = np.concatenate(walls)
    gx, gy = u0 + xy[:, 0] / 4.0 - 0.5, v0 + xy[:, 1] / 4.0 - 0.5      # tile px -> grid px centres
    x0, x1 = max(0, int(gx.min()) - 2), int(gx.max()) + 3
    y0, y1 = max(0, int(gy.min()) - 2), int(gy.max()) + 3
    sub = np.asarray(M[y0:y1, x0:x1], np.float32)
    v = map_coordinates(sub, [gy - y0, gx - x0], order=1, mode="nearest")
    acc, n, k = np.zeros(len(T_M)), 0, 0
    for w in walls:
        vv = v[k:k + len(w)].reshape(-1, len(T_M)).mean(0)
        k += len(w)
        if not np.isfinite(vv).all() or np.median(vv) <= 0:
            continue
        acc += vv / np.median(vv)
        n += 1
    return task["split"], acc, n, task.get("rc")


def wall_profile(npy, u, scene, workers=8):
    G = grid()
    tasks = [{"npy": npy, "u": list(u), "split": sp, "blds": buildings(anns),
              "uv0": ((img["geotransform_3857"][0] - G[0]) / G[1], (G[3] - img["geotransform_3857"][3]) / G[1])}
             for sp, img, anns, sc in tiles_by_split(scene)]
    acc = {s: [np.zeros(len(T_M)), 0] for s in ("train", "val", "test", "all")}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_walls, tasks, chunksize=4):
            if r:
                for key in (r[0], "all"):
                    acc[key][0] += r[1]
                    acc[key][1] += r[2]
    res = {}
    for key, (a, n) in acc.items():
        if n == 0:
            continue
        prof = 10 * np.log10(a / n)
        i = int(np.argmax(np.where(np.abs(T_M) <= 35, prof, -1e9)))
        y0, y1, y2 = prof[i - 1], prof[i], prof[i + 1]
        den = y0 - 2 * y1 + y2
        res[key] = {"walls": int(n), "peak_m": float(T_M[i] + (0.25 * (y0 - y2) / den if den else 0.0)),
                    "profile_db": prof.round(3).tolist()}
    return res


def profile(a):
    u = [float(v) for v in a.u.split(",")]
    os.makedirs(a.out, exist_ok=True)
    prev = os.path.join(a.out, "wall_profiles.json")
    store = json.load(open(prev)) if os.path.exists(prev) else {"layers": {}}
    store["t_m"] = T_M.tolist()
    for label, path in layers_arg(a.layers):
        r = wall_profile(as_npy(path), u, a.scene, a.workers)
        store["layers"][label] = r
        print("%-32s walls %6d  wall-base peak  all %+6.1f m   train %+6.1f  val %+6.1f  test %+6.1f"
              % (label, r["all"]["walls"], r["all"]["peak_m"], r.get("train", {}).get("peak_m", float("nan")),
                 r.get("val", {}).get("peak_m", float("nan")), r.get("test", {}).get("peak_m", float("nan"))),
              flush=True)
    json.dump(store, open(prev, "w"), indent=1)


def profile_map(a):
    """The wall-base peak per block of BLOCK x BLOCK tiles (~1.7 km for 2), per layer:
    is the registration uniform, or are there local biases (dense old town, tall CBD)?"""
    plt = _fonts()
    u = [float(v) for v in a.u.split(",")]
    G = grid()
    B = a.block
    tiles = tiles_by_split(a.scene)
    tasks_base = [{"u": u, "split": sp, "blds": buildings(anns),
                   "rc": (img["tile_row"] // B, img["tile_col"] // B, img["centre_lonlat"][0], img["centre_lonlat"][1]),
                   "uv0": ((img["geotransform_3857"][0] - G[0]) / G[1], (G[3] - img["geotransform_3857"][3]) / G[1])}
                  for sp, img, anns, sc in tiles]
    labs = layers_arg(a.layers)
    fig, axs = plt.subplots(1, len(labs) + 1, figsize=(18 / 2.54, 6.2 / 2.54),
                            gridspec_kw={"wspace": 0.25, "width_ratios": [1] * len(labs) + [0.9]})
    rows_out = []
    allpk = {}
    for k, (label, path) in enumerate(labs):
        npy = as_npy(path)
        blocks = {}
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for r in ex.map(_walls, [dict(t, npy=npy) for t in tasks_base], chunksize=4):
                if not r:
                    continue
                key = r[3][:2]
                b = blocks.setdefault(key, [np.zeros(len(T_M)), 0, [], []])
                b[0] += r[1]
                b[1] += r[2]
                b[2].append(r[3][2])
                b[3].append(r[3][3])
        pts = []
        for key, (acc, n, lons, lats) in blocks.items():
            if n < a.min_walls:
                continue
            prof = 10 * np.log10(acc / n)
            i = int(np.argmax(np.where(np.abs(T_M) <= 25, prof, -1e9)))
            y0, y1, y2 = prof[max(i - 1, 0)], prof[i], prof[min(i + 1, len(T_M) - 1)]
            den = y0 - 2 * y1 + y2
            pk = float(T_M[i] + (0.25 * (y0 - y2) / den if den else 0.0))
            pts.append((np.mean(lons), np.mean(lats), pk, n))
            rows_out.append((label, key[0], key[1], np.mean(lons), np.mean(lats), n, pk))
        pts = np.array(pts)
        allpk[label] = pts[:, 2]
        ax = axs[k]
        sc_ = ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap="RdBu", vmin=-15, vmax=15, s=14, marker="s",
                         edgecolors="none")
        ax.set_aspect(1 / math.cos(math.radians(39.9)))
        ax.set_title("%s\nmedian %+.1f m, IQR %.1f m" % (label, np.median(pts[:, 2]),
                                                         np.subtract(*np.percentile(pts[:, 2], [75, 25]))),
                     fontsize=7.5, pad=2)
        ax.tick_params(labelsize=6)
        print("%-28s blocks %3d  peak median %+.2f m  IQR [%+.2f, %+.2f]  |peak| <= 3 m: %.0f%%"
              % (label, len(pts), np.median(pts[:, 2]), *np.percentile(pts[:, 2], [25, 75]),
                 100 * (np.abs(pts[:, 2]) <= 3).mean()), flush=True)
    cb = fig.colorbar(sc_, ax=axs[:len(labs)], fraction=0.03, pad=0.02)
    cb.set_label("wall-base peak (m), + = towards the sensor", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    ax = axs[-1]
    bins = np.arange(-20, 20.1, 1.0)
    for label, v in allpk.items():
        ax.hist(v, bins, histtype="step", lw=1.0, label=label)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("wall-base peak per block (m)", fontsize=7)
    ax.set_ylabel("blocks", fontsize=7)
    ax.legend(fontsize=5.5, frameon=False)
    ax.tick_params(labelsize=6)
    os.makedirs(a.out, exist_ok=True)
    fig.savefig(os.path.join(a.out, "wall_peak_map.png"), bbox_inches="tight", pad_inches=0.02)
    with open(os.path.join(a.out, "wall_peak_map.csv"), "w") as f:
        f.write("layer,block_row,block_col,lon,lat,walls,peak_m\n")
        for r in rows_out:
            f.write("%s,%d,%d,%.5f,%.5f,%d,%.2f\n" % r)


# ---------------------------------------------------------------- compare

def _fonts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in glob.glob(os.path.expanduser("~/.local/share/fonts/*")):
        font_manager.fontManager.addfont(f)
    plt.rcParams.update({"font.family": ["Times New Roman", "SimSun"], "font.size": 8,
                         "mathtext.fontset": "stix", "savefig.dpi": 220, "axes.linewidth": 0.6})
    return plt


def tile_from_grid(arr, img):
    """Grid mosaic (linear power) -> the tile's own grid, bilinear with the
    pixel-centre term (make_tiles_gf3.py's transform); returns dB."""
    import cv2
    G = grid()
    gt = img["geotransform_3857"]
    s = gt[1] / G[1]
    u0, v0 = (gt[0] - G[0]) / G[1], (G[3] - gt[3]) / G[1]
    cx0, cy0 = int(np.floor(u0)) - 2, int(np.floor(v0)) - 2
    sub = np.asarray(arr[cy0:cy0 + int(img["height"] * s) + 6, cx0:cx0 + int(img["width"] * s) + 6], np.float32)
    off = 0.5 * s - 0.5
    M = np.array([[s, 0, u0 - cx0 + off], [0, s, v0 - cy0 + off]], np.float32)
    out = cv2.warpAffine(np.nan_to_num(sub, nan=1e-4), M, (img["width"], img["height"]),
                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    return 10 * np.log10(np.maximum(out, 10 ** -3.5))


def disp(db, smooth=0):
    import cv2
    x = db
    if smooth:
        x = 10 * np.log10(np.maximum(cv2.blur(np.power(10.0, db / 10.0), (smooth, smooth)), 1e-4))
    return np.clip((x + 25.0) / 40.0, 0, 1)


def compare(a):
    plt = _fonts()
    from matplotlib.patches import Polygon
    import cv2
    import tifffile
    os.makedirs(os.path.join(a.out, "pairs"), exist_ok=True)
    geo = {d: np.array(s["geometry"]["unit_px_towards_sensor"])
           for d, s in json.load(open(PRODUCT_JSON, encoding="utf-8"))["scenes"].items()}
    layers = [(lab, np.load(as_npy(p), mmap_mode="r")) for lab, p in layers_arg(a.layers)]
    want = [tuple(t.split(":")) for t in a.tiles]
    allt = {(sp, str(img["id"])): (sp, img, anns, sc) for sp, img, anns, sc in tiles_by_split()}
    viewer = []
    for key in want:
        sp, img, anns, sc = allt[key]
        u = geo[sc]
        blds = buildings(anns)
        rgb = tifffile.imread(os.path.join(ROOT, "images", sp, img["file_name"]))[..., :3]
        sars = [(lab, tile_from_grid(arr, img)) for lab, arr in layers]
        H, W = rgb.shape[:2]
        hs = np.array([b[2] for b in blds])
        c = np.mean([blds[k][0].mean(0) for k in np.argsort(hs)[::-1][:5]], 0)
        zs = 724                                            # 300 m
        zx, zy = int(np.clip(c[0] - zs / 2, 0, W - zs)), int(np.clip(c[1] - zs / 2, 0, H - zs))
        ncol = 1 + len(sars)
        fig, axs = plt.subplots(2, ncol, figsize=(18 / 2.54, 18 / 2.54 * 2 / ncol + 0.6),
                                gridspec_kw={"wspace": 0.02, "hspace": 0.06})
        for r, (x0, y0, sz) in enumerate(((0, 0, W), (zx, zy, zs))):
            for k in range(ncol):
                ax = axs[r, k]
                if k == 0:
                    ax.imshow(rgb, interpolation="lanczos")
                else:
                    ax.imshow(disp(sars[k - 1][1], smooth=3), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                lw = 0.3 if r == 0 else 0.8
                for foot, roof, h in blds:
                    ax.add_patch(Polygon(foot, closed=True, fill=False, ec="#00E5FF", lw=lw))
                    if k > 0:
                        for p, q, nrm, L in edges_facing(foot, u):
                            ax.plot([p[0], q[0]], [p[1], q[1]], color="#FFD600", lw=lw * 1.6)
                if r == 0 and k == 0:
                    ax.add_patch(Polygon([[zx, zy], [zx + zs, zy], [zx + zs, zy + zs], [zx, zy + zs]],
                                         closed=True, fill=False, ec="white", lw=0.8, ls="--"))
                if r == 1 and k > 0:
                    ax.annotate("", xy=(x0 + 0.1 * sz + u[0] * 0.15 * sz, y0 + 0.1 * sz + u[1] * 0.15 * sz),
                                xytext=(x0 + 0.1 * sz, y0 + 0.1 * sz),
                                arrowprops=dict(arrowstyle="-|>", color="#FF5252", lw=1.2))
                ax.set_xlim(x0, x0 + sz)
                ax.set_ylim(y0 + sz, y0)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title("Optical" if k == 0 else sars[k - 1][0], fontsize=7.5, pad=2)
        axs[0, 0].set_ylabel("tile 848 m", fontsize=7)
        axs[1, 0].set_ylabel("zoom 300 m", fontsize=7)
        fig.suptitle("%s %d  (%.4f E, %.4f N)  scene %s   cyan: footprint   yellow: edge facing the sensor   "
                     "red arrow: towards the sensor" % (sp, img["id"], *img["centre_lonlat"], sc), fontsize=7, y=0.995)
        fp = os.path.join(a.out, "pairs", "%s_%d.png" % (sp, img["id"]))
        fig.savefig(fp, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print("pair", fp, flush=True)

        def jpg(arr, size, is_rgb=False):
            im = arr if is_rgb else (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
            if im.ndim == 3:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            return "data:image/jpeg;base64," + base64.b64encode(
                cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()).decode()
        polys = [np.round(b[0], 1).tolist() for b in blds]
        edges = [[np.round(p, 1).tolist(), np.round(q, 1).tolist()]
                 for b in blds for p, q, _, _ in edges_facing(b[0], u)]
        for view, (x0, y0, sz), px in (("tile", (0, 0, W), 820), ("zoom", (zx, zy, zs), 724)):
            cr = (slice(y0, y0 + sz), slice(x0, x0 + sz))
            viewer.append({"label": "%s %d · %s" % (sp, img["id"], "whole tile, 848 m" if view == "tile" else "detail, 300 m"),
                           "scene": sc, "origin": [x0, y0], "size": sz, "u": u.tolist(),
                           "rgb": jpg(rgb[cr], px, True),
                           "sar": [{"name": lab, "raw": jpg(disp(db[cr]), px), "sm": jpg(disp(db[cr], 3), px)}
                                   for lab, db in sars],
                           "polys": polys, "edges": edges})
    write_viewer(os.path.join(a.out, "viewer.html"), viewer)
    print("viewer", os.path.join(a.out, "viewer.html"))
    prof_json = os.path.join(a.out, "wall_profiles.json")
    if os.path.exists(prof_json):
        P = json.load(open(prof_json))
        t = np.array(P["t_m"])
        fig, ax = plt.subplots(1, 1, figsize=(13 / 2.54, 7 / 2.54))
        for lab, r in P["layers"].items():
            if "all" in r:
                ax.plot(t, r["all"]["profile_db"], lw=1.0,
                        label="%s: peak %+.1f m (%d walls)" % (lab, r["all"]["peak_m"], r["all"]["walls"]))
        ax.axvline(0, color="#00B8D4", lw=0.8, ls="--")
        ax.set_xlim(-40, 40)
        ax.set_xlabel("Distance from the footprint edge along its normal (m), + = towards the sensor")
        ax.set_ylabel("σ0 / wall median (dB)")
        ax.set_title("Wall-base profile, walls facing the sensor (the dihedral should peak at 0)", fontsize=8, pad=2)
        ax.legend(fontsize=5.5, frameon=False, loc="upper left")
        fig.savefig(os.path.join(a.out, "wall_profiles.png"), bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


VIEWER = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GF-3 registration check</title>
<style>
:root{--bg:#111418;--panel:#1b2027;--ink:#e8eaed;--muted:#9aa4b2;--acc:#4fc3f7}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 "Segoe UI",Arial,sans-serif}
header{padding:10px 16px;border-bottom:1px solid #2a313b}
h1{font-size:16px;margin:0 0 4px}.sub{color:var(--muted);font-size:12px;max-width:1100px}
main{display:flex;gap:16px;padding:12px 16px;flex-wrap:wrap}
nav{min-width:200px;max-width:240px}
nav button{display:block;width:100%;text-align:left;margin:0 0 4px;padding:6px 8px;background:var(--panel);
 color:var(--ink);border:1px solid #2a313b;border-radius:6px;cursor:pointer;font-size:13px}
nav button.on{border-color:var(--acc);color:var(--acc)}
.stage{position:relative;width:min(78vh,calc(100vw - 32px));aspect-ratio:1/1;user-select:none;touch-action:none;
 background:#000;border:1px solid #2a313b}
.stage img,.stage svg{position:absolute;inset:0;width:100%;height:100%}
.clip{position:absolute;inset:0}
.handle{position:absolute;top:0;bottom:0;width:2px;background:#fff;box-shadow:0 0 4px #000;cursor:ew-resize}
.handle::after{content:"⇆";position:absolute;top:50%;left:-11px;width:24px;height:24px;margin-top:-12px;
 background:#fff;color:#111;border-radius:50%;text-align:center;line-height:24px;font-size:13px}
.ctl{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:8px 0;color:var(--muted);font-size:13px}
.ctl label{display:flex;gap:6px;align-items:center}
select{background:var(--panel);color:var(--ink);border:1px solid #2a313b;border-radius:4px;padding:2px 4px}
.info{color:var(--muted);font-size:12px;margin-top:4px}
.legend span{display:inline-block;width:14px;height:3px;margin:0 4px 3px 10px;vertical-align:middle}
</style></head><body>
<header><h1>GF-3 to optical registration check</h1>
<div class="sub">Drag the white line to switch between the optical image (left) and GF-3 (right), or blend the two. Cyan: the labelled building outlines; yellow: their edges facing the sensor (the red arrow points to the sensor).
When the registration is right, the bright wall-ground dihedral line lies on the yellow edges, the bright layover band extends from them towards the sensor and the shadow lies on the far side of the building; road and water boundaries coincide with the optical image.
"GF-3 version" switches between the product geocoded on the DSM and the one geocoded on the ground DEM (all other processing the same).</div></header>
<main><nav id="nav"></nav>
<div><div class="stage" id="stage"><img id="base"><div class="clip" id="clip"><img id="top"></div>
<svg id="ov" viewBox="0 0 1000 1000" preserveAspectRatio="none"></svg><div class="handle" id="handle"></div></div>
<div class="ctl">
<label>GF-3 version <select id="ver"></select></label>
<label><input type="checkbox" id="sm" checked>3x3 display smoothing</label>
<label><input type="checkbox" id="fp" checked>outlines</label>
<label><input type="checkbox" id="ed" checked>edges facing the sensor</label>
<label>mode <select id="mode"><option value="swipe">swipe</option><option value="blend">blend</option></select></label>
<label>GF-3 opacity <input type="range" id="alpha" min="0" max="100" value="60"></label>
</div>
<div class="legend"><span style="background:#00E5FF"></span>building outline<span style="background:#FFD600"></span>edge facing the sensor<span style="background:#FF5252"></span>towards the sensor</div>
<div class="info" id="info"></div></div></main>
<script>
const V = __DATA__;
let cur = 0, split = 0.5;
const $ = id => document.getElementById(id);
V[0].sar.forEach((s,i)=>{const o=document.createElement("option");o.value=i;o.textContent=s.name;$("ver").appendChild(o);});
function draw(){
  const v = V[cur], s = 1000 / v.size, o = v.origin, sv = v.sar[+$("ver").value];
  $("base").src = v.rgb; $("top").src = $("sm").checked ? sv.sm : sv.raw;
  if ($("mode").value === "swipe"){ $("clip").style.clipPath = `inset(0 0 0 ${split*100}%)`; $("clip").style.opacity = 1;
    $("handle").style.display = "block"; $("handle").style.left = `calc(${split*100}% - 1px)`; }
  else { $("clip").style.clipPath = "none"; $("clip").style.opacity = $("alpha").value/100; $("handle").style.display = "none"; }
  const X = p => ((p[0]-o[0])*s).toFixed(1), Y = p => ((p[1]-o[1])*s).toFixed(1);
  const w = v.size > 1500 ? 1.2 : 2.2;
  let svg = "";
  if ($("fp").checked) for (const poly of v.polys)
    svg += `<polygon points="${poly.map(p=>X(p)+","+Y(p)).join(" ")}" fill="none" stroke="#00E5FF" stroke-width="${w}" vector-effect="non-scaling-stroke"/>`;
  if ($("ed").checked) for (const e of v.edges)
    svg += `<line x1="${X(e[0])}" y1="${Y(e[0])}" x2="${X(e[1])}" y2="${Y(e[1])}" stroke="#FFD600" stroke-width="${w*1.4}" vector-effect="non-scaling-stroke"/>`;
  svg += `<line x1="60" y1="60" x2="${60+v.u[0]*90}" y2="${60+v.u[1]*90}" stroke="#FF5252" stroke-width="3" vector-effect="non-scaling-stroke"/><circle cx="${60+v.u[0]*90}" cy="${60+v.u[1]*90}" r="7" fill="#FF5252"/>`;
  $("ov").innerHTML = svg;
  $("info").textContent = `${v.label}, scene ${v.scene}`;
  document.querySelectorAll("nav button").forEach((b,i)=>b.classList.toggle("on", i===cur));
}
V.forEach((v,i)=>{const b=document.createElement("button");b.textContent=v.label;b.onclick=()=>{cur=i;draw();};$("nav").appendChild(b);});
["ver","sm","fp","ed","mode","alpha"].forEach(id=>$(id).addEventListener("input", draw));
let drag=false; const st=$("stage");
const move=e=>{ if(!drag) return; const r=st.getBoundingClientRect(); split=Math.min(1,Math.max(0,(e.clientX-r.left)/r.width)); draw(); };
st.addEventListener("pointerdown", e=>{ if($("mode").value!=="swipe") return; drag=true; st.setPointerCapture(e.pointerId); move(e); });
st.addEventListener("pointermove", move); st.addEventListener("pointerup", ()=>drag=false);
draw();
</script></body></html>
"""


def write_viewer(path, items):
    with open(path, "w", encoding="utf-8") as f:
        f.write(VIEWER.replace("__DATA__", json.dumps(items)))


# ---------------------------------------------------------------- export

def export(a):
    import shutil
    from osgeo import gdal, osr
    gdal.UseExceptions()
    d = os.path.join(a.out, "qgis")
    os.makedirs(d, exist_ok=True)
    R = 6378137.0
    want = {tuple(t.split(":")) for t in a.tiles}
    for sp, img, anns, sc in tiles_by_split():
        if (sp, str(img["id"])) not in want:
            continue
        stem = "%s_%d" % (sp, img["id"])
        shutil.copyfile(os.path.join(ROOT, "images", sp, img["file_name"]), os.path.join(d, stem + "_optical.tif"))
        gt = img["geotransform_3857"]
        bounds = (gt[0], gt[3] + img["height"] * gt[5], gt[0] + img["width"] * gt[1], gt[3])
        for label, path in layers_arg(a.layers):
            tag = label.split()[-1]
            tmp = gdal.Warp("", path, format="MEM", outputBounds=bounds, width=img["width"],
                            height=img["height"], resampleAlg="bilinear", dstSRS="EPSG:3857")
            p = tmp.GetRasterBand(1).ReadAsArray()
            tmp = None
            db = np.where(np.isfinite(p), 10 * np.log10(np.maximum(p, 10 ** -3.5)), np.nan).astype(np.float32)
            o = gdal.GetDriverByName("GTiff").Create(os.path.join(d, "%s_gf3_%s_db.tif" % (stem, tag)),
                                                     img["width"], img["height"], 1, gdal.GDT_Float32,
                                                     options=["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES"])
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(3857)
            o.SetProjection(srs.ExportToWkt())
            o.SetGeoTransform(tuple(gt))
            b = o.GetRasterBand(1)
            b.WriteArray(db)
            b.SetNoDataValue(float("nan"))
            b.SetDescription("%s, sigma0 HH (dB)" % label)
            o = None
        feats = []
        for foot, roof, h in buildings(anns):
            ring = [[math.degrees((gt[0] + x * gt[1]) / R),
                     math.degrees(2 * math.atan(math.exp((gt[3] + y * gt[5]) / R)) - math.pi / 2)]
                    for x, y in list(foot) + [foot[0]]]
            feats.append({"type": "Feature", "properties": {"height_m": h},
                          "geometry": {"type": "Polygon", "coordinates": [ring]}})
        json.dump({"type": "FeatureCollection", "features": feats},
                  open(os.path.join(d, stem + "_footprints.geojson"), "w"))
        print("exported", stem, flush=True)


def main():
    global ROOT, GRID_JSON, SUMMARY, PRODUCT_JSON, CACHE
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("profile", "profile-map", "compare", "export"))
    ap.add_argument("--layers", nargs="+", required=True, help='"label=path" full-grid sigma0 mosaics')
    ap.add_argument("--u", default="0.9836,0.1801", help="profile: unit vector towards the sensor (image px)")
    ap.add_argument("--scene", default="20240406", help="profile: tiles whose majority scene is this ('' = all)")
    ap.add_argument("--tiles", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--block", type=int, default=2, help="profile-map: tiles per block side")
    ap.add_argument("--min-walls", type=int, default=60, help="profile-map: fewer walls -> block skipped")
    ap.add_argument("--root", required=True, help="BDH-RGB-SAR dataset root (annotations_fixed/, images/)")
    ap.add_argument("--grid", required=True, help="json of the SAR grid (grid.geotransform_3857)")
    ap.add_argument("--summary", required=True, help="tile_summary_gf3.json (tile_summary_gf3.py)")
    ap.add_argument("--product-json", required=True, help="SAR_BANDS_gf3.json (mosaic_gf3.py)")
    ap.add_argument("--cache", default=None, help="the mosaics as .npy (default <out>/_cache)")
    a = ap.parse_args()
    ROOT, GRID_JSON, SUMMARY, PRODUCT_JSON = a.root, a.grid, a.summary, a.product_json
    CACHE = a.cache or os.path.join(a.out, "_cache")
    a.scene = a.scene or None
    {"profile": profile, "profile-map": profile_map, "compare": compare, "export": export}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
