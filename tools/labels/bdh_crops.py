# -*- coding: utf-8 -*-
"""Clipping BDH annotations to a crop window, for the evaluation crops
(prepare_bdh.py) and their ignore regions (make_ignore_regions.py).

The functions are those the evaluation files of the paper were made with
(bdh/data/mapper.py of the MaskDINO-BDH code the experiments started from),
copied unchanged so that the files come out byte for byte the same (only
negative zeros are written as 0.0, see _clip_poly_list): besides
the clipped roof, ``with_base`` clips the footprint ring (segmentation_base)
and ``box_spans_footprint`` makes the box cover the footprint as well.
wishnet/data/mapper.py clips training windows with its own copy, which has
neither option.  numpy and shapely only.
"""
import copy

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry import box as shp_box


def offset_vector(ann):
    """The roof -> footprint translation of one annotation, in pixels.

    ``segmentation`` is the **roof** polygon and ``offset`` the **footprint**
    (``coordinates`` is the same ring, closed).  For 98.4% of the BDH
    annotations the footprint is an exact translation of the roof; for the
    remaining 1.6% the per-vertex deltas differ slightly (median spread 3.7 px)
    and the mean delta is the best single-vector summary.

    Returns ``[dx, dy]``, or ``[0.0, 0.0]`` when the annotation carries no
    footprint or the two rings disagree in length.
    """
    seg = ann.get("segmentation")
    off = ann.get("offset")
    if not seg or not off:
        return [0.0, 0.0]
    try:
        a = np.asarray(seg[0], dtype=np.float64).reshape(-1, 2)
        b = np.asarray(off[0], dtype=np.float64).reshape(-1, 2)
    except Exception:
        return [0.0, 0.0]
    if a.shape != b.shape or a.size == 0:
        return [0.0, 0.0]
    d = (b - a).mean(axis=0)
    return [float(d[0]), float(d[1])]


def _to_shapely(poly_flat):
    pts = np.asarray(poly_flat, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return None
    g = Polygon(pts)
    if not g.is_valid:
        g = g.buffer(0)
    if g.is_empty:
        return None
    return g


def _geom_to_coco_polys(geom):
    """shapely geometry -> COCO ``segmentation`` list, plus its area."""
    polys = []
    if isinstance(geom, Polygon):
        geoms = [geom]
    elif isinstance(geom, MultiPolygon):
        geoms = list(geom.geoms)
    else:  # GeometryCollection etc.
        geoms = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    area = 0.0
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
    """Clip a list of flat polygons to ``rect``, then shift into crop pixels."""
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
        arr[:, 0] -= x0
        arr[:, 1] -= y0
        np.clip(arr[:, 0], 0.0, size, out=arr[:, 0])
        np.clip(arr[:, 1], 0.0, size, out=arr[:, 1])
        # a vertex on the window's edge can come out as -0.0 with newer numpy /
        # shapely; + 0.0 makes it 0.0 (the same number), as the paper's files have it
        arr += 0.0
        shifted.append(arr.reshape(-1).tolist())
    return shifted, area, orig_area


# A building counts as TRUNCATED when the window cuts its roof (less than
# TRUNC_FRAC of the roof area inside), its roof touches the window edge (a
# tile-edge cut leaves no area difference to see) or its footprint box
# reaches beyond the window.  Truncated buildings stay detection targets and
# are matched, but their height and offset are neither supervised nor scored:
# the full height label with part of the roof, facade and parallax missing
# measures the crop, not the model.
TRUNC_FRAC = 0.995
TRUNC_EDGE_PX = 1.0


def clip_annotations(annos, x0, y0, size, min_area=32.0, min_frac=0.1,
                     with_base=False, box_spans_footprint=True,
                     trunc_frac=TRUNC_FRAC, trunc_edge_px=TRUNC_EDGE_PX):
    """Shift annotations into crop coordinates and clip them to the crop.

    Every kept annotation also carries ``clip_frac`` (share of the roof area
    inside the window) and ``truncated`` (see TRUNC_FRAC above).

    Returns a new list of annotation dicts with ``segmentation`` (the **roof**)
    / ``bbox`` / ``area`` expressed in crop pixels.  ``bd_height`` is carried
    over untouched -- a partly visible building is still exactly as tall -- and
    so is ``bd_offset``, which is a translation and therefore invariant to
    cropping.

    With ``with_base`` the annotated **footprint** polygon is clipped as well
    and stored as ``segmentation_base``.  It is set to ``None`` when the base
    falls outside the crop, which happens for roofs near the edge because the
    offset runs up to ~100 px.
    """
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
        nb = copy.copy(a)
        nb["segmentation"] = shifted
        bx0, by0 = float(xs.min()), float(ys.min())
        bx1, by1 = float(xs.max()), float(ys.max())
        frac = float(area / orig_area) if orig_area > 0 else 1.0
        cut = (frac < trunc_frac or bx0 <= trunc_edge_px or by0 <= trunc_edge_px
               or bx1 >= size - trunc_edge_px or by1 >= size - trunc_edge_px)
        if box_spans_footprint:
            # Extend the roof box to cover the footprint too.  The footprint
            # IS the roof translated (exact on 98.4% of instances), so the
            # union box follows analytically from the clipped roof box and
            # the offset -- no second polygon clip needed.  Clipped to the
            # crop, like the roof box, because a regression target outside
            # the image is not observable.
            dx, dy = (a.get("bd_offset") or offset_vector(a))
            bx0, bx1 = min(bx0, bx0 + dx), max(bx1, bx1 + dx)
            by0, by1 = min(by0, by0 + dy), max(by1, by1 + dy)
            # the footprint reaching beyond the window is a cut as well
            cut = cut or bx0 < 0.0 or by0 < 0.0 or bx1 > float(size) or by1 > float(size)
            bx0 = max(0.0, bx0); by0 = max(0.0, by0)
            bx1 = min(float(size), bx1); by1 = min(float(size), by1)
        nb["bbox"] = [bx0, by0, max(bx1 - bx0, 1e-3), max(by1 - by0, 1e-3)]
        nb["area"] = float(area)
        nb["clip_frac"] = frac
        nb["truncated"] = bool(cut)
        nb["bd_offset"] = a.get("bd_offset") or offset_vector(a)
        if with_base:
            base = a.get("offset")
            nbase = None
            if isinstance(base, list) and len(base) > 0:
                nbase, barea, _ = _clip_poly_list(base, rect, x0, y0, size)
                if nbase is not None and barea < min_area:
                    nbase = None
            nb["segmentation_base"] = nbase
        out.append(nb)
    return out
