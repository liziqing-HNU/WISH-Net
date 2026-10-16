# -*- coding: utf-8 -*-
"""Apply the reviewed corrections of the BDH labels -- the change table,
datasets/bdh_labels_v2_changes.json -- to a split's annotations, giving the
reviewed labels (labels v2) the paper's models are trained and evaluated on.
The input files are never touched.

The change table lists per split, in the order they are applied, the buildings
the review corrected.  A building is named by its ``orig_id`` (unique within a
split's annotations_fixed file and carried into its crop and ignore files),
and its tile (``tile``, checked).  The corrections:

    height_m, height_from   the height (m) and its source: "review" (the
                            reviewer's floor count or value), "parallax"
                            (derived on the review page from the length of the
                            roof -> footprint offset), "model" (a model's
                            prediction the reviewer took) or "literature" (a
                            published height, set after the other corrections)
    offset_px, offset_from  the roof -> footprint offset [dx, dy], px of the
                            tile, and its source: "review" (drawn or typed),
                            "suggested" (the review page's suggestion: the
                            direction of the model's offset or of the image's
                            buildings, the length of the labelled height) or
                            "model" (a model's offset)
    swap                    the annotated roof is the footprint
    exclude                 no height or offset can be trusted
    truncate                cut by the tile edge although the 1 px criterion
                            of the crops misses it
    delete                  a second annotation of another building

What they do:

    height    bd_height := height_m in the crop file and the whole-tile file
    offset    the footprint ring := the roof ring moved by the offset -- in the
              crop file bd_offset, segmentation_base and the bbox (roof union
              footprint, clipped to the image); in the whole-tile file
              bd_offset, the ``offset`` ring and ``coordinates``
    swap      after the offset, the roof and the footprint trade places: the
              roof becomes the footprint ring as it then is, the footprint the
              roof, the offset its negative; the crop file's box (their union)
              stays, the areas and the whole-tile file's box are the new roof's
    exclude   no height or offset supervision, as for a building without a
              floor count: bd_height 0 in the whole-tile file (it stays for
              class, box and mask), and in the crop files the annotation moves
              to the ignore file (ignore_reason "review_exclude", iscrowd 1)
    truncate  bd_height 0 in the whole-tile file (the training's treatment of a
              cut building), ``truncated`` true in the crop file (it stays among
              the annotations, as the crop files' cut buildings do)
    delete    the annotation goes from the crop, ignore and whole-tile files
              altogether (excluding it would keep its mask as ground truth)

A building that is excluded, truncated or deleted carries no other correction;
a swapped one is neither excluded nor truncated.  A building of the ignore
file (no recorded floor count) given a height moves into the crop annotations
with a new id, its footprint from the whole-tile file's ring or the given
offset.

    python tools/labels/apply_label_changes.py --changes datasets/bdh_labels_v2_changes.json \\
        --split train|val --crop <prepared/instances_<split>_crop2048.json> \\
        --whole <annotations_fixed/instances_<split>.json> --out <new folder>

The crop and ignore files are those prepare_bdh.py and make_ignore_regions.py
make from annotations_fixed (the ignore file is the crop file's name with
_ignore before .json); --whole is the split's whole-tile annotation file
(training reads the train one; split val may leave it out).  --out must be a
new folder.  Written there: the corrected files under their own names,
changes.csv (one row per changed field: key, split, crop id, orig_id, field --
the whole-tile file's fields prefixed with the split, e.g. train.bd_height --,
old, new, how), model_sourced.csv (the buildings whose height or offset is a
model's, the values before and after) and summary.json.  Every building of the
table must be found, else nothing is written.  tools/labels/make_labels_v2.sh
runs the whole chain from annotations_fixed.

The table records the review of 2026-10-10 and the corrections decided on
2026-10-12; with it this script writes, byte for byte, the files the paper's
runs read (their md5 in datasets/bdh_md5.txt).  Inside, a correction becomes
the review page's decision it was made with (HEIGHT_FROM, OFFSET_FROM) and the
buildings are processed by the review's rules below.
"""
import argparse
import csv
import json
import os
import sys

HEIGHT_SET = ("offset", "model", "fromedit", "custom")
HEIGHT_KEEP = ("keep", "unsure", None)
OFFSET_SET = ("edited", "model", "suggest", "zero")
OFFSET_KEEP = ("keep", None)
OVERRIDES = ("swap", "exclude", "truncate", "height", "delete")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def num(v):
    """The ignore file stores its numbers as strings."""
    return float(v) if isinstance(v, str) else v


def rings(poly):
    """COCO polygon [[x, y, x, y, ...], ...] -> [[(x, y), ...], ...]."""
    out = []
    for r in poly or []:
        r = [float(v) for v in r]
        out.append(list(zip(r[0::2], r[1::2])))
    return out


def flat(rs):
    return [[v for p in r for v in p] for r in rs]


def moved(poly, dx, dy):
    return [[(x + dx, y + dy) for x, y in r] for r in rings(poly)]


def closed(ring):
    """A ring's points -> the training file's ``coordinates``: [[x, y], ..., the first again]."""
    return [list(p) for p in ring] + [list(ring[0])]


def poly_area(poly):
    """The files' ``area``: the shoelace areas of the rings, summed."""
    s = 0.0
    for r in rings(poly):
        n = len(r)
        s += abs(sum(r[i][0] * r[(i + 1) % n][1] - r[(i + 1) % n][0] * r[i][1] for i in range(n))) / 2.0
    return s


def roof_bbox(poly):
    """The training file's bbox: the roof rings' extent, not clipped."""
    pts = [p for r in rings(poly) for p in r]
    x0, y0 = min(x for x, _ in pts), min(y for _, y in pts)
    return [x0, y0, max(x for x, _ in pts) - x0, max(y for _, y in pts) - y0]


def union_bbox(polys, w, h):
    xs = [x for rs in polys for r in rs for x, _ in r]
    ys = [y for rs in polys for r in rs for _, y in r]
    x0, y0 = max(min(xs), 0.0), max(min(ys), 0.0)
    x1, y1 = min(max(xs), float(w)), min(max(ys), float(h))
    return [x0, y0, max(x1 - x0, 0.0), max(y1 - y0, 0.0)]


def mean_offset(roof, foot):
    """The training loader's offset_vector: mean vertex delta of the first rings."""
    a, b = rings(roof), rings(foot)
    if not a or not b or len(a[0]) != len(b[0]) or not a[0]:
        return [0.0, 0.0]
    n = len(a[0])
    return [sum(q[0] - p[0] for p, q in zip(a[0], b[0])) / n, sum(q[1] - p[1] for p, q in zip(a[0], b[0])) / n]


def decided_offset(o):
    """The offset a decision sets, [dx, dy] -- None when it keeps the annotated one."""
    act = o.get("action")
    if act in OFFSET_KEEP:
        return None
    if act not in OFFSET_SET:
        raise ValueError("unknown offset action %r" % act)
    return [float(o["dx"]), float(o["dy"])]


# the change table's sources -> the review page's decisions they were made with
HEIGHT_FROM = {"review": "custom", "parallax": "offset", "model": "model"}
OFFSET_FROM = {"review": "edited", "suggested": "suggest", "model": "model"}
FLAGS = ("swap", "exclude", "truncate", "delete")


def from_changes(path, split, crop, ign):
    """The split's corrections of the change table -> (decisions {key: decision},
    overrides {section: {key: ""} or {key: {"value_m": m}}}, keys in table order)
    with the review page's keys: "<split>_<crop file id>", or "<split>_n<ignore
    file id>" for a building of the ignore file."""
    doc = load(path)
    if split not in doc:
        sys.exit("%s: no corrections for split %s" % (path, split))
    imgs = {i["id"]: i for i in crop["images"]}
    where = {}
    for x in crop["annotations"]:
        where[str(x["orig_id"])] = ("%s_%s" % (split, x["id"]), imgs[x["image_id"]]["parent_file_name"])
    for x in ign["annotations"]:
        if str(x.get("orig_id")) not in where:
            where[str(x.get("orig_id"))] = ("%s_n%s" % (split, x["id"]),
                                            imgs[int(num(x["image_id"]))]["parent_file_name"])
    dec, ov, order, missing = {}, {s: {} for s in OVERRIDES}, [], []
    for e in doc[split]:
        known = {"orig_id", "tile", "height_m", "height_from", "offset_px", "offset_from"} | set(FLAGS)
        if set(e) - known:
            sys.exit("%s: unknown fields %s in %s" % (path, sorted(set(e) - known), e))
        hit = where.get(str(e["orig_id"]))
        if hit is None or hit[1] != e["tile"]:
            missing.append("%s (%s)" % (e["orig_id"], e["tile"]))
            continue
        k = hit[0]
        if k in order:
            sys.exit("%s: building %s twice" % (path, e["orig_id"]))
        flags = [f for f in FLAGS if e.get(f)]
        if [f for f in flags if f != "swap"] and (len(flags) > 1 or "height_m" in e or "offset_px" in e):
            sys.exit("%s: %s with another correction: %s" % (path, flags, e))
        d = {}
        if "height_m" in e:
            v = e["height_m"]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 < v < 1000:
                sys.exit("%s: height_m in metres, got %r" % (path, v))
            if e.get("height_from") == "literature":
                if "_n" in k:
                    sys.exit("%s: a literature height for a building without a floor count: %s" % (path, e))
                ov["height"][k] = {"value_m": v}
            elif e.get("height_from") in HEIGHT_FROM:
                d["height"] = {"action": HEIGHT_FROM[e["height_from"]], "value_m": v}
            else:
                sys.exit("%s: height_from %r" % (path, e.get("height_from")))
        if "offset_px" in e:
            if e.get("offset_from") not in OFFSET_FROM:
                sys.exit("%s: offset_from %r" % (path, e.get("offset_from")))
            d["offset"] = {"action": OFFSET_FROM[e["offset_from"]], "dx": e["offset_px"][0], "dy": e["offset_px"][1]}
        if e.get("exclude"):
            d["exclude"] = True
        for f in ("swap", "truncate", "delete"):
            if e.get(f):
                if "_n" in k:
                    sys.exit("%s: %s for a building without a floor count: %s" % (path, f, e))
                ov[f][k] = ""
        if d:
            dec[k] = d
        order.append(k)
    if missing:
        sys.exit("buildings of the change table not found in the annotations (nothing written): %s" % missing[:20])
    return dec, ov, order


def swap_crop(c):
    """The crop file's roof and footprint trade places -> (old offset, new offset)."""
    roof = c["segmentation"]
    off = list(c.get("bd_offset") or mean_offset(roof, c.get("segmentation_base")))
    foot = c.get("segmentation_base") or flat(moved(roof, off[0], off[1]))
    c["segmentation"], c["segmentation_base"] = foot, roof
    c["bd_offset"] = [-off[0], -off[1]]
    c["area"] = poly_area(foot)
    return off, c["bd_offset"]


def swap_train(t):
    """The training file's roof and footprint trade places -> (old offset, new offset)."""
    roof, foot = t["segmentation"], t.get("offset")
    off = list(t.get("bd_offset") or mean_offset(roof, foot))
    if not foot:
        foot = flat(moved(roof, off[0], off[1]))
    t["segmentation"], t["offset"] = foot, roof
    t["bd_offset"] = [-off[0], -off[1]]
    if "coordinates" in t:
        t["coordinates"] = closed(rings(roof)[0])
    t["bbox"] = roof_bbox(foot)
    t["area"] = poly_area(foot)
    return off, t["bd_offset"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--changes", required=True, help="the change table (datasets/bdh_labels_v2_changes.json)")
    ap.add_argument("--crop", required=True, help="instances_<split>_crop2048.json")
    ap.add_argument("--whole", default="",
                    help="annotations_fixed/instances_<split>.json, the split's whole-tile annotations")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train", choices=("train", "val"))
    a = ap.parse_args()
    if a.split == "train" and not a.whole:
        sys.exit("--split train needs --whole (annotations_fixed/instances_train.json, the file training reads)")
    ign_path = a.crop[:-len(".json")] + "_ignore.json"
    crop, ign = load(a.crop), load(ign_path)
    whole = load(a.whole) if a.whole else None
    dec, ov, order = from_changes(a.changes, a.split, crop, ign)
    out_crop = os.path.join(a.out, os.path.basename(a.crop))
    out_ign = os.path.join(a.out, os.path.basename(ign_path))
    out_whole = os.path.join(a.out, os.path.basename(a.whole)) if a.whole else None
    outs = [p for p in (out_crop, out_ign, out_whole, os.path.join(a.out, "changes.csv"),
                        os.path.join(a.out, "model_sourced.csv"), os.path.join(a.out, "summary.json")) if p]
    for p in outs:
        if os.path.exists(p):
            sys.exit("%s exists: --out must be a new folder" % p)
        if os.path.abspath(p) in {os.path.abspath(q) for q in (a.crop, ign_path, a.whole, a.changes)
                                  if q}:
            sys.exit("%s would overwrite an input" % p)

    imgs = {i["id"]: i for i in crop["images"]}
    by_id = {a_["id"]: a_ for a_ in crop["annotations"]}
    ign_by_id = {str(a_["id"]): a_ for a_ in ign["annotations"]}
    # the whole-tile file's fields are logged as <split>.<field> (train.bd_height, ...)
    W = a.split + "."
    wh_by_orig = {}
    if whole is not None:
        for t in whole["annotations"]:
            if t["orig_id"] in wh_by_orig:
                sys.exit("orig_id %s twice in %s" % (t["orig_id"], a.whole))
            wh_by_orig[t["orig_id"]] = t

    mine, extra = order, []
    missing, rows, model = [], [], []
    moved_in, moved_out, deleted = [], [], []
    next_id = max(int(x["id"]) for x in crop["annotations"]) + 1
    counts = {"height": 0, "offset": 0, "exclude": 0, "override_exclude": 0, "override_truncate": 0,
              "override_swap": 0, "override_height": 0, "override_delete": 0, "unchanged": 0}

    def log(k, cid, orig, field, old, new, what):
        rows.append({"key": k, "split": a.split, "crop_id": cid, "orig_id": orig, "field": field,
                     "old": json.dumps(old), "new": json.dumps(new), "decision": what})

    for k in mine + extra:
        d = dec.get(k) or {}
        sid = k[len(a.split) + 1:]
        h = d.get("height") or {}
        o = d.get("offset") or {}
        ha, oa = h.get("action"), o.get("action")
        try:
            off = decided_offset(o)
        except ValueError as e:
            sys.exit("%s: %s" % (k, e))
        exclude = bool(d.get("exclude"))
        if sid.startswith("n"):
            src = ign_by_id.get(sid[1:])
            if src is None:
                missing.append(k)
                continue
            orig = int(num(src["orig_id"]))
            t = wh_by_orig.get(orig) if whole is not None else None
            if whole is not None and t is None:
                missing.append(k)
                continue
            if exclude or ha not in HEIGHT_SET:
                if "model" in (ha, oa):
                    model.append({"key": k, "orig_id": orig, "h": 0, "o": 0, "h_old": 0.0, "o_old": None,
                                  "status": "exclude" if exclude else "kept without a floor count"})
                counts["unchanged"] += 1
                continue
            # a building without a floor count gets its height: into the crop annotations
            img = imgs[int(num(src["image_id"]))]
            roof = src["segmentation"] if not isinstance(src["segmentation"], str) else json.loads(src["segmentation"])
            if off is not None:
                foot_off = off
            elif t is not None and t.get("offset"):
                foot_off = mean_offset(t["segmentation"], t["offset"])
            else:
                foot_off = [0.0, 0.0]
            foot = moved(roof, foot_off[0], foot_off[1])
            new = {"id": next_id, "image_id": img["id"], "category_id": 1, "segmentation": roof,
                   "area": num(src["area"]), "bbox": union_bbox([rings(roof), foot], img["width"], img["height"]),
                   "iscrowd": 0, "bd_height": float(h["value_m"]), "bd_category_id": 1, "orig_id": orig,
                   "clip_frac": 1.0, "truncated": False, "bd_offset": foot_off, "segmentation_base": flat(foot)}
            next_id += 1
            moved_in.append((sid[1:], new))
            log(k, new["id"], orig, "moved", "ignore (unknown_height)", "annotations", ha)
            log(k, new["id"], orig, "bd_height", 0.0, new["bd_height"], ha)
            if "model" in (ha, oa):
                model.append({"key": k, "orig_id": orig, "h": int(ha == "model"), "o": int(oa == "model"),
                              "h_old": 0.0, "o_old": None, "status": "applied"})
            if t is not None:
                log(k, new["id"], orig, W + "bd_height", t["bd_height"], new["bd_height"], ha)
                t["bd_height"] = new["bd_height"]
                if off is not None:
                    tf = moved(t["segmentation"], off[0], off[1])
                    log(k, new["id"], orig, W + "bd_offset", mean_offset(t["segmentation"], t.get("offset")), off, oa)
                    t["bd_offset"], t["offset"] = off, flat(tf)
                    if "coordinates" in t:
                        t["coordinates"] = closed(tf[0])
            counts["height"] += 1
            counts["offset"] += int(off is not None)
            continue

        c = by_id.get(int(sid))
        if c is None:
            missing.append(k)
            continue
        orig = c["orig_id"]
        t = wh_by_orig.get(orig) if whole is not None else None
        if whole is not None and t is None:
            missing.append(k)
            continue
        mrow = None
        if "model" in (ha, oa):
            mrow = {"key": k, "orig_id": orig, "h": int(ha == "model"), "o": int(oa == "model"),
                    "h_old": c["bd_height"], "o_old": c.get("bd_offset"), "status": "applied"}
            model.append(mrow)
        if k in ov["delete"]:
            # a duplicate annotation: out of every file below, its decision not applied
            deleted.append((k, c["id"], orig))
            log(k, c["id"], orig, "deleted", "annotations", None, "override:delete")
            if t is not None:
                log(k, c["id"], orig, W + "deleted", "annotations", None, "override:delete")
            if mrow:
                mrow.update(h=0, o=0, status="override:delete")
            counts["override_delete"] += 1
            continue
        for sec, what in (("swap", "swapped"), ("truncate", "truncated"), ("height", "given a height")):
            if k in ov[sec] and exclude:
                sys.exit("%s: excluded and %s" % (k, what))
        if exclude or k in ov["exclude"]:
            what = "exclude" if exclude else "override:exclude"
            moved_out.append(c["id"])
            log(k, c["id"], orig, "moved", "annotations", "ignore (review_exclude)", what)
            if t is not None:
                log(k, c["id"], orig, W + "bd_height", t["bd_height"], 0.0, what)
                t["bd_height"] = 0.0
            if mrow:
                mrow.update(h=0, o=0, status=what)
            counts["exclude" if exclude else "override_exclude"] += 1
            continue
        if k in ov["truncate"]:
            log(k, c["id"], orig, "truncated", bool(c.get("truncated", False)), True, "override:truncate")
            c["truncated"] = True
            if t is not None:
                log(k, c["id"], orig, W + "bd_height", t["bd_height"], 0.0, "override:truncate")
                t["bd_height"] = 0.0
            if mrow:
                mrow.update(h=0, o=0, status="override:truncate")
            counts["override_truncate"] += 1
            continue
        changed = False
        if ha in HEIGHT_SET:
            v = float(h["value_m"])
            if v != float(c["bd_height"]):
                log(k, c["id"], orig, "bd_height", c["bd_height"], v, ha)
                c["bd_height"] = v
                if t is not None:
                    log(k, c["id"], orig, W + "bd_height", t["bd_height"], v, ha)
                    t["bd_height"] = v
                changed = True
                counts["height"] += 1
        elif ha not in HEIGHT_KEEP:
            sys.exit("%s: unknown height action %r" % (k, ha))
        if off is not None:
            img = imgs[c["image_id"]]
            foot = moved(c["segmentation"], off[0], off[1])
            log(k, c["id"], orig, "bd_offset", c.get("bd_offset"), off, oa)
            c["bd_offset"] = off
            c["segmentation_base"] = flat(foot)
            c["bbox"] = union_bbox([rings(c["segmentation"]), foot], img["width"], img["height"])
            if t is not None:
                tf = moved(t["segmentation"], off[0], off[1])
                log(k, c["id"], orig, W + "bd_offset", mean_offset(t["segmentation"], t.get("offset")), off, oa)
                t["bd_offset"], t["offset"] = off, flat(tf)
                if "coordinates" in t:
                    t["coordinates"] = closed(tf[0])
            changed = True
            counts["offset"] += 1
        if k in ov["swap"]:
            # roof <-> footprint, after the decision; logged as the offset before -> after
            o0, o1 = swap_crop(c)
            log(k, c["id"], orig, "swap", o0, o1, "override:swap")
            if t is not None:
                p0, p1 = swap_train(t)
                log(k, c["id"], orig, W + "swap", p0, p1, "override:swap")
            changed = True
            counts["override_swap"] += 1
        if k in ov["height"]:
            # after the decision: the override's height in both files
            v = float(ov["height"][k]["value_m"])
            hit = False
            if mrow:
                mrow["h"] = 0           # the override's height, not the model's
            if v != float(c["bd_height"]):
                log(k, c["id"], orig, "bd_height", c["bd_height"], v, "override:height")
                c["bd_height"], hit = v, True
            if t is not None and v != float(t["bd_height"]):
                log(k, c["id"], orig, W + "bd_height", t["bd_height"], v, "override:height")
                t["bd_height"], hit = v, True
            changed = changed or hit
            counts["override_height"] += int(hit)
        if not changed:
            counts["unchanged"] += 1

    if missing:
        sys.exit("buildings not found in the annotations (nothing written): %s" % missing[:20])

    # the deleted duplicates: out of the crop, ignore and training files
    del_ids = {cid for _, cid, _ in deleted}
    del_orig = {str(o) for _, _, o in deleted}
    crop["annotations"] = [x for x in crop["annotations"] if x["id"] not in del_ids]
    gone_ign = [x for x in ign["annotations"] if str(x.get("orig_id")) in del_orig]
    for x in gone_ign:
        k = next(k for k, _, o in deleted if str(o) == str(x.get("orig_id")))
        log(k, None, x.get("orig_id"), "ignore.deleted", "ignore (%s)" % x.get("ignore_reason"), None, "override:delete")
    ign["annotations"] = [x for x in ign["annotations"] if str(x.get("orig_id")) not in del_orig]
    if whole is not None:
        whole["annotations"] = [x for x in whole["annotations"] if str(x["orig_id"]) not in del_orig]

    # the moves between the crop annotations and the ignore file
    gone = {str(i) for i, _ in moved_in}
    ign["annotations"] = [x for x in ign["annotations"] if str(x["id"]) not in gone]
    crop["annotations"] += [n for _, n in moved_in]
    out_ids = set(moved_out)
    next_ign = max(int(num(x["id"])) for x in ign["annotations"]) + 1 if ign["annotations"] else 1
    for x in [x for x in crop["annotations"] if x["id"] in out_ids]:
        ign["annotations"].append({"id": str(next_ign), "image_id": str(x["image_id"]), "category_id": "1",
                                   "segmentation": json.dumps(x["segmentation"]), "bbox": json.dumps(x["bbox"]),
                                   "area": str(x["area"]), "iscrowd": "1", "orig_id": str(x["orig_id"]),
                                   "bd_height": "0.0", "ignore_reason": "review_exclude"})
        next_ign += 1
    crop["annotations"] = [x for x in crop["annotations"] if x["id"] not in out_ids]

    os.makedirs(a.out, exist_ok=True)
    for path, doc in ((out_crop, crop), (out_ign, ign), (out_whole, whole)):
        if doc is not None:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f)
    with open(os.path.join(a.out, "changes.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "split", "crop_id", "orig_id", "field", "old", "new", "decision"])
        w.writeheader()
        w.writerows(rows)
    # the buildings whose height or offset is the model's: one row per annotation of the
    # corrected crop file (its id there, v2_ann_id), one without an id when none is left
    by_orig = {}
    for x in crop["annotations"]:
        by_orig.setdefault(str(x["orig_id"]), []).append(x)
    mrows = []
    for m in sorted(model, key=lambda m: (str(m["orig_id"]), m["key"])):
        for x in (by_orig.get(str(m["orig_id"])) or [None]) if m["status"] == "applied" else [None]:
            mrows.append({"key": m["key"], "split": a.split, "v2_ann_id": "" if x is None else x["id"],
                          "image_id": "" if x is None else x["image_id"], "orig_id": m["orig_id"],
                          "height_from_model": m["h"], "offset_from_model": m["o"],
                          "height_old": json.dumps(m["h_old"]),
                          "height_new": "" if x is None else json.dumps(x["bd_height"]),
                          "offset_old": json.dumps(m["o_old"]),
                          "offset_new": "" if x is None else json.dumps(x.get("bd_offset")),
                          "status": m["status"]})
    with open(os.path.join(a.out, "model_sourced.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "split", "v2_ann_id", "image_id", "orig_id", "height_from_model",
                                          "offset_from_model", "height_old", "height_new", "offset_old",
                                          "offset_new", "status"])
        w.writeheader()
        w.writerows(mrows)
    summary = {"changes": a.changes, "split": a.split, "buildings_in_table": len(order),
               "per_override": {s: len(ov[s]) for s in OVERRIDES},
               "buildings": counts, "moved_into_annotations": len(moved_in), "moved_to_ignore": len(moved_out),
               "annotations": {"crop": len(crop["annotations"]), "ignore": len(ign["annotations"]),
                               a.split: len(whole["annotations"]) if whole is not None else None},
               "model_sourced": {"decisions": len(model), "height": sum(m["h"] for m in model),
                                 "offset": sum(m["o"] for m in model),
                                 "both": sum(bool(m["h"] and m["o"]) for m in model),
                                 "not_applied": sum(m["status"] != "applied" for m in model),
                                 "rows": len(mrows)},
               "inputs": {"crop": a.crop, "ignore": ign_path, a.split: a.whole or None}}
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("[labels v2] %d field changes written to %s" % (len(rows), a.out))


if __name__ == "__main__":
    main()
