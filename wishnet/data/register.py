# -*- coding: utf-8 -*-
"""Dataset registration: bdh_train / bdh_val (+ bdh_train_crops).

Training reads the whole tiles (the mapper cuts the window); val reads
the prepared evaluation crops (instances_<split>_crop<size>.json, size 2048 =
the whole tile; tools/labels/ makes them).  The records carry the GF-3 file of
the same ground (sar_file_name; the InSAR tile has the same name in its own
folder).
"""
import json
import logging
import os

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

logger = logging.getLogger(__name__)

BDH_CATEGORIES = [{"id": 1, "name": "building", "color": [220, 20, 60]}]


def offset_vector(ann):
    """Roof -> footprint translation [dx, dy] of an annotation, px: the mean
    vertex delta between the roof ring (``segmentation``) and the footprint ring
    (``offset``); [0, 0] when there is no footprint."""
    seg, off = ann.get("segmentation"), ann.get("offset")
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


def _load_bdh_json(json_file, image_root, sar_root, min_height):
    """Records of one split.  A building whose bd_height is below min_height (no
    recorded floor count) stays in as class / box / mask ground truth without
    height or offset terms (``bd_height_valid`` False)."""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    anns_by_image, n_seg = {}, 0
    for a in data["annotations"]:
        if float(a.get("bd_height", 0.0)) < min_height:
            n_seg += 1
        anns_by_image.setdefault(a["image_id"], []).append(a)
    if n_seg:
        logger.info("[WISH-Net] %s: %d buildings without a floor count kept for class / box / mask only",
                    os.path.basename(json_file), n_seg)
    records = []
    for img in data["images"]:
        rec = {
            "file_name": os.path.join(image_root, img.get("parent_file_name", img["file_name"])),
            "sar_file_name": os.path.join(sar_root, img["sar_file_name"]),
            "image_id": img["id"],
            "height": img["height"],
            "width": img["width"],
            "crop_box": img.get("crop_box", None),          # [x0, y0, w, h] in the tile
            "parent_height": img.get("parent_height", img["height"]),
            "parent_width": img.get("parent_width", img["width"]),
        }
        objs = []
        for a in anns_by_image.get(img["id"], []):
            objs.append({
                "bbox": a["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": a["segmentation"],          # roof polygon
                "category_id": 0,
                "iscrowd": a.get("iscrowd", 0),
                "bd_height": float(a["bd_height"]),
                "bd_offset": a.get("bd_offset") or offset_vector(a),
                "bd_height_valid": float(a.get("bd_height", 0.0)) >= min_height,
                # set by the prepared evaluation crops; a training window is cut
                # anew by the mapper, which recomputes both
                "truncated": bool(a.get("truncated", False)),
                "clip_frac": float(a.get("clip_frac", 1.0)),
            })
        rec["annotations"] = objs
        records.append(rec)
    return records


def register_bdh(root, prepared, eval_crop, min_height, ann_dir, sar_root):
    """``sar_root``: folder holding {train,val}/<sar tile>."""
    splits = {
        "bdh_train": (os.path.join(root, ann_dir, "instances_train.json"), "train"),
        "bdh_val": (os.path.join(prepared, "instances_val_crop%d.json" % eval_crop), "val"),
        # the fixed-set readout's training tiles, never trained on as such
        "bdh_train_crops": (os.path.join(prepared, "instances_train_crop%d.json" % eval_crop), "train"),
    }
    for name, (json_file, split) in splits.items():
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
        image_root = os.path.join(root, "images", split)
        s_root = os.path.join(sar_root, split)
        DatasetCatalog.register(
            name, lambda j=json_file, i=image_root, s=s_root: _load_bdh_json(j, i, s, min_height))
        MetadataCatalog.get(name).set(
            json_file=json_file, min_height=min_height, image_root=image_root, sar_root=s_root,
            evaluator_type="bdh", thing_classes=[c["name"] for c in BDH_CATEGORIES],
            thing_colors=[c["color"] for c in BDH_CATEGORIES],
            thing_dataset_id_to_contiguous_id={1: 0})
