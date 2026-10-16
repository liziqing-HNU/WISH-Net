# -*- coding: utf-8 -*-
"""The ignore regions of the evaluation crops (BDH_EVAL.IGNORE_REGIONS).

The evaluation treats two kinds of building as ignore regions (COCO iscrowd)
in the detection and segmentation metrics: a detection on one is neither a
true nor a false positive, missing one is not a false negative, and none of
them enters a height metric.

* buildings cut by the window or the tile edge -- already in the crop file with
  ``truncated``, turned into ignore regions by the evaluator itself
  (BDH_EVAL.IGNORE_TRUNCATED);
* buildings the crop file leaves out -- written here: every annotation of the
  parent tile with a visible part in the crop that is NOT in the crop's ground
  truth, i.e. no recorded floor count (bd_height 0, "unknown_height"), or a
  visible part below the size / fraction gates of prepare_bdh.py
  ("below_gate": a sliver at the crop edge, or a building under 32 px^2).

For each instances_<split>_crop<N>.json in the prepared folder this writes
instances_<split>_crop<N>_ignore.json next to it, with annotations in the
crop's image ids and coordinates, which the evaluator appends as iscrowd.

    python tools/labels/make_ignore_regions.py --root datasets/bdh --prepared datasets/bdh/prepared \\
        [--ann-dir annotations_fixed] [--splits train val] [--crops 2048]
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bdh_crops import clip_annotations  # noqa: E402

# written into the files' info: the name of the script the paper's files were
# made with, kept so that the files come out byte for byte the same
SCRIPT_NAME = "tools_bdh/make_ignore_regions.py"


def build(root, prepared, ann_dir, split, crop, min_height=3.0):
    src = os.path.join(root, ann_dir, "instances_%s.json" % split)
    dst_gt = os.path.join(prepared, "instances_%s_crop%d.json" % (split, crop))
    if not os.path.exists(dst_gt):
        print("skip %s (no eval json)" % dst_gt)
        return
    with open(src, "r", encoding="utf-8") as f:
        raw = json.load(f)
    with open(dst_gt, "r", encoding="utf-8") as f:
        gt = json.load(f)

    # the crop's GT is exactly what prepare_bdh.py kept: a recorded height and
    # a visible part passing its size / fraction gates (stored in the json's
    # info), re-derived here geometrically
    info = gt.get("info", {})
    min_area = float(info.get("min_inst_area", 32.0))
    min_frac = float(info.get("min_inst_frac", 0.1))
    min_height = float(info.get("min_height", min_height))
    raw_by_img = {}
    for a in raw["annotations"]:
        raw_by_img.setdefault(a["image_id"], []).append(a)

    out, why, n_gt = [], Counter(), 0
    for img in gt["images"]:
        x0, y0, w, h = img["crop_box"]
        if w != crop or h != crop:
            raise SystemExit("%s: crop_box %s does not match crop %d" % (dst_gt, img["crop_box"], crop))
        for a in raw_by_img.get(img["parent_id"], []):
            # any visible part counts: no size or fraction gate, roof box only
            vis = clip_annotations([a], x0, y0, crop, min_area=1.0, min_frac=0.0,
                                   box_spans_footprint=False)
            if not vis:
                continue
            tall = float(a.get("bd_height", 0.0)) >= min_height
            if tall and clip_annotations([a], x0, y0, crop, min_area, min_frac):
                n_gt += 1
                continue
            v = vis[0]
            reason = "below_gate" if tall else "unknown_height"
            why[reason] += 1
            out.append({
                "id": len(out) + 1,
                "image_id": img["id"],
                "category_id": v["category_id"],
                "segmentation": v["segmentation"],
                "bbox": v["bbox"],
                "area": v["area"],
                "iscrowd": 1,
                "orig_id": v.get("orig_id"),
                "bd_height": float(v.get("bd_height", 0.0)),
                "ignore_reason": reason,
            })
    if n_gt != len(gt["annotations"]):
        raise SystemExit("%s: re-derived %d GT buildings but the json has %d -- the gates differ"
                         % (dst_gt, n_gt, len(gt["annotations"])))

    dst = os.path.splitext(dst_gt)[0] + "_ignore.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"info": {"description": "ignore regions of %s (%s)"
                            % (os.path.basename(dst_gt), SCRIPT_NAME), "source": os.path.basename(src)},
                   "annotations": out}, f)
    n_tr = sum(1 for a in gt["annotations"] if a.get("truncated", False))
    print("%-5s crop %4d  images %4d  GT %6d (truncated %5d)  ignore regions %5d  %s  -> %s"
          % (split, crop, len(gt["images"]), len(gt["annotations"]), n_tr, len(out), dict(why), dst))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the dataset folder (datasets/bdh)")
    ap.add_argument("--prepared", required=True, help="the folder of the crop files")
    ap.add_argument("--ann-dir", default="annotations_fixed")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--crops", nargs="+", type=int, default=[2048])
    args = ap.parse_args()
    for s in args.splits:
        for c in args.crops:
            build(args.root, args.prepared, args.ann_dir, s, c)


if __name__ == "__main__":
    main()
