# -*- coding: utf-8 -*-
"""The evaluation crops of a BDH split: instances_<split>_crop<N>.json.

With --crop 2048 (the paper's protocol) every tile is one crop, the whole
tile; a smaller --crop cuts each tile into a fixed grid of crops.  Each crop
is one image entry carrying ``crop_box`` [x0, y0, w, h] and the parent tile's
name (``parent_file_name``); its annotations are clipped into crop pixels
(bdh_crops.clip_annotations with the footprint, ``segmentation_base``), with
``truncated`` and ``clip_frac`` set, so COCOeval runs on the file as it is.
Buildings without a recorded floor count (bd_height below --min-height) and
visible parts below the size / fraction gates are left out;
make_ignore_regions.py turns them into ignore regions.

    python tools/labels/prepare_bdh.py --root datasets/bdh --out datasets/bdh/prepared \\
        [--ann-dir annotations_fixed] [--splits train val] [--crop 2048]

The annotations are read from <root>/<ann-dir>/instances_<split>.json (the
original split, tools/labels/make_orig_split.py).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bdh_crops import clip_annotations  # noqa: E402


def build(split, root, out_dir, crop, min_area, min_frac, min_height,
          ann_dir="annotations_fixed"):
    src = os.path.join(root, ann_dir, "instances_%s.json" % split)
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    anns_by_img = {}
    n_drop = 0
    for a in data["annotations"]:
        # bd_height < 3 m means "0 floors": the height was never recorded, so the
        # annotation is invalid for this task and is removed from the GT.
        if float(a.get("bd_height", 0.0)) < min_height:
            n_drop += 1
            continue
        anns_by_img.setdefault(a["image_id"], []).append(a)

    images, annotations = [], []
    next_img_id = 1
    next_ann_id = 1
    for img in data["images"]:
        W, H = img["width"], img["height"]
        anns = anns_by_img.get(img["id"], [])
        xs = list(range(0, max(W - crop, 0) + 1, crop)) or [0]
        ys = list(range(0, max(H - crop, 0) + 1, crop)) or [0]
        for y0 in ys:
            for x0 in xs:
                kept = clip_annotations(anns, x0, y0, crop, min_area, min_frac,
                                        with_base=True)
                rec = dict(img)
                rec["id"] = next_img_id
                rec["parent_id"] = img["id"]
                rec["parent_file_name"] = img["file_name"]
                rec["file_name"] = "%s__x%04d_y%04d" % (img["file_name"], x0, y0)
                rec["parent_width"] = W
                rec["parent_height"] = H
                rec["width"] = crop
                rec["height"] = crop
                rec["crop_box"] = [x0, y0, crop, crop]
                images.append(rec)
                for a in kept:
                    na = dict(a)
                    na["id"] = next_ann_id
                    na["image_id"] = next_img_id
                    na["iscrowd"] = 0
                    # the full-resolution rings are big and no longer needed:
                    # `segmentation` (roof) and `segmentation_base` (footprint)
                    # are already clipped into crop coordinates, and the
                    # roof -> base translation is kept as `bd_offset`
                    na.pop("offset", None)
                    na.pop("coordinates", None)
                    annotations.append(na)
                    next_ann_id += 1
                next_img_id += 1

    out = {
        "info": {
            "description": "BDH-RGB-SAR deterministic %dx%d eval crops" % (crop, crop),
            "source": os.path.basename(src),
            "crop": crop,
            "min_inst_area": min_area,
            "min_inst_frac": min_frac,
            "min_height": min_height,
        },
        "images": images,
        "annotations": annotations,
        "categories": data["categories"],
    }
    dst = os.path.join(out_dir, "instances_%s_crop%d.json" % (split, crop))
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(
        "%-5s  tiles %4d -> crops %4d   anns %6d (-%d with h<%.0fm) -> %6d   %s"
        % (split, len(data["images"]), len(images), len(data["annotations"]),
           n_drop, min_height, len(annotations), dst)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the dataset folder (datasets/bdh)")
    ap.add_argument("--out", required=True, help="the folder of the crop files")
    ap.add_argument("--crop", type=int, default=2048)
    ap.add_argument("--min-area", type=float, default=32.0)
    ap.add_argument("--min-frac", type=float, default=0.1)
    ap.add_argument("--min-height", type=float, default=3.0,
                    help="drop annotations with bd_height below this (metres)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--ann-dir", default="annotations_fixed",
                    help="annotation subdirectory under --root")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for s in args.splits:
        build(s, args.root, args.out, args.crop, args.min_area, args.min_frac,
              args.min_height, args.ann_dir)


if __name__ == "__main__":
    main()
