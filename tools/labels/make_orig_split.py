# -*- coding: utf-8 -*-
"""Rebuild BDH's original random train / val split (558 / 373 tiles, no test).

BDH-RGB-SAR replaced the original BDH-Dataset split, whose train and val tiles
are randomly interleaved, by a geographically isolated train / val / test split
(561 / 186 / 184 tiles, 6x6-tile super-blocks); metadata/splits.csv keeps each
tile's original split in ``orig_split``.  The paper uses the original split:

  <dst>/<ann_dir>/instances_{train,val}.json   the three geographic annotation
        files merged and re-split by orig_split (ids renumbered, "split" set to
        the original split, "geo_split" added, all other fields unchanged);
  <dst>/images/{train,val}/<tile>.tif          symbolic links to <src>/images/<geo split>/;
  <dst>/<sar dir>/{train,val}/<tile>_sar.tif   for each --sar-dir (the GF-3 tiles sar_gf3 and
        the S1 InSAR tiles s1_insar_asc, named alike): symbolic links to
        <src>/<sar dir>/<geo split>/, and links to the folder's own .json files
        (S1_INSAR_ASC.json).

No data is copied.  The evaluation crops, their ignore regions and the reviewed
labels are then made by tools/labels/make_labels_v2.sh:

    python tools/labels/make_orig_split.py --src <BDH-RGB-SAR> --dst datasets/bdh \\
        [--sar-dir sar_gf3 s1_insar_asc]
"""
import argparse
import csv
import glob
import json
import os
from collections import Counter

GEO_SPLITS = ("train", "val", "test")
ORIG = {"bdh_train": "train", "bdh_val": "val"}


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="BDH-RGB-SAR: images/, the SAR folders, annotations, metadata/")
    ap.add_argument("--dst", required=True, help="the dataset folder of the original split (datasets/bdh)")
    ap.add_argument("--ann-dir", default="annotations_fixed")
    ap.add_argument("--sar-dir", nargs="+", default=["sar_gf3", "s1_insar_asc"],
                    help="SAR folders of <src> with per-split tiles named by sar_file_name")
    a = ap.parse_args()

    orig = {}
    with open(os.path.join(a.src, "metadata", "splits.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            orig[os.path.splitext(row["file_name"])[0]] = ORIG[row["orig_split"]]

    out = {s: {"images": [], "annotations": []} for s in ORIG.values()}
    head = None
    moves = Counter()
    for geo in GEO_SPLITS:
        with open(os.path.join(a.src, a.ann_dir, "instances_%s.json" % geo), encoding="utf-8") as f:
            d = json.load(f)
        head = head or {k: v for k, v in d.items() if k not in ("images", "annotations")}
        by_img = {}
        for ann in d["annotations"]:
            by_img.setdefault(ann["image_id"], []).append(ann)
        for img in d["images"]:
            stem = os.path.splitext(img["file_name"])[0]
            split = orig[stem]
            moves[(geo, split)] += 1
            o = out[split]
            new_id = len(o["images"]) + 1
            o["images"].append(dict(img, id=new_id, split=split, geo_split=geo))
            for ann in by_img.get(img["id"], []):
                o["annotations"].append(dict(ann, id=len(o["annotations"]) + 1, image_id=new_id))
            link(os.path.join(a.src, "images", geo, img["file_name"]),
                 os.path.join(a.dst, "images", split, img["file_name"]))
            if img.get("sar_file_name"):
                for sar_dir in a.sar_dir:
                    link(os.path.join(a.src, sar_dir, geo, img["sar_file_name"]),
                         os.path.join(a.dst, sar_dir, split, img["sar_file_name"]))
    for sar_dir in a.sar_dir:
        for meta in glob.glob(os.path.join(a.src, sar_dir, "*.json")):
            link(meta, os.path.join(a.dst, sar_dir, os.path.basename(meta)))

    os.makedirs(os.path.join(a.dst, a.ann_dir), exist_ok=True)
    for split, o in out.items():
        info = dict(head.get("info") or {}, split="BDH original %s (random, from metadata/splits.csv "
                                                  "orig_split)" % split)
        with open(os.path.join(a.dst, a.ann_dir, "instances_%s.json" % split), "w", encoding="utf-8") as f:
            json.dump(dict(head, info=info, images=o["images"], annotations=o["annotations"]), f)
        print("%s: %d tiles, %d annotations" % (split, len(o["images"]), len(o["annotations"])))
    print("from geographic split -> original split:",
          ", ".join("%s->%s %d" % (g, s, n) for (g, s), n in sorted(moves.items())))


if __name__ == "__main__":
    main()
