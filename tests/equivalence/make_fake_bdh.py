# -*- coding: utf-8 -*-
"""A small fake BDH dataset for the code tests: random tiles in BDH's layout,
with buildings (roof polygon, footprint polygon, floor-count height), GF-3
sigma0 in dB and a 17-channel InSAR tile per tile.

    python tests/equivalence/make_fake_bdh.py <root> [--size 512] [--train 2] [--val 1] [--seed 0]

Layout (as datasets/README.md):

    <root>/images/{train,val}/<tile>.tif            RGB uint8, size x size
    <root>/sar_gf3/{train,val}/<tile>.tif           float32 dB, size x size
    <root>/s1_insar_asc/{train,val}/<tile>.tif      float32 (17, size/16, size/16)
    <root>/s1_insar_asc/S1_INSAR_ASC.json           channel names, train_orig statistics
    <root>/annotations_v2/instances_train.json      whole training tiles
    <root>/prepared_v2/instances_{val,train}_crop<size>.json   the evaluation crops (the whole tile)
    <root>/prepared_v2/instances_{val,train}_crop<size>_ignore.json   their ignore regions (the
                                                    buildings without a floor count)

The numbers mean nothing; the files exercise every code path of training and
inference on a CPU in seconds (with INPUT.CROP_SIZE / EVAL_CROP_SIZE = size).
"""
import argparse
import json
import os

import numpy as np
import tifffile

CHANNELS = ["re_%d" % i for i in range(7)] + ["im_%d" % i for i in range(7)] + [
    "coh_mean", "sigma0_vv_db", "amp_dispersion"]


def building(rng, size, k):
    """A rectangular roof, its footprint (the roof moved by the offset), a height."""
    w, h = rng.integers(size // 16, size // 6, 2)
    x0, y0 = rng.integers(8, size - max(w, h) - 24, 2)
    floors = int(rng.integers(1, 30))
    height = 3.0 * floors if k % 7 else 0.0                  # every 7th without a floor count
    dx, dy = float(rng.uniform(-6, 6)), float(rng.uniform(4, 4 + 0.6 * floors))
    roof = [x0, y0, x0 + w, y0, x0 + w, y0 + h, x0, y0 + h]
    foot = [roof[i] + (dx if i % 2 == 0 else dy) for i in range(8)]
    return [float(v) for v in roof], [float(v) for v in foot], height


def write_split(root, split, n, size, rng, ann_id, img_id):
    images, anns = [], []
    for t in range(n):
        name = "%s_%03d.tif" % (split, t)
        rgb = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        tifffile.imwrite(os.path.join(root, "images", split, name), rgb)
        sar = rng.normal(-8.67, 6.79, (size, size)).astype(np.float32)
        sar[:4, :4] = np.nan                                   # some no-data pixels
        os.makedirs(os.path.join(root, "sar_gf3", split), exist_ok=True)
        tifffile.imwrite(os.path.join(root, "sar_gf3", split, name), sar)
        ins = rng.normal(0.0, 1.0, (len(CHANNELS), size // 16, size // 16)).astype(np.float32)
        ins[:, 0, 0] = np.nan
        os.makedirs(os.path.join(root, "s1_insar_asc", split), exist_ok=True)
        tifffile.imwrite(os.path.join(root, "s1_insar_asc", split, name), ins)
        images.append({"id": img_id, "file_name": name, "sar_file_name": name, "height": size, "width": size,
                       "crop_box": [0, 0, size, size], "parent_height": size, "parent_width": size})
        for k in range(int(rng.integers(6, 12))):
            roof, foot, h = building(rng, size, k)
            xs, ys = roof[0::2], roof[1::2]
            anns.append({"id": ann_id, "image_id": img_id, "category_id": 1, "iscrowd": 0,
                         "segmentation": [roof], "offset": [foot], "bd_height": h,
                         "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                         "area": (max(xs) - min(xs)) * (max(ys) - min(ys)), "truncated": False})
            ann_id += 1
        img_id += 1
    return {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "building"}]}, ann_id, img_id


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--train", type=int, default=2)
    ap.add_argument("--val", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    train, ann_id, img_id = write_split(a.root, "train", a.train, a.size, rng, 1, 1)
    val, _, _ = write_split(a.root, "val", a.val, a.size, rng, ann_id, img_id)
    os.makedirs(os.path.join(a.root, "annotations_v2"), exist_ok=True)
    with open(os.path.join(a.root, "annotations_v2", "instances_train.json"), "w") as f:
        json.dump(train, f)
    os.makedirs(os.path.join(a.root, "prepared_v2"), exist_ok=True)
    for split, d in (("val", val), ("train", train)):
        # the evaluation crops leave the buildings without a floor count out; they
        # are ignore regions of the evaluation
        keep = [x for x in d["annotations"] if x["bd_height"] > 0]
        ignore = [x for x in d["annotations"] if x["bd_height"] <= 0]
        base = os.path.join(a.root, "prepared_v2", "instances_%s_crop%d" % (split, a.size))
        with open(base + ".json", "w") as f:
            json.dump(dict(d, annotations=keep), f)
        with open(base + "_ignore.json", "w") as f:
            json.dump({"annotations": ignore}, f)
    stats = {"mean": [0.0] * len(CHANNELS), "std": [1.0] * len(CHANNELS)}
    with open(os.path.join(a.root, "s1_insar_asc", "S1_INSAR_ASC.json"), "w") as f:
        json.dump({"channels": CHANNELS, "normalisation": {"train_orig": stats}}, f)
    print("fake BDH at %s: %d training and %d validation tiles of %d px" % (a.root, a.train, a.val, a.size))


if __name__ == "__main__":
    main()
