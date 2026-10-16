# -*- coding: utf-8 -*-
"""torchvision ResNet-18 (ImageNet) -> detectron2 weights for WISH-Net's SAR branches.

* keys renamed to detectron2's ResNet (stem.conv1, res2.0.conv1, ...norm, shortcut);
* the three RGB stem kernels are summed onto the one SAR band -- a z-scored
  single band fed to them is then exactly an ImageNet-normalised grey image;
* the classification head (fc) is left out; every other layer is kept.

With --adabn the batch-norm statistics are re-estimated on GF-3 training
windows (the stem's pretrained statistics do not match SAR input): random
1536 px windows of the training tiles, z-scored and area-resampled to 1024 px,
run through the network in training mode with cumulative averaging; only
running_mean / running_var change.  The windows are drawn with a fixed seed
from the sorted list of the folder's tiles, so the folder decides them; the
statistics are accumulated on the GPU when there is one.

    python tools/pretrained/convert_r18_imagenet.py --src resnet18-f37072fd.pth \\
        --out pretrained/R-18_imagenet_sar1_adabn.pkl --adabn <BDH-RGB-SAR>/sar_gf3/train --n 400

The paper's file was made from the training tiles of BDH-RGB-SAR's geographic
split (<BDH-RGB-SAR>/sar_gf3/train, 561 tiles); made again from them it is
bit-identical (datasets/README.md).

The InSAR branch reads only the res2-res4 convolution weights of this file,
which --adabn leaves as they are; the GF-3 branch reads the file with the
statistics folded into its GroupNorms (fold_bn_to_gn.py).
"""
import argparse
import glob
import os
import pickle
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def rename(k):
    if k.startswith("fc.") or k.endswith("num_batches_tracked"):
        return None
    if k.startswith("conv1."):
        return "stem." + k
    if k.startswith("bn1."):
        return "stem.conv1.norm." + k[len("bn1."):]
    parts = k.split(".")                       # layerX.i.name...
    stage = "res%d" % (int(parts[0][len("layer"):]) + 1)
    rest = ".".join(parts[2:])
    rest = (rest.replace("bn1.", "conv1.norm.").replace("bn2.", "conv2.norm.")
            .replace("downsample.0.", "shortcut.").replace("downsample.1.", "shortcut.norm."))
    return "%s.%s.%s" % (stage, parts[1], rest)


def convert(src):
    sd = torch.load(src, map_location="cpu")
    out = {}
    for k, v in sd.items():
        nk = rename(k)
        if nk is None:
            continue
        v = v.detach().float()
        if nk == "stem.conv1.weight":
            v = v.sum(1, keepdim=True)            # (64, 3, 7, 7) -> (64, 1, 7, 7)
        out[nk] = v.numpy()
    return out


def adabn(weights, tile_dir, n, context, input_px, mean, std, seed=0):
    """Re-estimate the batch-norm statistics on GF-3 training windows."""
    import tifffile
    from detectron2.config import get_cfg
    from detectron2.layers import ShapeSpec
    from detectron2.modeling.backbone.resnet import build_resnet_backbone

    cfg = get_cfg()
    cfg.MODEL.RESNETS.DEPTH = 18
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
    cfg.MODEL.RESNETS.NORM = "BN"
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    net = build_resnet_backbone(cfg, ShapeSpec(channels=1))
    missing, unexpected = net.load_state_dict({k: torch.as_tensor(v) for k, v in weights.items()},
                                              strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    assert not missing and not unexpected, (missing, unexpected)
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None                     # cumulative average
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(dev).train()
    tiles = sorted(glob.glob(os.path.join(tile_dir, "*.tif")))
    rng = random.Random(seed)
    with torch.no_grad():
        for i in range(n):
            t = tifffile.imread(rng.choice(tiles)).astype(np.float32)
            if t.ndim == 3:
                t = t[..., 0]
            t = np.where(np.isfinite(t), t, np.nanmean(t))
            y0 = rng.randint(0, t.shape[0] - context)
            x0 = rng.randint(0, t.shape[1] - context)
            x = torch.as_tensor((t[y0:y0 + context, x0:x0 + context] - mean) / std)[None, None].to(dev)
            x = torch.nn.functional.interpolate(x, size=(input_px, input_px), mode="area")
            net(x)
            if (i + 1) % 100 == 0:
                print("adabn %d / %d" % (i + 1, n))
    sd = net.state_dict()
    for k in list(weights):
        if k.endswith("running_mean") or k.endswith("running_var"):
            weights[k] = sd[k].cpu().numpy()
    return weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="torchvision resnet18 state dict (.pth)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adabn", default="", help="folder of GF-3 training tiles; empty = off")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--context", type=int, default=1536)
    ap.add_argument("--input-px", type=int, default=1024)
    ap.add_argument("--mean", type=float, default=-8.67)
    ap.add_argument("--std", type=float, default=6.79)
    args = ap.parse_args()
    w = convert(args.src)
    print("converted %d tensors; stem %s" % (len(w), w["stem.conv1.weight"].shape))
    if args.adabn:
        w = adabn(w, args.adabn, args.n, args.context, args.input_px, args.mean, args.std)
        v = w["stem.conv1.norm.running_var"]
        print("adabn: stem running_var mean %.3f" % float(np.mean(v)))
    with open(args.out, "wb") as f:
        # the tag keeps the network's working name: it is written into the file,
        # and pretrained/md5.txt lists the files as they were made
        pickle.dump({"model": w, "__author__": "torchvision ImageNet ResNet-18 -> GFOPTNet SAR",
                     "matching_heuristics": False}, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
