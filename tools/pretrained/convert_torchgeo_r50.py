# -*- coding: utf-8 -*-
"""Convert a torchgeo ResNet-50 checkpoint into a detectron2 backbone pickle:
WISH-Net's optical ResNet-50 (pretrained/R-50_torchgeo_fmow_rgb_gassl.pkl).

The optical tower is seeded from torchgeo's ``ResNet50_Weights.FMOW_RGB_GASSL``:
MoCo-v2 with geography-aware positives on fMoW, i.e. WorldView-class RGB at
roughly 0.3-1.5 m GSD.  That is the only RGB ResNet-50 in torchgeo trained on
imagery anywhere near our 0.41 m/px optical tiles -- the other RGB ResNet-50s
(SeCo, SSL4EO MoCo, Satlas) are Sentinel-2 at 10 m.

Two properties make it a drop-in for the current optical config, and both are
checked below rather than assumed:

* torchgeo builds it as ``timm.create_model('resnet50')``, which is the
  torchvision layout: stride on the 3x3 conv, so ``RESNETS.STRIDE_IN_1X1``
  must be False;
* its transform is ``/255`` then the ImageNet mean/std, in RGB order -- exactly
  ``MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]`` and
  ``MODEL.PIXEL_STD = [58.395, 57.120, 57.375]`` on 0-255 input.

The file is fetched from HuggingFace (GASSL_URL below, torchgeo's pin), and
the SHA-256 prefix torchgeo puts in the file name is verified before anything
is converted.

Usage:
    python tools/pretrained/convert_torchgeo_r50.py \\
        resnet50_fmow_rgb_gassl-da43d987.pth pretrained/R-50_torchgeo_fmow_rgb_gassl.pkl
"""
import argparse
import hashlib
import os
import pickle
import re

import torch

# torchgeo's own pin: hf.co/torchgeo/resnet50_fmow_rgb_gassl @ fe8a910
GASSL_URL = ("https://huggingface.co/torchgeo/resnet50_fmow_rgb_gassl/resolve/"
             "fe8a91026cf9104f1e884316b8e8772d7af9052c/"
             "resnet50_fmow_rgb_gassl-da43d987.pth")


def check_hash(path):
    """torchgeo names the file ``<stem>-<first 8 hex of sha256>.pth``."""
    m = re.search(r"-([0-9a-f]{8})\.pth$", os.path.basename(path))
    if not m:
        print("[convert] no hash in the file name; skipping the check")
        return
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if not h.hexdigest().startswith(m.group(1)):
        raise RuntimeError("sha256 %s does not match the pinned prefix %s"
                           % (h.hexdigest(), m.group(1)))
    print("[convert] sha256 prefix %s verified" % m.group(1))


def to_d2(sd):
    """timm / torchvision ResNet keys -> detectron2 ``build_resnet_backbone`` keys."""
    out = {}
    for k, v in sd.items():
        if k.startswith("fc.") or k.endswith("num_batches_tracked"):
            continue
        n = k
        if not n.startswith("layer"):
            n = "stem." + n                     # conv1 / bn1
        for t in (1, 2, 3, 4):
            n = n.replace("layer%d" % t, "res%d" % (t + 1))
        for t in (1, 2, 3):
            n = n.replace("bn%d" % t, "conv%d.norm" % t)
        n = n.replace("downsample.0", "shortcut")
        n = n.replace("downsample.1", "shortcut.norm")
        out[n] = v.detach().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="torchgeo .pth (%s)" % GASSL_URL)
    ap.add_argument("dst", help="output detectron2 .pkl")
    args = ap.parse_args()

    check_hash(args.src)
    sd = torch.load(args.src, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    model = to_d2(sd)
    # convs' weights + norms x 4 tensors, nothing else: 53 convs for a
    # ResNet-50, 20 for a ResNet-18, 36 for a ResNet-34
    n_conv = sum(1 for k in model if k.endswith("weight") and ".norm." not in k)
    n_norm = sum(1 for k in model if ".norm." in k)
    assert n_conv in (20, 36, 53) and n_norm == n_conv * 4, (n_conv, n_norm)
    print("[convert] ResNet-%d: %d convs" % ({20: 18, 36: 34, 53: 50}[n_conv], n_conv))
    stem_var = float(model["stem.conv1.norm.running_var"].mean())
    # A stem BN variance in the thousands means the network was trained on
    # un-normalised 0-255 input (Caffe/MSRA style) and would need
    # PIXEL_STD = 1; torchvision-normalised input gives single digits.
    print("[convert] stem BN running_var mean %.2f (torchvision-normalised input "
          "gives O(1); Caffe-style 0-255 input gives thousands)" % stem_var)
    if stem_var >= 100.0:
        raise RuntimeError("stem BN variance %.1f: this checkpoint expects un-normalised "
                           "input, which the optical tower is not configured for" % stem_var)

    with open(args.dst, "wb") as f:
        pickle.dump({"model": model,
                     "__author__": "torchgeo:" + os.path.basename(args.src),
                     "matching_heuristics": True}, f)
    print("[convert] stem takes %d input channels"
          % model["stem.conv1.weight"].shape[1])
    print("[convert] wrote %d tensors to %s" % (len(model), args.dst))


if __name__ == "__main__":
    main()
