# -*- coding: utf-8 -*-
"""Fold a BatchNorm ResNet checkpoint's statistics into a GroupNorm checkpoint.

For WISH-Net's SAR ResNet-18, whose norms are GroupNorm(32).  Loading the BN
checkpoint (R-18_imagenet_sar1_adabn.pkl) into GroupNorm layers keeps the scale
and shift but drops the running statistics, so the initial network is neither
the ImageNet network nor its AdaBN version: GroupNorm pools the mean and
variance of 2 (stem, res2) to 16 (res5) channels, and the channels' different
means survive the normalisation.  For every conv -> norm pair with statistics
(mu_c, v_c) and s_c = sqrt(v_c + eps):

    W_c      <- W_c / s_c         (the conv output then has mean m_c = mu_c / s_c
                                    and unit variance at the statistics)
    gamma'_c  = gamma_c * s_G,    beta'_c = beta_c - gamma_c * (m_c - m_G)
    m_G = mean over the group's channels of m_c,  s_G = sqrt(1 + their variance of m_c)

so GroupNorm(groups, C) gives what BN gives in inference when the input has the
statistics' means and variances (per sample it is close, not exact).  The
running statistics are left out.  The convs have no bias (detectron2's ResNet
puts the norm after a bias-free conv).

    python tools/pretrained/fold_bn_to_gn.py --src pretrained/R-18_imagenet_sar1_adabn.pkl \\
        --out pretrained/R-18_imagenet_sar1_adabn_gn.pkl [--groups 32]
"""
import argparse
import pickle

import numpy as np


def fold(model, groups=32, eps=1e-5):
    out = dict(model)
    n = 0
    for k in sorted(model):
        if not k.endswith(".norm.running_mean"):
            continue
        base = k[: -len(".norm.running_mean")]
        mu = np.asarray(model[base + ".norm.running_mean"], np.float64)
        var = np.asarray(model[base + ".norm.running_var"], np.float64)
        gamma = np.asarray(model[base + ".norm.weight"], np.float64)
        beta = np.asarray(model[base + ".norm.bias"], np.float64)
        w = np.asarray(model[base + ".weight"], np.float64)
        c = mu.shape[0]
        if c % groups:
            raise ValueError("%s: %d channels not divisible into %d groups" % (base, c, groups))
        if base + ".bias" in model:
            raise ValueError("%s has a conv bias; the fold assumes none" % base)
        s = np.sqrt(var + eps)
        m = mu / s
        m_g = np.repeat(m.reshape(groups, -1).mean(1), c // groups)
        s_g = np.repeat(np.sqrt(1.0 + m.reshape(groups, -1).var(1)), c // groups)
        out[base + ".weight"] = (w / s[:, None, None, None]).astype(np.float32)
        out[base + ".norm.weight"] = (gamma * s_g).astype(np.float32)
        out[base + ".norm.bias"] = (beta - gamma * (m - m_g)).astype(np.float32)
        del out[base + ".norm.running_mean"], out[base + ".norm.running_var"]
        out.pop(base + ".norm.num_batches_tracked", None)
        n += 1
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", type=int, default=32)
    a = ap.parse_args()
    with open(a.src, "rb") as f:
        ck = pickle.load(f)
    model, n = fold(ck["model"], a.groups)
    with open(a.out, "wb") as f:
        pickle.dump({"model": model, "__author__": "%s, BN statistics folded into GroupNorm(%d)" % (
            ck.get("__author__", a.src), a.groups), "matching_heuristics": False}, f)
    print("folded %d conv/norm pairs into GroupNorm(%d): %s" % (n, a.groups, a.out))


if __name__ == "__main__":
    main()
