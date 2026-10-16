# -*- coding: utf-8 -*-
"""The SAR branch: GF-3 sigma0 (and the InSAR tile) -> ResNet-18 -> one SAR
Transformer encoder.

* Input: the GF-3 HH single-look sigma0 in dB over the tile, z-scored with the
  training-split statistics (SAR.MEAN, SAR.STD), area-resampled to
  SAR.INPUT_PX.  No random shift: the misregistration already in the data is
  what the learned sampling has to absorb.
* Backbone: detectron2's ResNet with basic blocks (depth 18) and GroupNorm(32),
  ImageNet weights whose RGB stem kernels are summed onto the one band and
  whose batch-norm statistics, re-estimated on GF-3 training windows, are
  folded into the convolutions and the GroupNorm scale and shift
  (tools/sar_backbone/); every stage res2-res5 is used.
* InSAR (MODEL.WISHNET.INSAR): the channels of the InSAR tile, each standardised
  with its training statistics from the product's metadata, no data -> 0 (the
  training mean); its own ResNet-18 res2-res4 behind a stride-1 stem (outputs
  at 13 m and 26 m for the 6.63 m Sentinel-1 tile), the two levels joined to
  the GF-3 levels in the same encoder.
* Encoder: a second instance of MaskDINO's multi-scale deformable encoder
  (``MaskDINOEncoder``), six layers over res3-res5 (and the InSAR levels), made
  pre-norm with a final LayerNorm (``to_prenorm``); res2 joins through its FPN
  lateral into the stride-4 map.

The backbone stages and the encoder layers run under activation checkpointing
in training.  Output: the SAR memory the decoder samples -- levels [s32, s16,
s8, s4] (+ [i32, i16]) -- flattened as MSDeformAttn wants it.
"""
import copy
import json
import logging
import os
import pickle
import re

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling.backbone.resnet import build_resnet_backbone

from maskdino.modeling.pixel_decoder.maskdino_encoder import MaskDINOEncoder, MSDeformAttnTransformerEncoderOnly

logger = logging.getLogger(__name__)


def build_sar_backbone(cfg):
    """ResNet-18 with GroupNorm(32) on one input band, all stages out, nothing
    frozen."""
    s = cfg.MODEL.WISHNET.SAR
    c = copy.deepcopy(cfg)
    c.defrost()
    c.MODEL.RESNETS.DEPTH = int(s.DEPTH)
    c.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
    c.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
    c.MODEL.RESNETS.NUM_GROUPS = 1
    c.MODEL.RESNETS.WIDTH_PER_GROUP = 64
    c.MODEL.RESNETS.RES5_DILATION = 1
    c.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
    c.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    # detectron2's GroupNorm(32, C): the same normalisation in training and
    # inference whatever the batch
    c.MODEL.RESNETS.NORM = "GN"
    c.MODEL.BACKBONE.FREEZE_AT = 0
    c.freeze()
    backbone = build_resnet_backbone(c, ShapeSpec(channels=1))
    if s.WEIGHTS:
        DetectionCheckpointer(backbone).load(s.WEIGHTS)
        logger.info("[WISH-Net] SAR ResNet-%d weights from %s", s.DEPTH, s.WEIGHTS)
    else:
        logger.warning("[WISH-Net] SAR ResNet-%d has NO pretrained weights", s.DEPTH)
    return backbone


class _StemStride1(nn.Module):
    """3x3 convolution, GroupNorm, ReLU, no downsampling: the InSAR tile is
    already at 6.6 m, near the resolution of the product."""

    def __init__(self, in_ch, out_ch=64):
        super().__init__()
        self.conv1 = Conv2d(in_ch, out_ch, 3, padding=1, bias=False, norm=nn.GroupNorm(32, out_ch))
        weight_init.c2_msra_fill(self.conv1)
        self.out_channels, self.stride = out_ch, 1

    def forward(self, x):
        return F.relu_(self.conv1(x))


def build_insar_backbone(cfg, in_ch):
    """ResNet-18 res2-res4 on the InSAR channels behind a stride-1 stem: res3 at
    stride 2 (13 m) and res4 at stride 4 (26 m) out, GroupNorm throughout.  Only the
    ImageNet convolution weights of res2-res4 are loaded: the stem is new and the
    GroupNorms start at scale 1 and shift 0, since no ImageNet (or GF-3) statistics
    describe these channels."""
    s = cfg.MODEL.WISHNET.INSAR
    c = copy.deepcopy(cfg)
    c.defrost()
    c.MODEL.RESNETS.DEPTH = 18
    c.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
    c.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
    c.MODEL.RESNETS.NUM_GROUPS = 1
    c.MODEL.RESNETS.WIDTH_PER_GROUP = 64
    c.MODEL.RESNETS.RES5_DILATION = 1
    c.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
    c.MODEL.RESNETS.OUT_FEATURES = ["res3", "res4"]
    c.MODEL.RESNETS.NORM = "GN"
    c.MODEL.BACKBONE.FREEZE_AT = 0
    c.freeze()
    bb = build_resnet_backbone(c, ShapeSpec(channels=in_ch))
    bb.stem = _StemStride1(in_ch)
    if s.WEIGHTS:
        with open(s.WEIGHTS, "rb") as f:
            ck = pickle.load(f, encoding="latin1")
        src = ck.get("model", ck)
        sd = bb.state_dict()
        take = {k: torch.as_tensor(np.asarray(v)) for k, v in src.items()
                if re.match(r"res[234]\.", k) and k in sd and np.asarray(v).ndim == 4
                and tuple(np.asarray(v).shape) == tuple(sd[k].shape)}
        bb.load_state_dict(take, strict=False)
        logger.info("[WISH-Net] InSAR ResNet-18: %d res2-res4 convolution weights from %s", len(take), s.WEIGHTS)
    else:
        logger.warning("[WISH-Net] InSAR ResNet-18 has NO pretrained weights")
    return bb


def joint_forward_features(enc, feats, extra):
    """MaskDINOEncoder.forward_features with extra levels: the GF-3 levels (res5,
    res4, res3 projected) and the projected InSAR levels ``extra`` go through one
    multi-scale deformable encoder whose sampling crosses all of them; the FPN to
    stride 4 is built from the GF-3 levels only, as in the original.
    -> (mask features (stride-4 map), [s32, s16, s8], [extra levels])."""
    srcs, pos = [], []
    for idx, f in enumerate(enc.transformer_in_features[::-1]):
        x = feats[f].float()
        srcs.append(enc.input_proj[idx](x))
        pos.append(enc.pe_layer(x))
    for s in extra:
        srcs.append(s)
        pos.append(enc.pe_layer(s))
    y, shapes, starts = enc.transformer(srcs, None, pos)
    bs = y.shape[0]
    sizes = [int(h) * int(w) for h, w in shapes]
    out = [z.transpose(1, 2).view(bs, -1, int(h), int(w)) for z, (h, w) in zip(torch.split(y, sizes, dim=1), shapes)]
    n = len(enc.transformer_in_features)
    gmaps, xmaps = out[:n], out[n:]
    for idx, f in enumerate(enc.in_features[:enc.num_fpn_levels][::-1]):
        x = feats[f].float()
        cur = enc.lateral_convs[idx](x)
        z = cur + F.interpolate(gmaps[enc.high_resolution_index], size=cur.shape[-2:], mode="bilinear",
                                align_corners=False)
        gmaps.append(enc.output_convs[idx](z))
    return enc.mask_features(gmaps[-1]), gmaps[:n], xmaps


def to_prenorm(encoder):
    """Make every layer of a ``MaskDINOEncoder`` pre-norm and normalise its output,
    so the SAR image travels along the residual stream instead of being
    renormalised after every update:

        x <- x + A(LN1(x) + pos, value LN1(x)),   x <- x + F(LN2(x)),   out = LN_f(x)

    The layers keep their modules and parameter names (MaskDINO's layers are
    post-norm, m = LN1(x + A(x)), y = LN2(m + F(m))).  A pre-norm stack's
    residual stream is not normalised, so a final LayerNorm is added as
    ``transformer.encoder.final_norm``; it normalises each position on its
    own and brings the memory to the scale its readers expect."""
    stack = encoder.transformer.encoder
    for layer in stack.layers:
        def forward(src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None,
                    L=layer):
            s = L.norm1(src)
            src = src + L.dropout1(L.self_attn(L.with_pos_embed(s, pos), reference_points, s,
                                               spatial_shapes, level_start_index, padding_mask))
            s = L.norm2(src)
            return src + L.dropout3(L.linear2(L.dropout2(L.activation(L.linear1(s)))))
        layer.forward = forward
    stack.final_norm = nn.LayerNorm(stack.layers[0].norm1.normalized_shape[0])
    run_layers = stack.forward

    def forward(*args, **kwargs):
        return stack.final_norm(run_layers(*args, **kwargs))

    stack.forward = forward
    return encoder


def checkpointed(module):
    """Run ``module`` under activation checkpointing while training with
    gradients: its activations are recomputed in the backward pass instead of
    kept.  The module and its parameter names stay as they are."""
    fwd = module.forward

    def run(*args, **kwargs):
        if module.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(fwd, *args, use_reentrant=False, **kwargs)
        return fwd(*args, **kwargs)

    module.forward = run
    return module


def flatten_levels(maps):
    """[(B, C, h, w), ...] -> value (B, sum hw, C), shapes (L, 2), starts (L,)."""
    value = torch.cat([m.flatten(2).transpose(1, 2) for m in maps], 1)
    shapes = torch.as_tensor([m.shape[-2:] for m in maps], dtype=torch.long, device=value.device)
    starts = torch.cat([shapes.new_zeros(1), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]])
    return value, shapes, starts


class SARBranch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        s = cfg.MODEL.WISHNET.SAR
        self.mean, self.std = float(s.MEAN), float(s.STD)
        self.input_px = int(s.INPUT_PX)
        # SAR.ZERO (the zeroing evaluations): the standardised GF-3 input replaced by zeros
        self.zero = bool(s.ZERO)
        self.backbone = build_sar_backbone(cfg)
        shapes = self.backbone.output_shape()
        sh = cfg.MODEL.SEM_SEG_HEAD
        self.encoder = MaskDINOEncoder(
            {k: shapes[k] for k in ("res2", "res3", "res4", "res5")},
            transformer_dropout=cfg.MODEL.MaskDINO.DROPOUT,
            transformer_nheads=cfg.MODEL.MaskDINO.NHEADS,
            transformer_dim_feedforward=int(s.ENC_DIM_FEEDFORWARD),
            transformer_enc_layers=int(s.ENC_LAYERS),
            conv_dim=sh.CONVS_DIM,
            mask_dim=sh.MASK_DIM,
            norm=sh.NORM,
            transformer_in_features=["res3", "res4", "res5"],
            common_stride=4,
            num_feature_levels=3,
            total_num_feature_levels=3,
            feature_order=sh.FEATURE_ORDER,
        )
        self.level_names = ["s32", "s16", "s8", "s4"]
        ins = cfg.MODEL.WISHNET.INSAR
        self.insar = None
        if ins.ENABLED:
            # the InSAR tile: its own ResNet-18 (res2-res4), its two levels appended
            # to the GF-3 levels of one encoder (5 levels instead of 3)
            root = ins.ROOT or os.path.join(cfg.DATASETS.BDH_ROOT, "s1_insar_asc")
            with open(os.path.join(root, ins.META)) as f:
                meta = json.load(f)
            names = list(meta["channels"])
            idx = [names.index(c) for c in (list(ins.CHANNELS) or names)]
            nk = meta["normalisation"][ins.NORM_KEY]
            self.register_buffer("ins_idx", torch.tensor(idx), persistent=False)
            self.register_buffer("ins_mean", torch.tensor([nk["mean"][i] for i in idx], dtype=torch.float32))
            self.register_buffer("ins_std", torch.tensor([max(nk["std"][i], 1e-6) for i in idx],
                                                         dtype=torch.float32))
            # INSAR.ZERO (the zeroing evaluations): the standardised InSAR input replaced by zeros
            self.ins_px, self.ins_zero = int(ins.INPUT_PX), bool(ins.ZERO)
            self.insar = build_insar_backbone(cfg, len(idx))
            ish = self.insar.output_shape()
            d = sh.CONVS_DIM
            self.insar_proj = nn.ModuleList(
                [nn.Sequential(nn.Conv2d(ish[k].channels, d, 1), nn.GroupNorm(32, d)) for k in ("res4", "res3")])
            for p in self.insar_proj:
                nn.init.xavier_uniform_(p[0].weight, gain=1)
                nn.init.constant_(p[0].bias, 0)
            self.encoder.transformer = MSDeformAttnTransformerEncoderOnly(
                d_model=d, dropout=cfg.MODEL.MaskDINO.DROPOUT, nhead=cfg.MODEL.MaskDINO.NHEADS,
                dim_feedforward=int(s.ENC_DIM_FEEDFORWARD), num_encoder_layers=int(s.ENC_LAYERS),
                num_feature_levels=len(self.encoder.transformer_in_features) + 2)
            self.level_names += ["i32", "i16"]
            logger.info("[WISH-Net] InSAR: %d channels from %s (%s, %s statistics), %d px, joint encoder of %d levels",
                        len(idx), root, ins.META, ins.NORM_KEY, self.ins_px,
                        len(self.encoder.transformer_in_features) + 2)
            if ins.ZERO:
                logger.info("[WISH-Net] INSAR.ZERO: the InSAR input is replaced by zeros")
        to_prenorm(self.encoder)            # before the checkpointing wraps the layers
        # memory: the SAR encoder holds as many tokens as the optical one (21.5 k
        # at a 1024 input); its layers and the backbone stages recompute their
        # activations in the backward pass
        for stage in (self.backbone.stem, self.backbone.res2, self.backbone.res3,
                      self.backbone.res4, self.backbone.res5):
            checkpointed(stage)
        for layer in self.encoder.transformer.encoder.layers:
            checkpointed(layer)
        if s.ZERO:
            logger.info("[WISH-Net] SAR.ZERO: the GF-3 input is replaced by zeros")

    def prepare(self, sar_list, device):
        """[(1, S, S) dB tensors] -> standardised, resampled (B, 1, INPUT_PX, INPUT_PX)."""
        x = torch.stack([(t.to(device).float() - self.mean) / self.std for t in sar_list])
        if x.shape[-1] != self.input_px or x.shape[-2] != self.input_px:
            x = F.interpolate(x, size=(self.input_px, self.input_px), mode="area")
        if self.zero:
            x = torch.zeros_like(x)
        return x

    def prepare_insar(self, insar_list, device):
        """[(C, h, w) InSAR tiles, NaN = no data] -> the used channels standardised
        with the training statistics, no data -> 0, resampled: (B, C', INPUT_PX, INPUT_PX)."""
        x = torch.stack([t.to(device).float() for t in insar_list])[:, self.ins_idx]
        x = (x - self.ins_mean[None, :, None, None]) / self.ins_std[None, :, None, None]
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.shape[-1] != self.ins_px or x.shape[-2] != self.ins_px:
            x = F.interpolate(x, size=(self.ins_px, self.ins_px), mode="area")
        if self.ins_zero:
            x = torch.zeros_like(x)
        return x

    def forward(self, x, xi=None):
        """x (B, 1, INPUT_PX, INPUT_PX), with INSAR xi (B, C', P, P) -> the SAR memory the
        decoder reads: levels [s32, s16, s8, s4] (+ the InSAR levels i32, i16)."""
        feats = self.backbone(x)
        if self.insar is None:
            s4, _, ms = self.encoder.forward_features(feats, None)  # ms: [s32, s16, s8]
            extra = []
        else:
            fi = self.insar(xi)
            s4, ms, extra = joint_forward_features(
                self.encoder, feats, [self.insar_proj[0](fi["res4"]), self.insar_proj[1](fi["res3"])])
        maps = list(ms) + [s4] + list(extra)
        value, shapes, starts = flatten_levels(maps)
        return {"maps": maps, "value": value, "shapes": shapes, "starts": starts}
