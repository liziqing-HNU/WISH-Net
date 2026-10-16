# -*- coding: utf-8 -*-
"""MODEL.WISHNET.ENCODER_EXCHANGE: the two-way exchange between the optical and
the SAR memory at the encoder stage.

After the encoders -- the optical pixel decoder's Transformer and the joint
GF-3 / InSAR encoder -- and before the optical FPN and the query selection,
one block exchanges in both directions, both reading the memories as the
encoders left them:

    SAR -> optical   t_o <- t_o + gamma_g * R_g(t_o) + gamma_i * R_i(t_o)
    optical -> SAR   t_g <- t_g + beta_g * R'_g(t_g),   t_i <- t_i + beta_i * R'_i(t_i)

R is the range read (fusion.RangeDeformAttn): from the token's own position,
carried into the other modality's frame, the points of every head and level
start at anchors along the direction towards the sensor (RANGE.ANCHORS_M; the
reverse reads at the mirrored anchors, -t) and move by learned offsets, with
weights from the query.  The query is phi([t; c]) + e_l, CF-Deformable DETR's
paired condition: c the other modality's feature at the same point
(bilinear), e_l an embedding of the token's level.  R_g reads GF-3's s32, s16,
s8 along GF-3's direction, R_i the InSAR levels i32, i16 along the InSAR
sensor's; the reverse reads take the optical s32, s16, s8.  Written are the
optical s32, s16, s8 (all three: the mask features come from s8 alone), GF-3's
s32, s16, s8 and the InSAR levels; GF-3's s4 (its FPN level) is not.

gamma (per channel) starts at 0, so the optical memory, the mask features and
the query selection start exactly as without the block; beta starts at
REVERSE_INIT.  A flipped or rotated sample (INPUT.GEO_AUG) turns the anchors
and the learned offsets with its geo_M, as in the decoder.  In training the
block runs under activation checkpointing (CHECKPOINT).

``optical_transformer`` and ``optical_fpn`` are MaskDINOEncoder.forward_features
cut in two at the FPN, the same operations in the same order, so that the
exchange can sit between them.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .fusion import RangeDeformAttn, sample_map
from .sar_branch import flatten_levels


# ------------------------------------------------- the optical encoder in two
@torch.autocast("cuda", enabled=False)
def optical_transformer(enc, features):
    """MaskDINOEncoder.forward_features up to its FPN (masks None, as the
    meta-architecture calls it) -> the Transformer's levels (B, C, h, w),
    [s32, s16, s8]."""
    srcs, pos, srcsl, posl = [], [], [], []
    if enc.total_num_feature_levels > enc.transformer_num_feature_levels:
        smallest_feat = features[enc.transformer_in_features[enc.low_resolution_index]].float()
        _len_srcs = enc.transformer_num_feature_levels
        for lvl in range(_len_srcs, enc.total_num_feature_levels):
            if lvl == _len_srcs:
                src = enc.input_proj[lvl](smallest_feat)
            else:
                src = enc.input_proj[lvl](srcsl[-1])
            srcsl.append(src)
            posl.append(enc.pe_layer(src))
    srcsl = srcsl[::-1]
    for idx, f in enumerate(enc.transformer_in_features[::-1]):
        x = features[f].float()
        srcs.append(enc.input_proj[idx](x))
        pos.append(enc.pe_layer(x))
    if enc.feature_order == "low2high":
        srcs.extend(srcsl)
        pos.extend(posl)
    else:
        srcsl.extend(srcs)
        posl.extend(pos)
        srcs, pos = srcsl, posl
    y, spatial_shapes, level_start_index = enc.transformer(srcs, None, pos)
    bs = y.shape[0]
    sections = [None] * enc.total_num_feature_levels
    for i in range(enc.total_num_feature_levels):
        if i < enc.total_num_feature_levels - 1:
            sections[i] = level_start_index[i + 1] - level_start_index[i]
        else:
            sections[i] = y.shape[1] - level_start_index[i]
    y = torch.split(y, sections, dim=1)
    return [z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]) for i, z in enumerate(y)]


@torch.autocast("cuda", enabled=False)
def optical_fpn(enc, features, levels):
    """The rest of MaskDINOEncoder.forward_features on the Transformer's levels
    ``levels`` -> (mask features (the stride-4 map), the multi-scale features
    [s32, s16, s8])."""
    out = list(levels)
    for idx, f in enumerate(enc.in_features[:enc.num_fpn_levels][::-1]):
        x = features[f].float()
        cur_fpn = enc.lateral_convs[idx](x)
        y = cur_fpn + F.interpolate(out[enc.high_resolution_index], size=cur_fpn.shape[-2:], mode="bilinear",
                                    align_corners=False)
        out.append(enc.output_convs[idx](y))
    return enc.mask_features(out[-1]), out[:enc.total_num_feature_levels]


# ------------------------------------------------------------------ helpers
def token_points(shapes, device):
    """(L, 2) level shapes -> (S, 2) the cells' centres in [0, 1] (x, y), level by
    level, row by row (the order of flatten_levels), and (S,) their levels."""
    pts, lvl = [], []
    for i, (h, w) in enumerate(shapes.tolist()):
        ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) / h
        xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) / w
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        pts.append(torch.stack([gx.reshape(-1), gy.reshape(-1)], -1))
        lvl.append(torch.full((h * w,), i, dtype=torch.long, device=device))
    return torch.cat(pts, 0), torch.cat(lvl, 0)


def unflatten_levels(x, maps):
    """(B, S, C) -> [(B, C, h, w)] shaped like ``maps``."""
    sizes = [int(m.shape[-2]) * int(m.shape[-1]) for m in maps]
    return [z.transpose(1, 2).reshape(m.shape) for z, m in zip(torch.split(x, sizes, dim=1), maps)]


def paired(maps, ref, shapes):
    """The other modality's feature under every token: ``maps`` [(B, C, h, w)], one
    per query level; ref (B, S, 2) the tokens' points in that modality's frame,
    level by level as ``shapes`` -> (B, S, C)."""
    sizes = [int(h) * int(w) for h, w in shapes.tolist()]
    return torch.cat([sample_map(m, r.contiguous()) for m, r in zip(maps, torch.split(ref, sizes, dim=1))], 1)


class ExchangeRead(nn.Module):
    """One read of the block: query phi([q; c]) + e_l, a RangeDeformAttn into
    the other modality's levels."""

    def __init__(self, d_model, n_levels, n_query_levels, n_heads, anchors_m, toward_sensor_deg, gsd_m):
        super().__init__()
        self.phi = nn.Linear(2 * d_model, d_model)
        nn.init.xavier_uniform_(self.phi.weight)
        nn.init.zeros_(self.phi.bias)
        self.level_embed = nn.Parameter(torch.empty(n_query_levels, d_model))
        nn.init.normal_(self.level_embed)
        self.attn = RangeDeformAttn(d_model, n_levels, n_heads, anchors_m, toward_sensor_deg, gsd_m)

    def forward(self, q, c, lvl, ref, win_px, value, shapes, starts):
        """q, c (B, S, d); lvl (S,) the query levels; ref (B, S, 2) the queries'
        points in the value's frame, win_px that frame's (W, H) in input px;
        value (B, S', d) flattened levels -> (B, S, d)."""
        z = self.phi(torch.cat([q, c], -1)) + self.level_embed[lvl][None]
        return self.attn(z, ref, win_px, value, shapes, starts)


class EncoderExchange(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        g = cfg.MODEL.WISHNET
        e = g.ENCODER_EXCHANGE
        d = int(cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM)
        heads = int(g.FUSION.N_HEADS)
        r = g.RANGE
        # the anchors in px of the input, as the decoder's (INPUT.TILE_SCALE)
        scale = float(cfg.INPUT.TILE_SCALE) if cfg.INPUT.TILE_SCALE > 0 else 1.0
        gsd = float(r.GSD_M) / scale
        anchors = [float(t) for t in r.ANCHORS_M]
        mirror = [-t for t in anchors]
        to_gf3 = float(r.TOWARD_SENSOR_DEG)
        self.opt_from_gf3 = ExchangeRead(d, 3, 3, heads, anchors, to_gf3, gsd)
        self.gf3_from_opt = ExchangeRead(d, 3, 3, heads, mirror, to_gf3, gsd)
        self.gamma_gf3 = nn.Parameter(torch.zeros(d))
        self.beta_gf3 = nn.Parameter(torch.full((d,), float(e.REVERSE_INIT)))
        self.insar = bool(g.INSAR.ENABLED)
        if self.insar:
            to_ins = float(g.INSAR.TOWARD_SENSOR_DEG)
            self.opt_from_ins = ExchangeRead(d, 2, 3, heads, anchors, to_ins, gsd)
            self.ins_from_opt = ExchangeRead(d, 3, 2, heads, mirror, to_ins, gsd)
            self.gamma_ins = nn.Parameter(torch.zeros(d))
            self.beta_ins = nn.Parameter(torch.full((d,), float(e.REVERSE_INIT)))
        self.use_checkpoint = bool(e.CHECKPOINT)
        self.stats = {}
        self._record = False

    def reads(self):
        return [m for m in (self.opt_from_gf3, self.gf3_from_opt, getattr(self, "opt_from_ins", None),
                            getattr(self, "ins_from_opt", None)) if m is not None]

    def forward(self, opt_maps, gf3_maps, ins_maps, img_size, sar_size, sar_off, geo=None):
        """opt_maps [s32, s16, s8] (B, d, h, w) of the optical Transformer;
        gf3_maps [s32, s16, s8] of GF-3; ins_maps [i32, i16] or None; img_size
        (H, W) of the optical input and sar_size (h, w) of the SAR window, in
        input px; sar_off (B, 2) the optical window's origin in the SAR window;
        geo (B, 2, 2) the samples' flips / rotations or None -> (optical maps,
        GF-3 maps, InSAR maps or None) after the exchange.  The statistics of
        this call go to ``stats``."""
        args = (list(opt_maps), list(gf3_maps), list(ins_maps) if ins_maps is not None else None,
                tuple(int(v) for v in img_size), tuple(int(v) for v in sar_size), sar_off, geo)
        self._record = True
        try:
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                return checkpoint(self._run, *args, use_reentrant=False)
            return self._run(*args)
        finally:
            self._record = False

    def _run(self, opt_maps, gf3_maps, ins_maps, img_size, sar_size, sar_off, geo):
        for m in self.reads():
            m.attn.geo = geo
        try:
            return self._exchange(opt_maps, gf3_maps, ins_maps, img_size, sar_size, sar_off)
        finally:
            for m in self.reads():
                m.attn.geo = None

    def _exchange(self, opt_maps, gf3_maps, ins_maps, img_size, sar_size, sar_off):
        H, W = img_size
        sh, sw = sar_size
        q_o, shp_o, st_o = flatten_levels(opt_maps)
        v_g, shp_g, st_g = flatten_levels(gf3_maps[:3])          # GF-3's s32, s16, s8: the forward reads' value
        dev = q_o.device
        size_o = q_o.new_tensor([float(W), float(H)])
        size_s = q_o.new_tensor([float(sw), float(sh)])
        off = sar_off.to(q_o.dtype)[:, None, :]
        if self.insar:
            v_i, shp_i, st_i = flatten_levels(ins_maps)
        # SAR -> optical: the optical tokens' points in the SAR window
        p_o, lvl_o = token_points(shp_o, dev)
        r_o = (p_o[None] * size_o + off) / size_s
        a_g = self.opt_from_gf3(q_o, paired(gf3_maps, r_o, shp_o), lvl_o, r_o, size_s, v_g, shp_g, st_g)
        d_o = self.gamma_gf3 * a_g
        a_i = None
        if self.insar:
            c_i = paired([ins_maps[0]] + [ins_maps[1]] * (len(opt_maps) - 1), r_o, shp_o)
            a_i = self.opt_from_ins(q_o, c_i, lvl_o, r_o, size_s, v_i, shp_i, st_i)
            d_o = d_o + self.gamma_ins * a_i
        o_new = q_o + d_o
        # optical -> SAR: the SAR tokens' points in the optical input, the optical
        # memory before the exchange as the value
        p_g, lvl_g = token_points(shp_g, dev)
        r_g = (p_g[None] * size_s - off) / size_o
        b_g = self.gf3_from_opt(v_g, paired(opt_maps, r_g, shp_g), lvl_g, r_g, size_o, q_o, shp_o, st_o)
        g_new = v_g + self.beta_gf3 * b_g
        i_new = b_i = None
        if self.insar:
            p_i, lvl_i = token_points(shp_i, dev)
            r_i = (p_i[None] * size_s - off) / size_o
            b_i = self.ins_from_opt(v_i, paired(opt_maps[:2], r_i, shp_i), lvl_i, r_i, size_o, q_o, shp_o, st_o)
            i_new = v_i + self.beta_ins * b_i
        if self._record:
            self._stats(q_o, a_g, a_i, v_g, b_g, v_i if self.insar else None, b_i)
        return (unflatten_levels(o_new, opt_maps), unflatten_levels(g_new, gf3_maps),
                unflatten_levels(i_new, ins_maps) if i_new is not None else None)

    @torch.no_grad()
    def _stats(self, q_o, a_g, a_i, v_g, b_g, v_i, b_i):
        """exchange/*: the coefficients' root mean squares and the written
        updates' size relative to the token they go into (mean over tokens)."""
        def rms(p):
            return p.detach().float().pow(2).mean().sqrt()

        def rel(delta, t):
            return (delta.float().norm(dim=-1) / t.float().norm(dim=-1).clamp(min=1e-6)).mean()

        s = {"beta_gf3": rms(self.beta_gf3), "rev_rel_gf3": rel(self.beta_gf3 * b_g, v_g),
             "gamma_gf3": rms(self.gamma_gf3), "inject_rel_gf3": rel(self.gamma_gf3 * a_g, q_o)}
        if b_i is not None:
            s.update({"beta_ins": rms(self.beta_ins), "rev_rel_ins": rel(self.beta_ins * b_i, v_i)})
        if a_i is not None:
            s.update({"gamma_ins": rms(self.gamma_ins), "inject_rel_ins": rel(self.gamma_ins * a_i, q_o)})
        self.stats = s
