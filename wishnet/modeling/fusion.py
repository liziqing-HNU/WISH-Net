# -*- coding: utf-8 -*-
"""The SAR reads of WISH-Net and the gated injection of what they read.

``RangeDeformAttn`` is the read built on the imaging geometry.  A side-looking
sensor sees a building's layover towards it and the shadow away from it, both
along the range direction u (RANGE.TOWARD_SENSOR_DEG; with InSAR, its own
direction for the InSAR levels).  The P sampling points of every head start
at P anchors spread along u around the footprint centre (RANGE.ANCHORS_M: the
shadow side, the dihedral line, the layover band); around each anchor the
head samples with learned offsets (in cells of each level, so the coarse
levels search tens of metres) and learned, query-conditioned weights.  The
anchors only say where to start looking: they do not scale with a height, and
no height is read out of positions.  The decoder's SAR step and both
directions of the encoder-stage exchange use it.

``GatedFusion`` writes what was read into a query through a channel gate:

    m = LN(W_m a)
    q <- q + sigmoid(W_g [q ; m] + b_g) * W_v m

with W_g = 0 and b_g = 0 at the start (the gate half open, so neither the gate
nor W_v starts without gradient) and W_v small, so that the injection starts at
about FUSION.INJECT_REL of the query.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from maskdino.modeling.pixel_decoder.ops.functions import MSDeformAttnFunction
from maskdino.modeling.pixel_decoder.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
from maskdino.modeling.pixel_decoder.ops.modules import MSDeformAttn


def sample_map(fmap, pts):
    """Bilinear read of fmap (B, C, h, w) at pts (B, Q, 2) in [0, 1] -> (B, Q, C)."""
    grid = (pts * 2.0 - 1.0)[:, :, None, :].to(fmap.dtype)
    out = F.grid_sample(fmap, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out[..., 0].transpose(1, 2)


class RangeDeformAttn(MSDeformAttn):
    """Multi-scale deformable attention whose sampling points start at anchors
    along the range direction around the footprint centre:

        loc(h, l, p) = fp + t_p u / W_px + delta(h, l, p) / (W_l, H_l)

    t_p the p-th anchor (m, converted to px of the SAR window as it goes in:
    gsd_m is RANGE.GSD_M / INPUT.TILE_SCALE), u the unit vector towards the
    sensor in the image (x right, y down), W_px the SAR window's size in those
    px, delta the learned offset in cells of level l (zero weights; the bias a
    0.5-cell ring whose direction differs per head).  The attention weights
    start uniform over the levels and anchors.

    A training sample flipped or rotated as a whole (INPUT.GEO_AUG) comes with
    the linear part M of its transform, which the decoder puts into ``geo``
    (B, 2, 2) for the forward: the anchors run along M u and the learned
    offsets are taken in the untransformed frame, M delta, so every sample
    shares one pattern of offsets around the range direction.  ``geo`` None:
    exactly the computation above."""

    def __init__(self, d_model, n_levels, n_heads, anchors_m, toward_sensor_deg, gsd_m):
        super().__init__(d_model, n_levels, n_heads, len(anchors_m))
        self.register_buffer("t_px", torch.tensor([float(t) / float(gsd_m) for t in anchors_m]),
                             persistent=False)
        # one direction for every level, or one per level (GF-3 levels along GF-3's
        # direction to the sensor, the InSAR levels along the InSAR sensor's)
        dirs = list(toward_sensor_deg) if isinstance(toward_sensor_deg, (list, tuple)) else [toward_sensor_deg]
        if len(dirs) not in (1, n_levels):
            raise ValueError("RangeDeformAttn: %d directions for %d levels" % (len(dirs), n_levels))
        u = torch.tensor([[math.sin(math.radians(float(a))), -math.cos(math.radians(float(a)))] for a in dirs])
        self.register_buffer("u", u[0] if len(dirs) == 1 else u, persistent=False)
        with torch.no_grad():
            th = torch.arange(n_heads, dtype=torch.float32) * (2.0 * math.pi / n_heads)
            ring = 0.5 * torch.stack([th.cos(), th.sin()], -1)
            self.sampling_offsets.bias.copy_(
                ring[:, None, None, :].expand(n_heads, n_levels, self.n_points, 2).reshape(-1))
        self.record = False          # analysis: keep the last sampling locations and weights
        self.last = None
        self.geo = None              # per forward: (B, 2, 2) the samples' transforms, or None

    def anchor_points(self, fp, win_px):
        """fp (B, Q, 2) footprint centres in [0, 1] of the SAR window; win_px (2,)
        the window's (W, H) in optical px -> anchors (B, Q, P, 2) in that frame, or
        (B, Q, L, P, 2) with a direction per level."""
        if self.geo is not None:
            m = self.geo.to(self.u.dtype)
            if self.u.dim() == 1:
                u = torch.einsum("bij,j->bi", m, self.u)                                    # (B, 2)
                a = self.t_px[None, :, None] * u[:, None, :] / win_px.to(u.dtype)          # (B, P, 2)
                return fp[:, :, None, :] + a[:, None]
            u = torch.einsum("bij,lj->bli", m, self.u)                                      # (B, L, 2)
            a = self.t_px[None, None, :, None] * u[:, :, None, :] / win_px.to(u.dtype)     # (B, L, P, 2)
            return fp[:, :, None, None, :] + a[:, None]
        if self.u.dim() == 1:
            a = self.t_px[:, None] * self.u[None, :] / win_px.to(self.u.dtype)
            return fp[:, :, None, :] + a[None, None]
        a = self.t_px[None, :, None] * self.u[:, None, :] / win_px.to(self.u.dtype)
        return fp[:, :, None, None, :] + a[None, None]

    def sampling(self, query, fp, win_px, shapes):
        """-> locations (B, Q, heads, levels, P, 2) and weights (B, Q, heads, levels, P)."""
        N, Lq, _ = query.shape
        off = self.sampling_offsets(query).view(N, Lq, self.n_heads, self.n_levels, self.n_points, 2)
        if self.geo is not None:
            # the learned offsets live in the untransformed frame
            off = torch.einsum("bij,bqhlpj->bqhlpi", self.geo.to(off.dtype), off)
        w = self.attention_weights(query).view(N, Lq, self.n_heads, self.n_levels * self.n_points)
        w = F.softmax(w, -1).view(N, Lq, self.n_heads, self.n_levels, self.n_points)
        norm = torch.stack([shapes[..., 1], shapes[..., 0]], -1).to(off.dtype)
        anc = self.anchor_points(fp, win_px)
        anc = anc[:, :, None, None] if anc.dim() == 4 else anc[:, :, None]      # (B, Q, 1, 1|L, P, 2)
        loc = anc + off / norm[None, None, None, :, None, :]
        return loc, w

    def forward(self, query, fp, win_px, value_in, shapes, starts):
        N, Lin, _ = value_in.shape
        value = self.value_proj(value_in).view(N, Lin, self.n_heads, self.d_model // self.n_heads)
        loc, w = self.sampling(query, fp, win_px, shapes)
        self.last = (loc.detach(), w.detach()) if self.record else None
        if value.is_cuda:
            # the CUDA op needs contiguous locations and weights; the footprints
            # come from the sequence-first boxes, which for a batch of more than
            # one image makes them a transposed, non-contiguous view
            out = MSDeformAttnFunction.apply(value, shapes, starts, loc.contiguous(), w.contiguous(),
                                             self.im2col_step)
        else:
            out = ms_deform_attn_core_pytorch(value, shapes, loc, w)
        return self.output_proj(out)


class GatedFusion(nn.Module):
    """q <- q + sigmoid(W_g [q; m] + b_g) * W_v m, m = LN(W_m a)."""

    def __init__(self, d_model, gate_bias=0.0, inject_rel=0.1):
        super().__init__()
        self.mix = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(2 * d_model, d_model)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_bias))
        self.value = nn.Linear(d_model, d_model)
        # |W_v m| per element ~ sqrt(d) * std; with the gate at g0 the
        # injection is g0 * sqrt(d) * std = inject_rel of a unit-scale query
        g0 = 1.0 / (1.0 + math.exp(-float(gate_bias)))
        nn.init.normal_(self.value.weight, std=float(inject_rel) / (g0 * math.sqrt(d_model)))
        nn.init.zeros_(self.value.bias)
        self.last = {}

    def forward(self, q, a):
        m = self.norm(self.mix(a))
        g = torch.sigmoid(self.gate(torch.cat([q, m], -1)))
        delta = g * self.value(m)
        with torch.no_grad():
            self.last = {
                "gate": g.mean(),
                "gate_closed": (g < 0.05).float().mean(),
                "inject_rel": (delta.norm(dim=-1) / q.norm(dim=-1).clamp(min=1e-6)).mean(),
            }
        return q + delta
