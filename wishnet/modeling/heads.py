# -*- coding: utf-8 -*-
"""Per-query height and offset heads, and the height losses that read them.

* ``BinnedHeightHead``: floor-aligned bins (3 m wide, centred on 3, 6, ...,
  180 m, since the heights are labelled as floor counts x 3 m), read out as
  the interpolated median of the bin distribution plus a bounded in-bin
  residual;
* ``OffsetHead``: the roof -> footprint translation in pixels;
* ``dc_loss``: HTC-DC Net's distribution constraint on the bins;
* ``soft_bin_target``: SORD's soft ordinal target for the bin cross-entropy
  (MODEL.WISHNET.HEIGHT.BIN_CE_SOFT_SIGMA_M).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from maskdino.utils.utils import MLP


def make_bin_edges(lo, hi, step):
    """Floor-aligned bin centres (N,) and edges (N + 1,)."""
    n = int(round((hi - lo) / step)) + 1
    centres = lo + step * torch.arange(n, dtype=torch.float32)
    edges = torch.empty(n + 1, dtype=torch.float32)
    edges[0] = lo - step / 2.0
    edges[1:] = centres + step / 2.0
    return centres, edges


def mask_boxes_vectorised(masks):
    """Tight xyxy boxes of (N, H, W) bool masks on their own device.

    Same numbers as ``BitMasks.get_bounding_boxes()`` ([x0, y0, x1 + 1, y1 + 1],
    zeros for an empty mask) without its per-instance Python loop.
    """
    n, h, w = masks.shape
    x_any = masks.any(dim=1)
    y_any = masks.any(dim=2)
    idx_x = torch.arange(w, device=masks.device)
    idx_y = torch.arange(h, device=masks.device)
    big = torch.iinfo(torch.int32).max
    x0 = torch.where(x_any, idx_x, big).min(dim=1).values
    x1 = torch.where(x_any, idx_x, -1).max(dim=1).values + 1
    y0 = torch.where(y_any, idx_y, big).min(dim=1).values
    y1 = torch.where(y_any, idx_y, -1).max(dim=1).values + 1
    out = torch.stack([x0, y0, x1, y1], dim=1).float()
    out[~(x_any.any(dim=1) & y_any.any(dim=1))] = 0.0
    return out


class TaskBranch(nn.Module):
    """LayerNorm -> two-layer MLP -> linear predictor; one per task, nothing shared."""

    def __init__(self, hidden_dim, out_dim, trunk_layers=2):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.trunk = MLP(hidden_dim, hidden_dim, hidden_dim, trunk_layers)
        self.pred = nn.Linear(hidden_dim, out_dim)

    def forward(self, q):
        return self.pred(self.trunk(self.norm(q)))


class OffsetHead(nn.Module):
    """branch(q) * scale -> (dx, dy) from roof to footprint, px; starts at 0."""

    def __init__(self, hidden_dim, scale=50.0):
        super().__init__()
        self.branch = TaskBranch(hidden_dim, 2)
        self.scale = float(scale)
        nn.init.constant_(self.branch.pred.weight, 0.0)
        nn.init.constant_(self.branch.pred.bias, 0.0)

    def forward(self, q):
        return self.branch(q) * self.scale


class BinnedHeightHead(nn.Module):
    """Classify the query into floor bins and read one height (m) out of them:
    the interpolated median of the bin distribution plus a bounded in-bin
    residual."""

    # floor on the crossing bin's probability in the median interpolation
    MEDIAN_MIN_P = 1e-3

    def __init__(self, hidden_dim, lo=3.0, hi=180.0, step=3.0, residual_weight=1.0, height_init=25.0):
        super().__init__()
        centres, edges = make_bin_edges(lo, hi, step)
        self.register_buffer("centres", centres, persistent=False)
        self.register_buffer("edges", edges, persistent=False)
        self.n_bins = centres.numel()
        self.residual_weight = float(residual_weight)
        extra = 1 if self.residual_weight > 0 else 0
        self.branch = TaskBranch(hidden_dim, self.n_bins + extra)
        # every query starts at height_init: zero weights, and as bias the log
        # of a Gaussian over the bin centres (sigma = height_init / 2, at least 3 m)
        nn.init.constant_(self.branch.pred.weight, 0.0)
        sigma = max(float(height_init) * 0.5, 3.0)
        bias = self.branch.pred.bias.data
        bias.zero_()
        bias[:self.n_bins] = -0.5 * ((centres - float(height_init)) / sigma) ** 2

    def median(self, p):
        """Interpolated median of (..., N) bin probabilities, in metres."""
        cdf = p.cumsum(-1)
        k = (cdf < 0.5).sum(-1, keepdim=True).clamp_(max=self.n_bins - 1)
        p_k = p.gather(-1, k)
        below = cdf.gather(-1, k) - p_k
        frac = ((0.5 - below) / p_k.clamp(min=self.MEDIAN_MIN_P)).clamp(0.0, 1.0)
        lo = self.edges[:-1][k]
        width = (self.edges[1:] - self.edges[:-1])[k]
        return (lo + frac * width).squeeze(-1)

    def forward(self, q):
        """q (..., C) -> metres (...), log-probabilities (..., N)."""
        out = self.branch(q).float()
        logp = F.log_softmax(out[..., :self.n_bins], dim=-1)
        p = logp.exp()
        h = self.median(p)
        if self.residual_weight > 0:
            half = 0.5 * float(self.centres[1] - self.centres[0])
            h = h + torch.tanh(out[..., self.n_bins]) * half * self.residual_weight
        return h, logp


def soft_bin_target(edges, gt, sigma, fallback):
    """SORD's soft ordinal target (Diaz & Marathe, CVPR 2019) for the bin
    cross-entropy: a Gaussian of width ``sigma`` (m) centred on each
    ground-truth height, integrated over the bins and renormalised over them --
    the mass beyond the first and the last edge is dropped.  A height with
    (almost) no mass inside the bins keeps the one-hot target of the bin
    ``fallback`` (the clamped bin index).
    edges (n + 1,), gt (N,), fallback (N,) -> (N, n)."""
    edges = edges.float().to(gt.device)
    cdf = 0.5 * (1.0 + torch.erf((edges[None, :] - gt.float()[:, None]) / (float(sigma) * math.sqrt(2.0))))
    q = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(0.0)
    mass = q.sum(-1, keepdim=True)
    onehot = F.one_hot(fallback, q.shape[-1]).to(q.dtype)
    return torch.where(mass > 1e-6, q / mass.clamp_min(1e-12), onehot)


def dc_loss(edges, gt, logp, min_sigma_frac=0.5, weight=None, per_building=False):
    """HTC-DC Net's distribution constraint, KL(reference || predicted).

    The reference is a Gaussian centred on the ground truth whose sigma is
    solved from the probability the model already gives the ground-truth bin,
    so it enforces unimodality and symmetry rather than confidence.
    ``weight`` (N,): every building's divergence times its weight, still
    divided by N (MODEL.WISHNET.HEIGHT.BIN_WEIGHTS); ``per_building``: the
    divergences (N,) themselves.
    """
    if gt.numel() == 0:
        return logp.sum() * 0.0
    logp = logp.float()
    with torch.no_grad():
        gt = gt.float()
        edges = edges.float().to(logp.device)
        n = logp.shape[-1]
        lo, hi = edges[:-1], edges[1:]
        idx = (torch.bucketize(gt, edges) - 1).clamp_(0, n - 1)
        width = (hi - lo)[idx]
        p_m = logp.exp().gather(1, idx[:, None]).squeeze(1).clamp(1e-4, 0.999)
        sigma = width / (2.0 * math.sqrt(2.0) * torch.erfinv(p_m))
        sigma = sigma.clamp_min(min_sigma_frac * width)
        z = math.sqrt(2.0)
        cdf_hi = 0.5 * (1.0 + torch.erf((hi[None, :] - gt[:, None]) / (sigma[:, None] * z)))
        cdf_lo = 0.5 * (1.0 + torch.erf((lo[None, :] - gt[:, None]) / (sigma[:, None] * z)))
        ref = (cdf_hi - cdf_lo).clamp_min(0.0)
        ref = ref / ref.sum(-1, keepdim=True).clamp_min(1e-12)
    if weight is None and not per_building:
        return F.kl_div(logp, ref, reduction="batchmean")
    kl = F.kl_div(logp, ref, reduction="none").sum(-1)
    if per_building:
        return kl
    return (weight.float() * kl).sum() / logp.shape[0]
