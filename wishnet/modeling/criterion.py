# -*- coding: utf-8 -*-
"""MaskDINO's set criterion with the height and offset terms.

* heights: L1 (in units of LOSS_SCALE metres), the distribution constraint
  and the bin cross-entropy on the binned head, whose target is the one-hot
  bin of the labelled height or, with HEIGHT.BIN_CE_SOFT_SIGMA_M, SORD's soft
  ordinal target around it; with HEIGHT.BIN_WEIGHTS every building's height
  terms are weighted by its height bin, and with BIN_WEIGHTS_KEEP_TOTAL each
  weighted term is rescaled to its unweighted total;
* offsets: smooth L1 on the roof -> footprint translation, only where the
  labelled offset is physically plausible for the building's height;
* a matched building that is truncated by the window, or whose floor count was
  never recorded, gets neither term (``truncated`` in the targets);
* the denoising queries (noised ground-truth boxes) take the height and offset
  terms as well as class, box and mask; the two-stage encoder proposals take
  class, box and mask only.
"""
import logging

import torch
import torch.nn.functional as F

from detectron2.utils.events import get_event_storage

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from maskdino.modeling.criterion import SetCriterion, calculate_uncertainty, dice_loss_jit, sigmoid_ce_loss_jit
from maskdino.modeling.matcher import HungarianMatcher
from maskdino.utils.misc import is_dist_avail_and_initialized
from detectron2.utils.comm import get_world_size

from .heads import dc_loss, make_bin_edges, soft_bin_target

# output entries the matcher must not see
_SKIP_KEYS = ("aux_outputs", "interm_outputs", "optical_pass")
# MODEL.WISHNET.OPTICAL_PASS: the terms of the SAR-free pass (it predicts no masks)
OPTICAL_PASS_LOSSES = ("labels", "boxes", "heights", "offsets")
# their weight keys
OPTICAL_PASS_KEYS = ("loss_ce", "loss_bbox", "loss_giou", "loss_height_l1", "loss_height_dc",
                     "loss_height_ce", "loss_offset")

# ground-truth masks converted to float at a time for point sampling: a whole
# tile holds up to 405 buildings, 1.7 GB as one float tensor at 1024 x 1024
MASK_CHUNK = 64


def sample_matched_masks(targets, indices, coords, dtype, chunk=MASK_CHUNK):
    """point_sample of the matched ground-truth masks (indices: per image
    (queries, targets)), in the order of _get_tgt_permutation_idx, at coords
    (N, P, 2) -> (N, P), taken from every image's bool masks ``chunk`` at a
    time, without first gathering them into one float tensor."""
    out, k = [], 0
    for t, (_, J) in zip(targets, indices):
        for i in range(0, len(J), chunk):
            j = J[i:i + chunk]
            out.append(point_sample(t["masks"][j][:, None].to(dtype), coords[k:k + len(j)],
                                    align_corners=False).squeeze(1))
            k += len(j)
    return torch.cat(out) if out else coords.new_zeros((0, coords.shape[1]), dtype=dtype)


def _put(name, value):
    try:
        get_event_storage().put_scalar(name, float(value))
    except Exception:
        pass


class WISHCriterion(SetCriterion):
    def __init__(self, *args, bin_edges=None, height_loss_scale=10.0, dc_weight=2.5,
                 dc_min_sigma_frac=0.5, bin_ce_weight=0.1, bin_ce_soft_sigma=0.0, offset_loss_scale=20.0,
                 offset_beta=1.0, offset_max_px_per_m=6.0, offset_abs_floor_px=30.0,
                 height_bin_weights=None, height_keep_total=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("bin_edges", bin_edges, persistent=False)
        self.height_loss_scale = float(height_loss_scale)
        self.dc_weight = float(dc_weight)
        self.dc_min_sigma_frac = float(dc_min_sigma_frac)
        self.bin_ce_weight = float(bin_ce_weight)
        self.bin_ce_soft_sigma = float(bin_ce_soft_sigma)
        self.offset_loss_scale = float(offset_loss_scale)
        self.offset_beta = float(offset_beta)
        self.offset_max_px_per_m = float(offset_max_px_per_m)
        self.offset_abs_floor_px = float(offset_abs_floor_px)
        # height/train_mae_m and offset/train_epe_px describe the matched
        # queries of the final layer; the other terms do not write them
        self._log_train = True
        # the two-stage encoder proposals have no height or offset term
        self.losses_interm = [l for l in self.losses if l not in ("heights", "offsets")]
        # MODEL.WISHNET.HEIGHT.BIN_WEIGHTS: the height terms' weight per height bin (None: off)
        self.register_buffer("height_bin_weights", torch.as_tensor(height_bin_weights, dtype=torch.float32)
                             if height_bin_weights else None, persistent=False)
        # MODEL.WISHNET.HEIGHT.BIN_WEIGHTS_KEEP_TOTAL: the weighted terms rescaled to the unweighted totals
        self.height_keep_total = bool(height_keep_total)
        if height_bin_weights:
            logging.getLogger(__name__).info(
                "[WISHCriterion] height terms weighted per height bin (weights %.2f-%.2f)%s",
                min(height_bin_weights), max(height_bin_weights),
                ", every term rescaled to its unweighted total" if self.height_keep_total else "")

    @staticmethod
    def kept(targets, indices):
        """Matched buildings that supervise height and offset."""
        return torch.cat([~t["truncated"][i] for t, (_, i) in zip(targets, indices)], dim=0)

    @torch.autocast("cuda", enabled=False)
    def loss_heights(self, outputs, targets, indices, num_masks):
        idx = self._get_src_permutation_idx(indices)
        src_h = outputs["pred_heights"][idx].squeeze(-1).float()
        tgt_h = torch.cat([t["heights"][i] for t, (_, i) in zip(targets, indices)], dim=0).float()
        keep = self.kept(targets, indices)
        if src_h.numel() == 0 or not bool(keep.any()):
            z = outputs["pred_heights"].sum() * 0.0
            zl = outputs["pred_height_logp"].sum() * 0.0
            return {"loss_height_l1": z, "loss_height_dc": zl, "loss_height_ce": zl}
        kc = keep.to(idx[0].device)          # the matcher's indices live on the CPU
        src_h, tgt_h, idx = src_h[keep], tgt_h[keep], tuple(v[kc] for v in idx)
        if self.height_bin_weights is not None:
            return self._weighted_heights(outputs, src_h, tgt_h, idx, num_masks)
        losses = {"loss_height_l1": ((src_h - tgt_h) / self.height_loss_scale).abs().sum()
                  / num_masks}
        logp = outputs["pred_height_logp"][idx].float()
        edges = self.bin_edges.to(logp.device)
        losses["loss_height_dc"] = dc_loss(edges, tgt_h, logp, self.dc_min_sigma_frac)
        tgt_bin = (torch.bucketize(tgt_h, edges) - 1).clamp_(0, logp.shape[-1] - 1)
        if self.bin_ce_soft_sigma > 0:
            # HEIGHT.BIN_CE_SOFT_SIGMA_M: the soft ordinal target in place of the one-hot bin
            q = soft_bin_target(edges, tgt_h, self.bin_ce_soft_sigma, tgt_bin)
            losses["loss_height_ce"] = -(q * logp).sum() / num_masks
        else:
            losses["loss_height_ce"] = F.nll_loss(logp, tgt_bin, reduction="sum") / num_masks
        if self._log_train:
            _put("height/train_mae_m", (src_h - tgt_h).abs().mean().item())
        return losses

    def _weighted_heights(self, outputs, src_h, tgt_h, idx, num_masks):
        """loss_heights with MODEL.WISHNET.HEIGHT.BIN_WEIGHTS: every kept
        building's three height terms times the weight of its ground-truth
        height's bin; a height beyond the last bin keeps weight 1."""
        logp = outputs["pred_height_logp"][idx].float()
        edges = self.bin_edges.to(logp.device)
        tgt_bin = (torch.bucketize(tgt_h, edges) - 1).clamp_(0, logp.shape[-1] - 1)
        w = self.height_bin_weights.to(logp.device)[tgt_bin]
        w = torch.where(tgt_h > edges[-1], torch.ones_like(w), w)
        if self.height_keep_total:
            return self._kept_total_heights(src_h, tgt_h, logp, edges, tgt_bin, w, num_masks)
        losses = {"loss_height_l1": (w * ((src_h - tgt_h) / self.height_loss_scale).abs()).sum()
                  / num_masks}
        losses["loss_height_dc"] = dc_loss(edges, tgt_h, logp, self.dc_min_sigma_frac, weight=w)
        if self.bin_ce_soft_sigma > 0:
            q = soft_bin_target(edges, tgt_h, self.bin_ce_soft_sigma, tgt_bin)
            losses["loss_height_ce"] = -(w * (q * logp).sum(-1)).sum() / num_masks
        else:
            losses["loss_height_ce"] = (w * F.nll_loss(logp, tgt_bin, reduction="none")).sum() / num_masks
        if self._log_train:
            _put("height/train_mae_m", (src_h - tgt_h).abs().mean().item())
        return losses

    def _kept_total_heights(self, src_h, tgt_h, logp, edges, tgt_bin, w, num_masks):
        """_weighted_heights with MODEL.WISHNET.HEIGHT.BIN_WEIGHTS_KEEP_TOTAL: every
        term is c * sum_i w_i l_i with c = sum_i l_i / sum_i w_i l_i of this
        batch, taken without gradient -- the gradient c * sum_i w_i grad l_i, the
        value that of the unweighted term."""
        def kept(l):
            lw = w * l
            c = (l.sum() / lw.sum().clamp_min(1e-12)).detach()
            return c * lw.sum(), c

        l1, c1 = kept(((src_h - tgt_h) / self.height_loss_scale).abs())
        losses = {"loss_height_l1": l1 / num_masks}
        dc, _ = kept(dc_loss(edges, tgt_h, logp, self.dc_min_sigma_frac, per_building=True))
        losses["loss_height_dc"] = dc / logp.shape[0]
        if self.bin_ce_soft_sigma > 0:
            q = soft_bin_target(edges, tgt_h, self.bin_ce_soft_sigma, tgt_bin)
            ce, _ = kept(-(q * logp).sum(-1))
        else:
            ce, _ = kept(F.nll_loss(logp, tgt_bin, reduction="none"))
        losses["loss_height_ce"] = ce / num_masks
        if self._log_train:
            _put("height/train_mae_m", (src_h - tgt_h).abs().mean().item())
            # the weighted L1 over the unweighted: the gain the rescaling takes back
            _put("height/weight_gain", 1.0 / max(float(c1), 1e-12))
        return losses

    @torch.autocast("cuda", enabled=False)
    def loss_offsets(self, outputs, targets, indices, num_masks):
        idx = self._get_src_permutation_idx(indices)
        src = outputs["pred_offsets"][idx].float()
        tgt = torch.cat([t["offsets"][i] for t, (_, i) in zip(targets, indices)], dim=0).float()
        tgt_h = torch.cat([t["heights"][i] for t, (_, i) in zip(targets, indices)], dim=0).float()
        if src.numel() == 0:
            return {"loss_offset": outputs["pred_offsets"].sum() * 0.0}
        limit = self.offset_max_px_per_m * tgt_h + self.offset_abs_floor_px
        keep = (tgt.norm(dim=-1) <= limit) & self.kept(targets, indices)
        if not bool(keep.any()):
            return {"loss_offset": outputs["pred_offsets"].sum() * 0.0}
        l = F.smooth_l1_loss(src[keep], tgt[keep], reduction="none", beta=self.offset_beta)
        if self._log_train:
            _put("offset/train_epe_px", (src[keep] - tgt[keep]).norm(dim=-1).mean().item())
        return {"loss_offset": l.sum() / self.offset_loss_scale / max(float(num_masks), 1.0)}

    def loss_masks(self, outputs, targets, indices, num_masks):
        """MaskDINO's mask loss (point-sampled sigmoid CE and dice) with the same
        values and random points; the point labels come from every image's bool
        masks directly, a chunk of them at a time, instead of a float copy of the
        whole padded target stack (about 6 GB for the densest pair of whole tiles)."""
        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx][:, None]
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks, lambda logits: calculate_uncertainty(logits), self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            point_labels = sample_matched_masks(targets, indices, point_coords, src_masks.dtype)
        point_logits = point_sample(src_masks, point_coords, align_corners=False).squeeze(1)
        return {"loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
                "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks)}

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        if loss == "heights":
            return self.loss_heights(outputs, targets, indices, num_masks)
        if loss == "offsets":
            return self.loss_offsets(outputs, targets, indices, num_masks)
        return super().get_loss(loss, outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, mask_dict=None):
        """MaskDINO's SetCriterion.forward.  The two-stage proposals take no
        height / offset terms; the denoising part takes them."""
        outputs_without_aux = {k: v for k, v in outputs.items() if k not in _SKIP_KEYS}
        dev = outputs["pred_logits"].device
        if self.dn != "no" and mask_dict is not None:
            output_known_lbs_bboxes, num_tgt, single_pad, scalar = self.prep_for_dn(mask_dict)
            exc_idx = []
            for t in targets:
                n = len(t["labels"])
                if n > 0:
                    tt = torch.arange(n, device=dev).long().unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = tt.flatten()
                    output_idx = ((torch.arange(scalar, device=dev) * single_pad).long()
                                  .unsqueeze(1) + tt).flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([], dtype=torch.long, device=dev)
                exc_idx.append((output_idx, tgt_idx))

        indices = self.matcher(outputs_without_aux, targets)
        num_masks = torch.as_tensor([sum(len(t["labels"]) for t in targets)],
                                    dtype=torch.float, device=dev)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        # this call's matching, for optical_pass_losses
        self._last = {"final": indices, "aux": [], "num_masks": num_masks}
        if self.dn != "no" and mask_dict is not None:
            self._last.update(exc_idx=exc_idx, dn_num=num_masks * scalar)

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        dn_on = self.dn != "no" and mask_dict is not None
        if dn_on:
            l_dict = {}
            self._log_train = False
            try:
                for loss in self.dn_losses:
                    l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, exc_idx,
                                                num_masks * scalar))
            finally:
                self._log_train = True
            losses.update({k + "_dn": v for k, v in l_dict.items()})
        elif self.dn != "no":
            losses.update(self._zero_dn(""))

        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            ind = self.matcher(aux, targets)
            self._last["aux"].append(ind)
            # train_mae_m / train_epe_px describe the final layer only
            self._log_train = False
            try:
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux, targets, ind, num_masks)
                    losses.update({k + f"_{i}": v for k, v in l_dict.items()})
            finally:
                self._log_train = True
            start = 0 if "interm_outputs" in outputs else 1
            if i >= start:
                if dn_on:
                    out_ = output_known_lbs_bboxes["aux_outputs"][i]
                    l_dict = {}
                    self._log_train = False
                    try:
                        for loss in self.dn_losses:
                            l_dict.update(self.get_loss(loss, out_, targets, exc_idx, num_masks * scalar))
                    finally:
                        self._log_train = True
                    losses.update({k + f"_dn_{i}": v for k, v in l_dict.items()})
                elif self.dn != "no":
                    losses.update(self._zero_dn(f"_{i}"))

        if "interm_outputs" in outputs:
            interm = outputs["interm_outputs"]
            ind = self.matcher(interm, targets)
            for loss in self.losses_interm:
                l_dict = self.get_loss(loss, interm, targets, ind, num_masks)
                losses.update({k + "_interm": v for k, v in l_dict.items()})
        return losses

    def optical_pass_losses(self, opt, targets):
        """MODEL.WISHNET.OPTICAL_PASS: the class, box, height and offset terms of the
        decoder's SAR-free pass (WISHDecoder._optical_pass), right after the
        fused call of forward on the same targets: decoder layer j's matching
        queries against the matching that call made for the same layer, the
        denoising queries against their own boxes.  Keys <term>_optpass for the
        last layer and <term>_optpass_<j> for layer j (as the auxiliary terms),
        with _dn before _optpass for the denoising queries."""
        last = self._last
        layer_ind = last["aux"][1:] + [last["final"]]       # decoder layer j <-> fused prediction j + 1
        n = len(opt["layers"])
        losses = {}
        self._log_train = False
        try:
            for j, out in enumerate(opt["layers"]):
                sfx = "_optpass" if j == n - 1 else "_optpass_%d" % (j + 1)
                for loss in OPTICAL_PASS_LOSSES:
                    l_dict = self.get_loss(loss, out, targets, layer_ind[j], last["num_masks"])
                    losses.update({k + sfx: v for k, v in l_dict.items()})
                if opt["dn_layers"] is not None and "exc_idx" in last:
                    for loss in OPTICAL_PASS_LOSSES:
                        l_dict = self.get_loss(loss, opt["dn_layers"][j], targets, last["exc_idx"], last["dn_num"])
                        losses.update({k + "_dn" + sfx: v for k, v in l_dict.items()})
        finally:
            self._log_train = True
        return losses

    def _zero_dn(self, suffix):
        z = torch.as_tensor(0.0).to("cuda")
        keys = ["loss_bbox_dn", "loss_giou_dn", "loss_ce_dn"]
        if self.dn == "seg":
            keys += ["loss_mask_dn", "loss_dice_dn"]
        if "heights" in self.dn_losses:
            keys += ["loss_height_l1_dn", "loss_height_dc_dn", "loss_height_ce_dn"]
        if "offsets" in self.dn_losses:
            keys += ["loss_offset_dn"]
        return {k + suffix: z for k in keys}


def build_criterion(cfg):
    """The weight dictionary and the criterion."""
    m = cfg.MODEL.MaskDINO
    h, o = cfg.MODEL.WISHNET.HEIGHT, cfg.MODEL.WISHNET.OFFSET
    matcher = HungarianMatcher(
        cost_class=m.COST_CLASS_WEIGHT, cost_mask=m.COST_MASK_WEIGHT,
        cost_dice=m.COST_DICE_WEIGHT, cost_box=m.COST_BOX_WEIGHT,
        cost_giou=m.COST_GIOU_WEIGHT, num_points=m.TRAIN_NUM_POINTS)
    weight_dict = {
        "loss_ce": m.CLASS_WEIGHT, "loss_mask": m.MASK_WEIGHT, "loss_dice": m.DICE_WEIGHT,
        "loss_bbox": m.BOX_WEIGHT, "loss_giou": m.GIOU_WEIGHT,
        "loss_height_l1": h.L1_WEIGHT, "loss_height_dc": h.DC_WEIGHT,
        "loss_height_ce": h.BIN_CE_WEIGHT, "loss_offset": o.WEIGHT,
    }
    if m.TWO_STAGE:
        weight_dict.update({k + "_interm": v for k, v in weight_dict.items()})
    if m.DN == "standard":
        weight_dict.update({k + "_dn": v for k, v in weight_dict.items()
                            if k not in ("loss_mask", "loss_dice")})
        dn_losses = ["labels", "boxes"]
    elif m.DN == "seg":
        weight_dict.update({k + "_dn": v for k, v in weight_dict.items()})
        dn_losses = ["labels", "masks", "boxes"]
    else:
        dn_losses = []
    if dn_losses:
        # the denoising queries also take the height / offset terms (their
        # weights, *_dn and *_dn_<i>, are in the dictionary already)
        dn_losses = dn_losses + ["heights", "offsets"]
    base = dict(weight_dict)
    if m.DEEP_SUPERVISION:
        aux = {}
        for i in range(m.DEC_LAYERS):
            aux.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux)
    lam = float(cfg.MODEL.WISHNET.OPTICAL_PASS.WEIGHT)
    if lam > 0:
        # the optical pass: WEIGHT times the base weight, every decoder layer
        for j in range(1, m.DEC_LAYERS + 1):
            sfx = "_optpass" if j == m.DEC_LAYERS else "_optpass_%d" % j
            for k in OPTICAL_PASS_KEYS:
                weight_dict[k + sfx] = lam * base[k]
                weight_dict[k + "_dn" + sfx] = lam * base[k]
    _, edges = make_bin_edges(h.BIN_MIN_M, h.BIN_MAX_M, h.BIN_STEP_M)
    return WISHCriterion(
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        matcher=matcher, weight_dict=weight_dict, eos_coef=m.NO_OBJECT_WEIGHT,
        losses=["labels", "masks", "boxes", "heights", "offsets"],
        num_points=m.TRAIN_NUM_POINTS, oversample_ratio=m.OVERSAMPLE_RATIO,
        importance_sample_ratio=m.IMPORTANCE_SAMPLE_RATIO, dn=m.DN, dn_losses=dn_losses,
        panoptic_on=False, semantic_ce_loss=False,
        bin_edges=edges, height_loss_scale=h.LOSS_SCALE, dc_weight=h.DC_WEIGHT,
        dc_min_sigma_frac=h.DC_MIN_SIGMA_FRAC, bin_ce_weight=h.BIN_CE_WEIGHT,
        bin_ce_soft_sigma=h.BIN_CE_SOFT_SIGMA_M,
        offset_loss_scale=o.LOSS_SCALE, offset_beta=o.BETA,
        offset_max_px_per_m=o.MAX_PX_PER_M, offset_abs_floor_px=o.ABS_FLOOR_PX,
        height_bin_weights=list(h.BIN_WEIGHTS), height_keep_total=bool(h.BIN_WEIGHTS_KEEP_TOTAL))
