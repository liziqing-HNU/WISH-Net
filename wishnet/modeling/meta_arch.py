# -*- coding: utf-8 -*-
"""WISH-Net: building instances with heights from VHR optical imagery, GF-3
SAR and Sentinel-1 InSAR.

The optical path is MaskDINO as released (ResNet-50, the multi-scale
deformable pixel decoder, two-stage query selection, denoising, nine decoder
layers) with a height and an offset head at every decoder layer.
MODEL.WISHNET.MODE adds the SAR side:

    opt     no SAR (the optical model)
    fusion  the SAR branch (sar_branch.py: GF-3, optionally InSAR, one joint
            encoder), optionally the two-way exchange at the encoder stage
            (exchange.py), and a range-anchored SAR read with a gated
            injection in every decoder layer (decoder.py)

Outputs per instance: class, roof mask, box (roof and footprint), height in
metres and the roof -> footprint offset in pixels.  The whole 2048 tile goes
in resampled by INPUT.TILE_SCALE (1024 px, 0.83 m) and the outputs are mapped
back to the tile.
"""
import logging
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom

from maskdino.utils import box_ops

from .criterion import build_criterion
from .exchange import EncoderExchange, optical_fpn, optical_transformer
from .sar_branch import SARBranch, checkpointed, flatten_levels

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class WISHNet(nn.Module):
    @configurable
    def __init__(self, *, backbone, sem_seg_head, criterion, sar_branch, num_queries: int,
                 size_divisibility: int, pixel_mean: Tuple[float], pixel_std: Tuple[float],
                 test_topk_per_image: int, focus_on_box: bool = False, exchange=None,
                 opt_encoder_checkpoint: bool = False):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.sar_branch = sar_branch
        # MODEL.WISHNET.ENCODER_EXCHANGE: the two-way exchange at the encoder stage (None: off)
        self.exchange = exchange
        if opt_encoder_checkpoint:
            # MODEL.WISHNET.OPT_ENCODER_CHECKPOINT: the optical encoder's Transformer
            # layers recompute their activations in the backward
            for layer in self.sem_seg_head.pixel_decoder.transformer.encoder.layers:
                checkpointed(layer)
            logger.info("[WISH-Net] the optical encoder's Transformer layers under activation checkpointing")
        self.num_queries = num_queries
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.test_topk_per_image = test_topk_per_image
        self.focus_on_box = focus_on_box

    @classmethod
    def from_config(cls, cfg):
        g = cfg.MODEL.WISHNET
        backbone = build_backbone(cfg)
        if g.OPT_BACKBONE_WEIGHTS:
            DetectionCheckpointer(backbone).load(g.OPT_BACKBONE_WEIGHTS)
            logger.info("[WISH-Net] optical ResNet-50 weights from %s", g.OPT_BACKBONE_WEIGHTS)
        return {
            "backbone": backbone,
            "sem_seg_head": build_sem_seg_head(cfg, backbone.output_shape()),
            "criterion": build_criterion(cfg),
            "sar_branch": SARBranch(cfg) if g.MODE == "fusion" else None,
            "num_queries": cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES,
            "size_divisibility": cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "focus_on_box": cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX,
            "opt_encoder_checkpoint": bool(g.OPT_ENCODER_CHECKPOINT),
            # built last: every other module starts as without it
            "exchange": EncoderExchange(cfg) if g.ENCODER_EXCHANGE.ENABLED else None,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _log(values):
        try:
            st = get_event_storage()
            for k, v in values.items():
                st.put_scalar(k, float(v))
        except Exception:
            pass

    def _weighted(self, losses):
        """Apply the weight dictionary; a term without a weight is dropped."""
        for k in list(losses):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                losses.pop(k)
        return losses

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        out = []
        for t in targets:
            h, w = t.image_size
            size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            masks = torch.zeros((t.gt_masks.shape[0], h_pad, w_pad), dtype=t.gt_masks.dtype,
                                device=t.gt_masks.device)
            masks[:, :t.gt_masks.shape[1], :t.gt_masks.shape[2]] = t.gt_masks
            out.append({
                "labels": t.gt_classes,
                "masks": masks,
                "boxes": box_ops.box_xyxy_to_cxcywh(t.gt_boxes.tensor) / size_xyxy,
                "heights": t.gt_heights,
                "offsets": t.gt_offsets,
                # truncated by the window, or no recorded floor count: matched,
                # but no height or offset term
                "truncated": (t.gt_truncated if t.has("gt_truncated")
                              else torch.zeros(len(t), dtype=torch.bool, device=self.device)),
            })
        return out

    def _insar_input(self, batched_inputs):
        """The InSAR tiles of the batch, standardised (None without INSAR)."""
        if self.sar_branch is None or self.sar_branch.insar is None:
            return None
        return self.sar_branch.prepare_insar([x["insar_image"] for x in batched_inputs], self.device)

    def _sar_memory(self, sar_input, batched_inputs, images, insar_input=None):
        """The SAR memory plus what the decoder needs to put a point of the
        optical input into the SAR window."""
        mem = self.sar_branch(sar_input, insar_input)
        mem["img_size"] = tuple(images.tensor.shape[-2:])
        mem["sar_size"] = tuple(batched_inputs[0]["sar_image"].shape[-2:])
        mem["sar_off"] = torch.as_tensor(
            [[float(v) for v in x.get("sar_offset", (0.0, 0.0))] for x in batched_inputs],
            dtype=torch.float32, device=self.device)
        if any("geo_M" in x for x in batched_inputs):
            # INPUT.GEO_AUG: each sample's flip / rotation, for the range reads
            mem["geo"] = torch.stack([x["geo_M"] if "geo_M" in x else torch.eye(2)
                                      for x in batched_inputs]).to(self.device, torch.float32)
        return mem

    def _batch_sar_memory(self, batched_inputs, images):
        """The batch's SAR memory."""
        sar = self.sar_branch.prepare([x["sar_image"] for x in batched_inputs], self.device)
        return self._sar_memory(sar, batched_inputs, images, self._insar_input(batched_inputs))

    def _exchanged(self, features, levels, ctx):
        """MODEL.WISHNET.ENCODER_EXCHANGE: the optical Transformer's levels
        ``levels`` and the SAR memory ``ctx`` -- as the SAR branch made it, or
        zeroed or replaced by a readout, which then reaches the optical memory
        too -- through the exchange, then the optical FPN -> (the optical
        multi-scale features, the mask features, the SAR context after the
        exchange: GF-3's s32, s16, s8 and the InSAR levels rewritten, s4 as it
        was)."""
        maps = ctx["maps"]
        n = 4                                   # GF-3's s32, s16, s8, s4; then i32, i16
        o, gm, im = self.exchange(levels, maps[:3], maps[n:] if len(maps) > n else None, ctx["img_size"],
                                  ctx["sar_size"], ctx["sar_off"], ctx.get("geo"))
        mask_features, ms_feats = optical_fpn(self.sem_seg_head.pixel_decoder, features, o)
        new = list(gm) + list(maps[3:n]) + (list(im) if im is not None else [])
        out = dict(ctx)
        out["maps"] = new
        out["value"], out["shapes"], out["starts"] = flatten_levels(new)
        return ms_feats, mask_features, out

    def decoder_inputs(self, features, ctx):
        """What the decoder reads for the backbone features ``features`` and the
        SAR context ``ctx`` (None: no SAR) -> (the optical multi-scale
        features, the mask features, the SAR context): MaskDINO's
        forward_features and ``ctx`` as it is, or with ENCODER_EXCHANGE the
        optical Transformer, the exchange with ``ctx`` and the FPN (``ctx``
        None: the optical memory before the exchange).  For analyses that
        decode outside forward."""
        pd = self.sem_seg_head.pixel_decoder
        if self.exchange is None:
            mask_features, _, ms_feats = pd.forward_features(features, None)
            return ms_feats, mask_features, ctx
        levels = optical_transformer(pd, features)
        if ctx is None:
            mask_features, ms_feats = optical_fpn(pd, features, levels)
            return ms_feats, mask_features, None
        return self._exchanged(features, levels, ctx)

    # ---------------------------------------------------------------- forward
    def forward(self, batched_inputs):
        images = [(x["image"].to(self.device) - self.pixel_mean) / self.pixel_std
                  for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        targets = None
        if self.training:
            targets = self.prepare_targets([x["instances"].to(self.device) for x in batched_inputs],
                                           images)

        features = self.backbone(images.tensor)
        pred = self.sem_seg_head.predictor
        ctx, levels = None, None
        if self.exchange is None:
            mask_features, _, ms_feats = self.sem_seg_head.pixel_decoder.forward_features(features, None)
            if self.sar_branch is not None:
                ctx = self._batch_sar_memory(batched_inputs, images)
        else:
            # MODEL.WISHNET.ENCODER_EXCHANGE: the optical Transformer, the SAR branch,
            # the exchange, then the optical FPN
            levels = optical_transformer(self.sem_seg_head.pixel_decoder, features)
            ms_feats, mask_features, ctx = self._exchanged(features, levels,
                                                           self._batch_sar_memory(batched_inputs, images))
            if self.training:
                self._log({"exchange/" + k: v for k, v in self.exchange.stats.items()})
        pred.sar_ctx = ctx
        pred.optical_pass_active = self.training          # MODEL.WISHNET.OPTICAL_PASS, this forward only
        # ENCODER_EXCHANGE: the optical pass decodes from the optical memory before the exchange
        pred.exchange_pre = {"levels": levels} if (self.training and levels is not None) else None
        try:
            outputs, mask_dict = pred(ms_feats, mask_features, None, targets=targets)
        finally:
            pred.sar_ctx = None
            pred.optical_pass_active = False
            pred.exchange_pre = None
        if not self.training:
            return self.inference(outputs, batched_inputs, images)

        op_args = outputs.pop("optical_pass_args", None)
        losses = self.criterion(outputs, targets, mask_dict)
        if op_args is not None:
            # MODEL.WISHNET.OPTICAL_PASS, after the fused criterion: nothing of the pass
            # is alive at the memory peak of the fused mask terms
            opt = pred._optical_pass(*op_args)
            losses.update(self.criterion.optical_pass_losses(opt, targets))
            del opt
        del op_args
        losses = self._weighted(losses)
        self._log({"fusion/" + k: v for k, v in pred.stats.items()})
        return losses

    # ----------------------------------------------------- fixed-set readout
    @torch.no_grad()
    def sar_memory_maps(self, batched_inputs):
        """The SAR memory of the batch (the SAR branch alone): [(B, C, h, w)] per level."""
        sar = self.sar_branch.prepare([x["sar_image"] for x in batched_inputs], self.device)
        return self.sar_branch(sar, self._insar_input(batched_inputs))["maps"]

    @torch.no_grad()
    def gtq_sar_readout(self, batched_inputs, mean_maps=None, with_zero=True, optpass=False, per_modality=False,
                        detection=False):
        """The fixed-set readout (wishnet/evaluation/readout.py): every image's
        ground-truth boxes decoded as queries (WISHDecoder.forward_gt_queries,
        the SAR reads at the predicted offsets) with the image's SAR ("real"),
        with the SAR memory replaced by ``mean_maps`` (one (C, h, w) per level;
        "mean") and with a zero SAR input ("zero"); with ``per_modality``
        (INSAR) also with GF-3 alone and InSAR alone zeroed ("zero_gf3",
        "zero_ins"); with ``optpass`` (a model trained with
        MODEL.WISHNET.OPTICAL_PASS) once more with every SAR step skipped, the
        state that pass trains ("opt", decoded last).  -> per image {"gt",
        "truncated", "real", "mean", "zero", ...} (heights, CPU).  With
        ENCODER_EXCHANGE every SAR state goes through the exchange, into the
        optical memory as well, and "opt" decodes the optical memory before it.
        With ``detection`` also "det_<state>": the detection record of that
        decoding's matching queries (readout.det_record).  Needs the evaluation
        mapper's instances (eval mode).  Without a SAR branch (mode opt) only
        "real", the model's own."""
        images = [(x["image"].to(self.device) - self.pixel_mean) / self.pixel_std for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)
        pd = self.sem_seg_head.pixel_decoder
        levels = None
        if self.exchange is None:
            mask_features, _, ms_feats = pd.forward_features(features, None)
        else:
            levels = optical_transformer(pd, features)

        def decoder_inputs(c):
            """(multi-scale features, mask features, SAR context) for the SAR state c (None: no SAR)."""
            if levels is None:
                return ms_feats, mask_features, c
            if c is None:
                mf, ms = optical_fpn(pd, features, levels)
                return ms, mf, None
            return self._exchanged(features, levels, c)

        insts = [x["instances"].to(self.device) for x in batched_inputs]
        boxes = []
        for t in insts:
            h, w = t.image_size
            size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            boxes.append(box_ops.box_xyxy_to_cxcywh(t.gt_boxes.tensor) / size_xyxy)
        if self.sar_branch is None:
            variants = [("real", None)]
        else:
            sar = self.sar_branch.prepare([x["sar_image"] for x in batched_inputs], self.device)
            insar = self._insar_input(batched_inputs)
            ctx = self._sar_memory(sar, batched_inputs, images, insar)
            variants = [("real", ctx)]
            if mean_maps is not None:
                alt = dict(ctx)
                alt["maps"] = [m[None].to(ref.dtype).expand(ref.shape[0], -1, -1, -1).contiguous()
                               for m, ref in zip(mean_maps, ctx["maps"])]
                alt["value"], alt["shapes"], alt["starts"] = flatten_levels(alt["maps"])
                variants.append(("mean", alt))
            if with_zero:
                zi = None if insar is None else torch.zeros_like(insar)
                variants.append(("zero", self._sar_memory(torch.zeros_like(sar), batched_inputs, images, zi)))
            if per_modality and insar is not None:
                variants.append(("zero_gf3", self._sar_memory(torch.zeros_like(sar), batched_inputs, images, insar)))
                variants.append(("zero_ins", self._sar_memory(sar, batched_inputs, images, torch.zeros_like(insar))))
            if optpass:
                # no SAR context: the decoder skips every SAR step
                variants.append(("opt", None))
        out = [{"gt": t.gt_heights.float().cpu(), "truncated": t.gt_truncated.cpu()} for t in insts]
        pred = self.sem_seg_head.predictor
        if detection:
            from ..evaluation.readout import det_record
        for name, c in variants:
            ms, mf, cx = decoder_inputs(c)
            pred.sar_ctx = cx
            try:
                res = pred.forward_gt_queries(ms, mf, boxes, detection=detection)
            finally:
                pred.sar_ctx = None
            for o, hq in zip(out, res[0]):
                o[name] = hq.float().cpu()
            if detection:
                for o, t, dd in zip(out, insts, res[2]):
                    o["det_" + name] = det_record(dd, t.gt_masks, t.gt_truncated)
            del ms, mf, cx, res
        return out

    # ------------------------------------------------------------- inference
    def inference(self, outputs, batched_inputs, images):
        mask_pred = F.interpolate(outputs["pred_masks"], size=images.tensor.shape[-2:],
                                  mode="bilinear", align_corners=False)
        processed = []
        for mc, mp, mb, hr, orr, inp, image_size in zip(
                outputs["pred_logits"], mask_pred, outputs["pred_boxes"], outputs["pred_heights"],
                outputs["pred_offsets"], batched_inputs, images.image_sizes):
            height = inp.get("height", image_size[0])
            width = inp.get("width", image_size[1])
            new_size = mp.shape[-2:]
            # INPUT.TILE_SCALE: the window is larger than the input; its masks are
            # made in instance_inference, from the input-size logits in chunks
            # (300 float masks at 2048 would be 5 GB at once)
            scaled = (int(height), int(width)) != (int(image_size[0]), int(image_size[1]))
            if scaled:
                mp = mp[:, :image_size[0], :image_size[1]]
            else:
                mp = retry_if_cuda_oom(sem_seg_postprocess)(mp, image_size, height, width)
            mc, mb = mc.to(mp), mb.to(mp)
            bh = new_size[0] / image_size[0] * height
            bw = new_size[1] / image_size[1] * width
            mb = box_ops.box_cxcywh_to_xyxy(mb) * torch.tensor([bw, bh, bw, bh]).to(mb)
            # offsets are in input px: to px of the original window (INPUT.TILE_SCALE)
            orr = orr.to(mp) * torch.tensor([width / image_size[1], height / image_size[0]]).to(mp)
            processed.append({"instances": retry_if_cuda_oom(self.instance_inference)(
                mc, mp, mb, hr.to(mp), orr, (int(height), int(width)) if scaled else None)})
        return processed

    def instance_inference(self, mask_cls, mask_pred, boxes, heights, offsets, out_size=None):
        num_classes = self.sem_seg_head.num_classes
        scores = mask_cls.sigmoid()
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(
            self.num_queries, 1).flatten(0, 1)
        topk = min(self.test_topk_per_image, scores.numel())
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(topk, sorted=False)
        labels_per_image = labels[topk_indices]
        topk_indices = torch.div(topk_indices, num_classes, rounding_mode="floor")
        mask_pred = mask_pred[topk_indices]
        binary = mask_pred > 0
        flat = binary.flatten(1)
        mask_scores = (mask_pred.sigmoid().flatten(1) * flat).sum(1) / (flat.sum(1) + 1e-6)
        if out_size is not None:
            # the window's masks: the logits resampled bilinearly, 16 at a time,
            # and thresholded; they stay on the GPU, where the evaluator encodes
            # them (450 masks at 2048 are 1.9 GB of bool, too slow to copy out)
            binary = torch.empty((len(mask_pred),) + tuple(out_size), dtype=torch.bool,
                                 device=mask_pred.device)
            for i in range(0, len(mask_pred), 16):
                binary[i:i + 16] = F.interpolate(mask_pred[i:i + 16][None], size=out_size,
                                                 mode="bilinear", align_corners=False)[0] > 0
        result = Instances(binary.shape[-2:])
        result.pred_masks = binary
        result.pred_boxes = Boxes(boxes[topk_indices])
        result.scores = scores_per_image * (1.0 if self.focus_on_box else mask_scores)
        result.pred_classes = labels_per_image
        result.pred_heights = heights[topk_indices].squeeze(-1)
        result.pred_offsets = offsets[topk_indices]
        return result
