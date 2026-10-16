# -*- coding: utf-8 -*-
"""WISH-Net's transformer decoder.

MaskDINO's decoder (two-stage query selection, mask-enhanced box
initialisation, denoising, nine layers with box refinement) with

* a binned height head and an offset head read at every layer, like the
  class, mask and box heads; the denoising queries take the height and offset
  terms too;
* the proposal chain -- the encoder proposals' classes, boxes and masks (and
  from the masks the initial boxes), the initial queries' prediction and the
  offsets the layer-0 SAR reads start from -- with heads of its own
  (``proposal_heads``), deep copies of the shared ones at initialisation, as
  DINO keeps its two-stage heads apart; the decoder layers keep the shared
  heads;
* in mode "fusion", a SAR step in every layer, between the optical deformable
  cross-attention and the FFN: the optical query reads the SAR memory with a
  RangeDeformAttn (offsets and weights from the query and its position, as in
  MaskDINO's cross-attention) and a gated fusion writes the reading into it.

The SAR read starts its points at anchors along the direction towards the
sensor around the footprint centre (fusion.RangeDeformAttn).  The footprint
centre is the layer's reference box centre moved by half the roof ->
footprint offset the previous layer predicted (the box spans roof and
footprint), detached; at layer 0, whose matching queries' reference boxes are
MaskDINO's boxes of their predicted roof masks, it is the roof centre moved by
the whole offset.  It is only the reference point, and where around it to
read is learned.

The decoding loop is MaskDINO's ``TransformerDecoder.forward`` written out
here with the SAR step inserted; the layers' own modules are MaskDINO's, so
their parameters keep upstream names.

MODEL.WISHNET.OPTICAL_PASS (training only, mode fusion): after the fused
decoding the layers run once more on the same queries with every SAR step
skipped -- the optical computation with the same weights and heads -- and
give, per decoder layer, classes, boxes, heights and offsets (no masks) for
the criterion's optical-pass terms.  The forward hands its inputs on
(``out["optical_pass_args"]``) and the meta-architecture runs the pass after
the fused criterion: at that criterion's memory peak (the mask terms) nothing
of the pass is alive, and in the backward the pass, built last, is undone
first.  The optical part then has to answer on its own and cannot hand its
height cues over to the SAR.  The pass's layers are recomputed in the
backward (activation checkpointing) unless OPTICAL_PASS.CHECKPOINT is off.
With MODEL.WISHNET.ENCODER_EXCHANGE the pass decodes from the optical memory
before the exchange (the meta-architecture hands it over as
``exchange_pre``): its matching queries start from the fused decoding's tokens
with their content from that memory (_pre_exchange_inputs), so nothing of the
SAR reaches it.
"""
import copy
import logging

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from detectron2.config import configurable

from maskdino.modeling.transformer_decoder.maskdino_decoder import (
    TRANSFORMER_DECODER_REGISTRY,
    MaskDINODecoder,
)
from maskdino.utils import box_ops
from maskdino.utils.utils import gen_encoder_output_proposals, gen_sineembed_for_position, inverse_sigmoid

from .fusion import GatedFusion, RangeDeformAttn
from .heads import BinnedHeightHead, OffsetHead, mask_boxes_vectorised

logger = logging.getLogger(__name__)

EXTRA_KEYS = ("pred_heights", "pred_height_logp", "pred_offsets")
# the proposal chain's heads and the shared heads they start from
PROPOSAL_HEADS = {"class_embed": "class_embed", "bbox_embed": "_bbox_embed", "norm": "decoder_norm",
                  "mask_embed": "mask_embed", "offset_embed": "offset_embed", "height_embed": "height_embed"}


@TRANSFORMER_DECODER_REGISTRY.register()
class WISHDecoder(MaskDINODecoder):
    @configurable
    def __init__(self, in_channels, mask_classification=True, *, mode="opt", sar_levels=4, fusion_heads=8,
                 gate_bias=0.0, inject_rel=0.1, height_cfg=None, offset_scale=50.0, range_cfg=None,
                 optical_pass=0.0, optical_pass_checkpoint=True, **kwargs):
        super().__init__(in_channels, mask_classification, **kwargs)
        if not (self.two_stage and self.initial_pred):
            raise ValueError("WISHDecoder is written for MaskDINO's two-stage decoder with "
                             "the initial prediction (TWO_STAGE, INITIAL_PRED)")
        d = self.hidden_dim
        self.height_embed = BinnedHeightHead(d, **(height_cfg or {}))
        self.offset_embed = OffsetHead(d, offset_scale)
        # the proposal chain's own heads, copies of the shared ones (no random
        # numbers drawn: the other modules start as without them)
        self.proposal_heads = nn.ModuleDict(
            {k: copy.deepcopy(getattr(self, n)) for k, n in PROPOSAL_HEADS.items()})
        # MODEL.WISHNET.OPTICAL_PASS: the SAR-free second decoding in training; the
        # meta-architecture sets optical_pass_active for its training forward only
        self.optical_pass = float(optical_pass) > 0
        self.optical_pass_checkpoint = bool(optical_pass_checkpoint)
        self.optical_pass_active = False
        # MODEL.WISHNET.ENCODER_EXCHANGE: the optical memory before the exchange, set
        # by the meta-architecture for its training forward
        self.exchange_pre = None
        self._topk = None        # per forward: the tokens the query selection took
        self.mode = mode
        if mode == "fusion":
            keys = [str(i) for i in range(self.num_layers)]
            rc = dict(range_cfg)
            # the optical query reads the SAR memory; the gated fusion writes the reading
            self.sar_read = nn.ModuleDict({k: RangeDeformAttn(d, sar_levels, fusion_heads, **rc) for k in keys})
            self.sar_fuse = nn.ModuleDict({k: GatedFusion(d, gate_bias, inject_rel) for k in keys})
            logger.info("[WISHDecoder] mode fusion, SAR step in all %d layers, range anchors %s m",
                        self.num_layers, list(rc["anchors_m"]))
        else:
            logger.info("[WISHDecoder] mode opt, no SAR")
        self._fp0 = None         # per forward: the layer-0 offset factor of every query
        self.sar_ctx = None      # set by the meta architecture for every forward
        self.stats = {}

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = MaskDINODecoder.from_config(cfg, in_channels, mask_classification)
        g = cfg.MODEL.WISHNET
        ret["mode"] = g.MODE
        # the GF-3 levels s32, s16, s8, s4 (+ the InSAR levels i32, i16)
        ret["sar_levels"] = 4 + (2 if g.INSAR.ENABLED else 0)
        ret["fusion_heads"] = int(g.FUSION.N_HEADS)
        ret["gate_bias"] = float(g.FUSION.GATE_BIAS)
        ret["inject_rel"] = float(g.FUSION.INJECT_REL)
        ret["height_cfg"] = dict(lo=g.HEIGHT.BIN_MIN_M, hi=g.HEIGHT.BIN_MAX_M, step=g.HEIGHT.BIN_STEP_M,
                                 residual_weight=g.HEIGHT.BIN_RESIDUAL_WEIGHT, height_init=g.HEIGHT.INIT)
        ret["offset_scale"] = float(g.OFFSET.SCALE)
        ret["optical_pass"] = float(g.OPTICAL_PASS.WEIGHT)
        ret["optical_pass_checkpoint"] = bool(g.OPTICAL_PASS.CHECKPOINT)
        r = g.RANGE
        # the range anchors, with the metres per px of the SAR window as it goes
        # in: the optical grid, made coarser by INPUT.TILE_SCALE (the whole tile
        # read at 1024: 0.83 m)
        scale = float(cfg.INPUT.TILE_SCALE) if cfg.INPUT.TILE_SCALE > 0 else 1.0
        rc = dict(anchors_m=[float(t) for t in r.ANCHORS_M], toward_sensor_deg=float(r.TOWARD_SENSOR_DEG),
                  gsd_m=float(r.GSD_M) / scale)
        if g.INSAR.ENABLED:
            # anchors of the GF-3 levels towards GF-3, of the InSAR levels towards
            # the InSAR sensor (Sentinel-1 ascending: the opposite side of the buildings)
            rc = dict(rc, toward_sensor_deg=[float(r.TOWARD_SENSOR_DEG)] * 4
                      + [float(g.INSAR.TOWARD_SENSOR_DEG)] * 2)
        ret["range_cfg"] = rc
        return ret

    # ------------------------------------------------------------------ heads
    def _head(self, name, proposal=False):
        """The head ``name`` (a key of PROPOSAL_HEADS): the proposal chain's own
        with ``proposal``, else the shared one."""
        if proposal:
            return self.proposal_heads[name]
        return getattr(self, PROPOSAL_HEADS[name])

    def forward_prediction_heads(self, output, mask_features, pred_mask=True, proposal=False):
        """MaskDINO's heads plus height and offset; output (Q, B, d).  ``proposal``:
        the proposal chain's prediction (the proposals, the initial queries)."""
        x = self._head("norm", proposal)(output).transpose(0, 1)
        outputs_class = self._head("class_embed", proposal)(x)
        outputs_mask = None
        if pred_mask:
            outputs_mask = torch.einsum("bqc,bchw->bqhw", self._head("mask_embed", proposal)(x), mask_features)
        extra = {"pred_offsets": self._head("offset_embed", proposal)(x)}
        h, logp = self._head("height_embed", proposal)(x)
        extra.update({"pred_heights": h.unsqueeze(-1), "pred_height_logp": logp})
        return outputs_class, outputs_mask, extra

    def dn_post_process(self, outputs_class, outputs_coord, mask_dict, outputs_mask, extra):
        """outputs_class and outputs_mask are lists over the predictions (the
        initial one and every layer's), sliced one by one: MaskDINO stacks the
        masks of all ten first, a (10, B, Q, H/4, W/4) copy whose gradient
        buffer alone is 3.4 GB for 450 queries plus the denoising ones."""
        pad = mask_dict["pad_size"]
        out = {"pred_logits": outputs_class[-1][:, :pad], "pred_boxes": outputs_coord[-1][:, :pad],
               "pred_masks": outputs_mask[-1][:, :pad]}
        out["aux_outputs"] = self._set_aux_loss([c[:, :pad] for c in outputs_class],
                                                [m[:, :pad] for m in outputs_mask], outputs_coord[:, :, :pad])
        # the denoising queries' heights and offsets, final and auxiliary, for
        # the criterion's denoising height / offset terms
        dn_extra = {k: v[:, :, :pad] for k, v in extra.items()}
        out.update({k: v[-1] for k, v in dn_extra.items()})
        for i, aux in enumerate(out["aux_outputs"]):
            aux.update({k: v[i] for k, v in dn_extra.items()})
        mask_dict["output_known_lbs_bboxes"] = out
        # the matching queries' part goes on
        extra = {k: v[:, :, pad:] for k, v in extra.items()}
        return ([c[:, pad:] for c in outputs_class], outputs_coord[:, :, pad:],
                [m[:, pad:] for m in outputs_mask], extra)

    # -------------------------------------------------------------- geometry
    def _footprint_sar(self, ref_xy, off, k=0.5):
        """ref_xy (B, Q, 2) box centres in [0, 1] of the input; off (B, Q, 2) px
        -> footprint centres in [0, 1] of the SAR window.  k: the share of the
        offset from the box centre to the footprint centre -- 0.5 for a
        roof-union-footprint box, 1 for a roof box; a number or (1, Q, 1)."""
        c = self.sar_ctx
        H, W = c["img_size"]
        px = ref_xy * ref_xy.new_tensor([W, H]) + k * off
        sh, sw = c["sar_size"]
        return (px + c["sar_off"][:, None, :]) / px.new_tensor([sw, sh])

    # ------------------------------------------------------------ SAR steps
    def _sa_ca(self, layer, tgt, pos, ref_input, memory, memory_mask, starts, shapes, attn_mask):
        """Self-attention and deformable cross-attention of MaskDINO's layer."""
        q = k = layer.with_pos_embed(tgt, pos)
        tgt = layer.norm2(tgt + layer.dropout2(layer.self_attn(q, k, tgt, attn_mask=attn_mask)[0]))
        tgt2 = layer.cross_attn(layer.with_pos_embed(tgt, pos).transpose(0, 1),
                                ref_input.transpose(0, 1).contiguous(),
                                memory.transpose(0, 1), shapes, starts, memory_mask).transpose(0, 1)
        return layer.norm1(tgt + layer.dropout1(tgt2))

    def _sar_step(self, lid, output, pos, ref, off):
        """The optical query reads the SAR memory and the gated fusion writes the
        reading into it.  output, pos (Q, B, d); ref (Q, B, 4); off (B, Q, 2) px."""
        c = self.sar_ctx
        ref_b = ref.transpose(0, 1)
        k = self._fp0 if (lid == 0 and self._fp0 is not None) else 0.5
        fp = self._footprint_sar(ref_b[..., :2], off, k)                 # (B, Q, 2)
        sh, sw = c["sar_size"]
        win = fp.new_tensor([float(sw), float(sh)])                      # the SAR window (W, H), px
        q, p = output.transpose(0, 1), pos.transpose(0, 1)
        key = str(lid)
        a = self.sar_read[key](q + p, fp, win, c["value"], c["shapes"], c["starts"])
        q_new = self.sar_fuse[key](q, a)
        self._stat(self.sar_fuse[key], "opt")
        return q_new.transpose(0, 1)

    def _stat(self, module, tag):
        for key, v in module.last.items():
            self._stat_lists.setdefault("%s_%s" % (tag, key), []).append(v)

    def _decode(self, tgt, memory, memory_mask, refpoints_unsigmoid, starts, shapes, valid_ratios, attn_mask):
        """MaskDINO's TransformerDecoder.forward with the SAR step; all (Q, B, *).
        A batch of flipped or rotated training samples (INPUT.GEO_AUG) hands its
        transforms, sar_ctx["geo"], to the range reads for the decoding."""
        use_sar = self.mode == "fusion" and self.sar_ctx is not None
        geo = self.sar_ctx.get("geo") if use_sar else None
        if geo is None:
            return self._decode_layers(tgt, memory, memory_mask, refpoints_unsigmoid, starts, shapes,
                                       valid_ratios, attn_mask)
        for m in self.sar_read.values():
            m.geo = geo
        try:
            return self._decode_layers(tgt, memory, memory_mask, refpoints_unsigmoid, starts, shapes,
                                       valid_ratios, attn_mask)
        finally:
            for m in self.sar_read.values():
                m.geo = None

    def _optical_layer(self, layer, output, query_pos, ref_input, memory, memory_mask, starts, shapes,
                       attn_mask):
        """One decoder layer without the SAR step (the optical computation)."""
        output = self._sa_ca(layer, output, query_pos, ref_input, memory, memory_mask, starts, shapes, attn_mask)
        return layer.forward_ffn(output)

    def _decode_layers(self, tgt, memory, memory_mask, refpoints_unsigmoid, starts, shapes,
                       valid_ratios, attn_mask, sar=True):
        """``sar`` False: every SAR step skipped (OPTICAL_PASS), the layers
        recomputed in the backward with OPTICAL_PASS.CHECKPOINT."""
        dec = self.decoder
        use_sar = sar and self.mode == "fusion" and self.sar_ctx is not None
        ckpt = not sar and self.optical_pass_checkpoint and torch.is_grad_enabled()
        output = tgt
        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]
        for lid, layer in enumerate(dec.layers):
            ref_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            query_pos = dec.ref_point_head(gen_sineembed_for_position(ref_input[:, :, 0, :]))
            layer_in = output
            if ckpt:
                output = checkpoint(self._optical_layer, layer, output, query_pos, ref_input, memory,
                                    memory_mask, starts, shapes, attn_mask, use_reentrant=False)
            else:
                output = self._sa_ca(layer, output, query_pos, ref_input, memory, memory_mask,
                                     starts, shapes, attn_mask)
                if use_sar:
                    with torch.no_grad():
                        # the offset the heads read from this layer's input: the
                        # previous layer's prediction (the initial one for layer 0)
                        x = layer_in if lid == 0 else dec.norm(layer_in)
                        first = lid == 0
                        off = self._head("offset_embed", first)(self._head("norm", first)(x).transpose(0, 1))
                    output = self._sar_step(lid, output, query_pos, reference_points, off)
                output = layer.forward_ffn(output)
            if dec.bbox_embed is not None:
                new_ref = (dec.bbox_embed[lid](output) + inverse_sigmoid(reference_points)).sigmoid()
                reference_points = new_ref.detach()
                ref_points.append(new_ref)
            intermediate.append(dec.norm(output))
        return ([o.transpose(0, 1) for o in intermediate], [r.transpose(0, 1) for r in ref_points])

    # ----------------------------------------------------------------- forward
    def _select_queries(self, x, mask_features):
        """The first part of MaskDINODecoder.forward: the flattened memory, the
        two-stage query selection and the mask-enhanced box initialisation."""
        assert len(x) == self.num_feature_levels
        device = x[0].device
        masks = [torch.zeros((s.size(0), s.size(2), s.size(3)), device=device, dtype=torch.bool)
                 for s in x]
        src_flatten, mask_flatten, spatial_shapes = [], [], []
        for i in range(self.num_feature_levels):
            idx = self.num_feature_levels - 1 - i
            bs = x[idx].shape[0]
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)),
                                       spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two-stage query selection and mask-enhanced box initialisation
        output_memory, output_proposals = gen_encoder_output_proposals(
            src_flatten, mask_flatten, spatial_shapes)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        enc_class = self._head("class_embed", True)(output_memory)
        enc_coord = self._head("bbox_embed", True)(output_memory) + output_proposals
        topk = torch.topk(enc_class.max(-1)[0], self.num_queries, dim=1)[1]
        self._topk = topk
        refpoint_embed_undetach = torch.gather(enc_coord, 1, topk.unsqueeze(-1).repeat(1, 1, 4))
        refpoint_embed = refpoint_embed_undetach.detach()
        tgt_undetach = torch.gather(output_memory, 1, topk.unsqueeze(-1).repeat(1, 1, self.hidden_dim))
        outputs_class, outputs_mask, _ = self.forward_prediction_heads(
            tgt_undetach.transpose(0, 1), mask_features, proposal=True)
        tgt = tgt_undetach.detach()
        if self.learn_tgt:
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
        interm_outputs = {"pred_logits": outputs_class, "pred_boxes": refpoint_embed_undetach.sigmoid(),
                          "pred_masks": outputs_mask}
        if self.initialize_box_type != "no":
            flat = outputs_mask.detach().flatten(0, 1)
            h, w = outputs_mask.shape[-2:]
            if self.initialize_box_type == "bitmask":
                boxes = mask_boxes_vectorised(flat > 0)
            else:
                boxes = box_ops.masks_to_boxes(flat > 0)
            refpoint_embed = box_ops.box_xyxy_to_cxcywh(boxes.to(device)) / torch.as_tensor(
                [w, h, w, h], dtype=torch.float, device=device)
            refpoint_embed = inverse_sigmoid(refpoint_embed.reshape(outputs_mask.shape[0],
                                                                    outputs_mask.shape[1], 4))
        return (src_flatten, mask_flatten, spatial_shapes, level_start_index, valid_ratios,
                tgt, refpoint_embed, interm_outputs)

    def forward(self, x, mask_features, masks, targets=None):
        """MaskDINODecoder.forward with the height / offset read-outs and the SAR steps."""
        self._stat_lists = {}
        (src_flatten, mask_flatten, spatial_shapes, level_start_index, valid_ratios,
         tgt, refpoint_embed, interm_outputs) = self._select_queries(x, mask_features)
        predictions_class, predictions_mask, predictions_extra = [], [], []

        attn_mask, mask_dict = None, None
        if self.dn != "no" and self.training:
            assert targets is not None
            input_query_label, input_query_bbox, attn_mask, mask_dict = self.prepare_for_dn(
                targets, None, None, x[0].shape[0])
            if mask_dict is not None:
                tgt = torch.cat([input_query_label, tgt], dim=1)

        outputs_class, outputs_mask, extra = self.forward_prediction_heads(
            tgt.transpose(0, 1), mask_features, self.training, proposal=True)
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_extra.append(extra)
        if self.dn != "no" and self.training and mask_dict is not None:
            refpoint_embed = torch.cat([input_query_bbox, refpoint_embed], dim=1)

        # at layer 0 the matching queries' reference boxes are MaskDINO's boxes
        # of their predicted roof masks (footprint = roof centre + the whole
        # offset); the denoising queries in front of them carry noised
        # roof-union-footprint boxes (centre + half the offset), as every box
        # refined by a layer does
        self._fp0 = None
        if self.initialize_box_type != "no" and self.sar_ctx is not None:
            pad = mask_dict["pad_size"] if mask_dict is not None else 0
            k = refpoint_embed.new_ones(refpoint_embed.shape[1])
            k[:pad] = 0.5
            self._fp0 = k[None, :, None]
        hs, references = self._decode(tgt.transpose(0, 1), src_flatten.transpose(0, 1), mask_flatten,
                                      refpoint_embed.transpose(0, 1), level_start_index,
                                      spatial_shapes, valid_ratios, attn_mask)
        for i, output in enumerate(hs):
            outputs_class, outputs_mask, extra = self.forward_prediction_heads(
                output.transpose(0, 1), mask_features, self.training or (i == len(hs) - 1))
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_extra.append(extra)
        out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
        stacked_extra = {k: torch.stack([e[k] for e in predictions_extra]) for k in EXTRA_KEYS}

        if mask_dict is not None:
            predictions_class, out_boxes, predictions_mask, stacked_extra = self.dn_post_process(
                predictions_class, out_boxes, mask_dict, predictions_mask, stacked_extra)
        elif self.training:
            predictions_class[-1] = predictions_class[-1] + 0.0 * self.label_enc.weight.sum()

        aux = []
        for i in range(len(predictions_class) - 1):
            d = {"pred_logits": predictions_class[i], "pred_masks": predictions_mask[i],
                 "pred_boxes": out_boxes[i]}
            d.update({k: v[i] for k, v in stacked_extra.items()})
            aux.append(d)
        out = {"pred_logits": predictions_class[-1], "pred_masks": predictions_mask[-1],
               "pred_boxes": out_boxes[-1], "aux_outputs": aux, "interm_outputs": interm_outputs}
        out.update({k: v[-1] for k, v in stacked_extra.items()})
        if self.optical_pass and self.optical_pass_active and self.training and self.mode == "fusion":
            # the pass itself runs after the fused criterion (WISHNet.forward), so
            # that nothing of it is alive at the memory peak of the fused mask terms
            out["optical_pass_args"] = (tgt, refpoint_embed, src_flatten, mask_flatten, level_start_index,
                                        spatial_shapes, valid_ratios, attn_mask, mask_dict,
                                        None if self.exchange_pre is None else dict(self.exchange_pre,
                                                                                    topk=self._topk))
        self.stats = {k: torch.stack(v).mean() for k, v in self._stat_lists.items()}
        return out, mask_dict

    def _optical_pass(self, tgt, refpoint_embed, src_flatten, mask_flatten, starts, shapes, valid_ratios,
                      attn_mask, mask_dict, exchange=None):
        """MODEL.WISHNET.OPTICAL_PASS: the decoding once more on the same queries
        (tgt, refpoint_embed (B, Q, *), the denoising ones in front) with every
        SAR step skipped -> {"layers": per decoder layer the matching queries'
        pred_logits, pred_boxes and the EXTRA_KEYS, "dn_layers": the same for the
        denoising queries, or None}.  With ENCODER_EXCHANGE (``exchange``) the
        memory and the matching queries' content come from the optical memory
        before the exchange."""
        if exchange is not None:
            src_flatten, tgt, refpoint_embed = self._pre_exchange_inputs(exchange, mask_flatten, shapes, tgt,
                                                                         refpoint_embed, mask_dict)
        pad = mask_dict["pad_size"] if mask_dict is not None else 0
        hs, refs = self._decode_layers(tgt.transpose(0, 1), src_flatten.transpose(0, 1), mask_flatten,
                                       refpoint_embed.transpose(0, 1), starts, shapes, valid_ratios,
                                       attn_mask, sar=False)
        boxes = self.pred_box(refs, hs)
        layers, dn_layers = [], []
        for i, output in enumerate(hs):
            cls, _, extra = self.forward_prediction_heads(output.transpose(0, 1), None, pred_mask=False)
            d = dict(extra, pred_logits=cls, pred_boxes=boxes[i])
            layers.append({k: v[:, pad:] for k, v in d.items()})
            if pad:
                dn_layers.append({k: v[:, :pad] for k, v in d.items()})
        return {"layers": layers, "dn_layers": dn_layers or None}

    def _pre_exchange_inputs(self, exchange, mask_flatten, shapes, tgt, refpoint_embed, mask_dict):
        """MODEL.WISHNET.ENCODER_EXCHANGE: the optical pass's memory and matching
        queries from the optical memory before the exchange (exchange["levels"],
        [s32, s16, s8]).  The memory is flattened as _select_queries flattens
        it, with gradient (the optical encoder learns to stand on its own); the
        matching queries keep the fused decoding's tokens (exchange["topk"]) and
        initial boxes, their content the encoder output of the memory before the
        exchange at those tokens, without gradient as in the fused decoding.
        The denoising queries are as they are.  -> (memory (B, S, d), tgt,
        refpoint_embed)."""
        x = exchange["levels"]
        src = torch.cat([self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2)
                         for idx in range(self.num_feature_levels - 1, -1, -1)], 1)
        pad = mask_dict["pad_size"] if mask_dict is not None else 0
        idx = exchange["topk"].unsqueeze(-1)
        with torch.no_grad():
            memory, _ = gen_encoder_output_proposals(src, mask_flatten, shapes)
            content = self.enc_output_norm(self.enc_output(torch.gather(memory, 1, idx.repeat(1, 1, self.hidden_dim))))
            boxes = refpoint_embed[:, pad:]
        if self.learn_tgt:          # the queries' content is learned, not taken from the memory
            content = tgt[:, pad:]
        return (src, torch.cat([tgt[:, :pad], content], 1),
                torch.cat([refpoint_embed[:, :pad], boxes.to(refpoint_embed.dtype)], 1))

    @torch.no_grad()
    def forward_gt_queries(self, x, mask_features, gt_boxes, detection=False):
        """For the fixed-set readout (evaluation only): the heights the decoder
        gives the ground-truth buildings when their boxes come in as queries.

        Every image's ground-truth boxes (roof union footprint, normalised
        cxcywh, the training targets' format) enter as one group of denoising
        queries without noise -- content the class embedding, reference the box
        itself -- in front of the image's own matching queries, under
        MaskDINO's denoising attention mask (the matching queries cannot see
        them; they see each other and the matching queries).  This is a second
        decoding of its own, whose outputs no other decoding sees.  In mode
        fusion the SAR reads start as for training's denoising queries: at half
        the predicted offset from the box centre.

        x, mask_features as in forward; gt_boxes [(n_i, 4)] -> per image the
        last layer's heights (n_i,) and offsets (n_i, 2) of those queries.  With
        ``detection`` (the fixed-set detection readings) also, per image, the
        matching queries' last-layer class logits (Q, classes) and mask logits
        (Q, h, w) and the encoder proposals' mask logits (Q, h, w) -- the
        matching queries cannot see the ground-truth ones, so these are the
        image's inference outputs -> (heights, offsets, det)."""
        (src_flatten, mask_flatten, spatial_shapes, level_start_index, valid_ratios,
         tgt, refpoint_embed, interm) = self._select_queries(x, mask_features)
        B, Q, d = tgt.shape
        n = [int(len(b)) for b in gt_boxes]
        pad = max(n + [0])
        if pad == 0 and not detection:
            return [tgt.new_zeros((0,)) for _ in n], [tgt.new_zeros((0, 2)) for _ in n]
        q_lab = tgt.new_zeros((B, pad, d))
        q_box = tgt.new_zeros((B, pad, 4))
        for i, b in enumerate(gt_boxes):
            if n[i]:
                labels = torch.zeros(n[i], dtype=torch.long, device=tgt.device)
                q_lab[i, :n[i]] = self.label_enc(labels)
                q_box[i, :n[i]] = inverse_sigmoid(b.to(q_box.dtype))
        attn_mask = torch.zeros((pad + Q, pad + Q), dtype=torch.bool, device=tgt.device)
        attn_mask[pad:, :pad] = True          # the matching queries cannot see them
        tgt = torch.cat([q_lab, tgt], dim=1)
        refpoint_embed = torch.cat([q_box, refpoint_embed], dim=1)

        self._stat_lists = {}
        self._fp0 = None
        if self.initialize_box_type != "no" and self.sar_ctx is not None:
            k = refpoint_embed.new_ones(refpoint_embed.shape[1])
            k[:pad] = 0.5
            self._fp0 = k[None, :, None]
        hs, _ = self._decode(tgt.transpose(0, 1), src_flatten.transpose(0, 1), mask_flatten,
                             refpoint_embed.transpose(0, 1), level_start_index,
                             spatial_shapes, valid_ratios, attn_mask)
        if pad:
            last = self.decoder_norm(hs[-1][:, :pad])            # as forward_prediction_heads reads it
            h, _ = self.height_embed(last)
            off = self.offset_embed(last)
            heights, offsets = [h[i, :n[i]] for i in range(B)], [off[i, :n[i]] for i in range(B)]
        else:
            heights, offsets = [tgt.new_zeros((0,)) for _ in n], [tgt.new_zeros((0, 2)) for _ in n]
        if not detection:
            return heights, offsets
        cls, masks, _ = self.forward_prediction_heads(hs[-1][:, pad:].transpose(0, 1), mask_features, True)
        det = [{"logits": cls[i], "masks": masks[i], "prop": interm["pred_masks"][i]} for i in range(B)]
        return heights, offsets, det
