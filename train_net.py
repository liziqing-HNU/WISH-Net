# -*- coding: utf-8 -*-
"""WISH-Net training / evaluation entry point.

Launch from the repository root:

    python train_net.py --num-gpus 1 --config-file configs/bdh/wishnet.yaml
    python train_net.py --num-gpus 1 --eval-only --config-file <cfg> MODEL.WEIGHTS <model_best_h.pth>

The trainer is detectron2's DefaultTrainer with AdamW, full-model gradient
clipping and the optical backbone at 0.1x the base learning rate (the SAR side
at the full base rate), on the fixed schedule of SOLVER.STEPS and MAX_ITER.
With SOLVER.EMA the validations, model_best.pth (the lowest Hungarian height
MAE), model_best_h.pth (the weight-selection rule) and model_final.pth hold
the moving average of the weights (wishnet/engine/ema.py).  TensorBoard gets
the curves worth watching mirrored under key/.  A file named STOP in
OUTPUT_DIR ends the training after the current step, as at the planned end.
"""
try:
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except Exception:
    pass

import itertools
import logging
import os
import weakref
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import get_detection_dataset_dicts
from detectron2.data.samplers import TrainingSampler
from detectron2.engine import DefaultTrainer, HookBase, default_argument_parser, default_setup, launch
from detectron2.engine import hooks as d2hooks
from detectron2.evaluation import verify_results
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.utils.logger import setup_logger

from maskdino import add_maskdino_config

from wishnet import BDHEvaluator, WISHDatasetMapper, add_wishnet_config, check_config, register_bdh
from wishnet.engine import (BestCheckpointHook, EMACheckpointer, EMAHook, ModelEMA, RuleBestCheckpointHook,
                             StopFileHook)

logger = logging.getLogger("wishnet")

# the metric of model_best.pth
BEST_METRIC = "bdh/hungarian/MAE"
# the fixed-set heights mirrored as key curves (MODEL.WISHNET.FIXED_SET_HEIGHT)
FIXED_KEYS = ("train_mae", "val_mae", "gap", "val_mae_50", "train_mae_optpass", "val_mae_optpass")
# the fixed-set detection readings mirrored as key curves (FIXED_SET_HEIGHT.DETECTION)
FIXDET_KEYS = (("train_f1", "val_f1", "train_ap50", "val_ap50", "train_prop_recall", "val_prop_recall")
               + tuple("val_%s_d_%s" % (m, s) for m in ("f1", "prop_recall")
                       for s in ("zero_gf3", "zero_ins", "zero", "opt")))
# SAR_MEAN_READOUT.PER_MODALITY: the heights with one modality zeroed, as key curves
MODALITY_KEYS = ("sar_zero_gf3/val_dmae", "sar_zero_ins/val_dmae", "sar_zero_gf3/ratio_cal",
                 "sar_zero_ins/ratio_cal")


class KeyCurvesHook(HookBase):
    """Mirror the curves worth watching under key/ in TensorBoard."""

    KEYS = [
        "total_loss", "lr",
        # last decoder layer
        "loss_ce", "loss_mask", "loss_dice", "loss_bbox", "loss_giou",
        "loss_height_l1", "loss_height_dc", "loss_height_ce", "loss_offset",
        # the proposal chain, before any SAR step: the encoder proposals' masks,
        # the initial queries' boxes (from those masks) and heights
        "loss_dice_interm", "loss_bbox_0", "loss_height_l1_0",
        # MODEL.WISHNET.OPTICAL_PASS: the last layer's terms of the SAR-free pass
        # (weighted, WEIGHT times the fused terms' weight)
        "loss_height_l1_optpass", "loss_ce_optpass", "loss_bbox_optpass",
        # readable units on the matched training buildings
        "height/train_mae_m", "offset/train_epe_px",
        # the gated injection into the optical queries: mean gate, share of
        # closed channels (< 0.05), size of the injection next to the query
        "fusion/opt_gate", "fusion/opt_gate_closed", "fusion/opt_inject_rel",
        # gradients before the clip
        "grad/total_norm_preclip", "grad/clip_coef",
        "grad/sar_backbone_norm", "grad/sar_encoder_norm", "grad/fusion_norm", "grad/proposal_heads_norm",
        # SOLVER.EMA: the decay of the step, ||average - weights|| / ||weights||
        "ema/decay", "ema/rel_dist",
        # MODEL.WISHNET.ENCODER_EXCHANGE: the coefficients' root mean squares and the
        # written updates next to the tokens they go into; gamma / inject_rel into
        # the optical memory (from GF-3, from InSAR), beta / rev_rel into GF-3 and InSAR
        "exchange/gamma_gf3", "exchange/gamma_ins", "exchange/inject_rel_gf3", "exchange/inject_rel_ins",
        "exchange/beta_gf3", "exchange/beta_ins", "exchange/rev_rel_gf3", "exchange/rev_rel_ins",
        "grad/exchange_norm",
    ]

    def after_step(self):
        storage = self.trainer.storage
        latest = storage.latest()
        for k in self.KEYS:
            if k in latest:
                storage.put_scalar("key/" + k.split("/")[-1], latest[k][0], smoothing_hint=False)


class GateWarmupHook(HookBase):
    """MODEL.WISHNET.FUSION.GATE_WARMUP_ITERS: for the first N iterations the
    gates (W_g, b_g) and their value projections W_v take no gradient, so they
    stay at their initialisation (g = 0.5, the injection at INJECT_REL of the
    query) while the SAR side learns; from iteration N on they train.
    Parameters without a gradient are skipped by the optimiser (no step, no
    weight decay).  Works across a resume: the state follows the iteration."""

    def __init__(self, n_iters):
        self.n_iters = int(n_iters)
        self.frozen = None

    def _params(self):
        model = self.trainer.model
        model = getattr(model, "module", model)
        pred = model.sem_seg_head.predictor
        for fuse in pred.sar_fuse.values():
            yield from fuse.gate.parameters()
            yield from fuse.value.parameters()

    def before_step(self):
        freeze = self.trainer.iter < self.n_iters
        if freeze == self.frozen:
            return
        params = list(self._params())
        for p in params:
            p.requires_grad_(not freeze)
            if freeze:
                p.grad = None
        self.frozen = freeze
        logger.info("[GateWarmupHook] iter %d: %d gate / W_v tensors %s",
                    self.trainer.iter, len(params), "held at their initialisation until iter %d"
                    % self.n_iters if freeze else "released")


def _log_grad_norms(param_groups, clip_value):
    """Pre-clip gradient norms, in total and per module group."""
    try:
        storage = get_event_storage()
    except Exception:
        return
    grads = {}
    for g in param_groups:
        for p in g["params"]:
            if p.grad is not None:
                grads.setdefault(g.get("grad_tag", "other"), []).append(p.grad.detach())
    if not grads:
        return
    tags = sorted(grads)
    sq = torch.stack([torch.stack(torch._foreach_norm(grads[t])).float().pow(2).sum()
                      for t in tags]).cpu()
    total = float(sq.sum().sqrt())
    storage.put_scalar("grad/total_norm_preclip", total)
    storage.put_scalar("grad/clip_coef", min(1.0, clip_value / max(total, 1e-12)))
    for t, v in zip(tags, sq.tolist()):
        storage.put_scalar("grad/%s_norm" % t, v ** 0.5)


class Trainer(DefaultTrainer):
    # the gradient-norm group of a parameter: first matching prefix wins
    GRAD_TAGS = (
        ("sar_branch.backbone", "sar_backbone"),
        ("sar_branch.encoder", "sar_encoder"),
        ("sem_seg_head.predictor.sar_", "fusion"),
        ("sem_seg_head.predictor.proposal_heads", "proposal_heads"),
        ("sem_seg_head.predictor", "decoder"),
        ("sem_seg_head.pixel_decoder", "opt_encoder"),
        ("exchange.", "exchange"),
        ("backbone", "opt_backbone"),
    )

    def __init__(self, cfg):
        super().__init__(cfg)
        self._install_batched_metrics()

    def train(self):
        """detectron2's loop as a while, so the STOP file (it lowers
        trainer.max_iter) takes effect."""
        tlog = logging.getLogger("detectron2.engine.train_loop")
        tlog.info("Starting training from iteration {}".format(self.start_iter))
        self.iter = self.start_iter
        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                while self.iter < self.max_iter:
                    self.before_step()
                    self.run_step()
                    self.after_step()
                    self.iter += 1
            except Exception:
                tlog.exception("Exception during training:")
                raise
            finally:
                self.after_train()
        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def _install_batched_metrics(self):
        """One device-to-host copy for all loss terms instead of one each."""
        inner = getattr(self, "_trainer", None)
        if inner is None or not hasattr(inner, "_write_metrics"):
            return
        original = inner._write_metrics

        def batched(loss_dict, data_time, prefix="", iter=None):
            if loss_dict:
                keys = list(loss_dict)
                vals = torch.stack([loss_dict[k].detach() for k in keys]).cpu()
                loss_dict = {k: vals[i] for i, k in enumerate(keys)}
            return original(loss_dict, data_time, prefix, iter)

        inner._write_metrics = batched

    def build_hooks(self):
        # SOLVER.EMA: the average, and the checkpointer that keeps it, in place
        # before DefaultTrainer.build_hooks hands the checkpointer to its hooks
        self.ema = None
        if self.cfg.SOLVER.EMA.ENABLED:
            e = self.cfg.SOLVER.EMA
            self.ema = ModelEMA(self.model, e.DECAY, e.WARMUP)
            self.checkpointer = EMACheckpointer(self.model, self.cfg.OUTPUT_DIR, ema=self.ema,
                                                trainer=weakref.proxy(self))
        state_fn = self.ema.model_state if self.ema is not None else None
        hooks_list = super().build_hooks()
        # a STOP file in the run directory ends the training after the current
        # step; before EvalHook and the periodic checkpointer, which then see
        # the lowered max_iter at the same step
        hooks_list.insert(1, StopFileHook(self.cfg.OUTPUT_DIR))
        hooks_list.insert(len(hooks_list) - 1, KeyCurvesHook())
        if self.cfg.MODEL.WISHNET.FUSION.GATE_WARMUP_ITERS > 0 and self.cfg.MODEL.WISHNET.MODE == "fusion":
            # before the step's forward: the first hook after IterationTimer
            hooks_list.insert(1, GateWarmupHook(self.cfg.MODEL.WISHNET.FUSION.GATE_WARMUP_ITERS))
        if self.cfg.SOLVER.KEEP_BEST and comm.is_main_process():
            hooks_list.insert(len(hooks_list) - 1, BestCheckpointHook(
                BEST_METRIC, mode="min", checkpointer=self.checkpointer, state_fn=state_fn))
        if self.cfg.SOLVER.KEEP_BEST_H and comm.is_main_process():
            hooks_list.insert(len(hooks_list) - 1, RuleBestCheckpointHook(state_fn=state_fn))
        if self.cfg.SOLVER.KEEP_BEST_HEIGHT and comm.is_main_process():
            # model_best_height by the fixed-set height MAE
            hooks_list.insert(len(hooks_list) - 1, BestCheckpointHook(
                "fixed/val_mae", mode="min", checkpointer=self.checkpointer, name="model_best_height",
                state_fn=state_fn, record=("fixed/val_mae_optpass",)))
        # the periodic checkpointer goes after evaluation and the best-checkpoint hooks
        for i, h in enumerate(hooks_list):
            if isinstance(h, d2hooks.PeriodicCheckpointer):
                h.max_to_keep = self.cfg.SOLVER.CHECKPOINT_MAX_KEEP or None
                hooks_list.pop(i)
                hooks_list.insert(len(hooks_list) - 1, h)
                break
        g = self.cfg.MODEL.WISHNET
        sar_mean = g.MODE == "fusion" and g.SAR_MEAN_READOUT.ENABLED
        if (sar_mean or g.FIXED_SET_HEIGHT.ENABLED) and comm.is_main_process():
            # the optional fixed-set readout after every validation, of the same
            # weights (inside the moving average's swap below)
            from wishnet.evaluation.readout import ValidationReadout
            self.readout = ValidationReadout(self.cfg)
            for h in hooks_list:
                if isinstance(h, d2hooks.EvalHook):
                    h._func = self._with_readout(h._func)
                    break
        if self.ema is not None:
            # the average is updated right before EvalHook, which evaluates it
            for i, h in enumerate(hooks_list):
                if isinstance(h, d2hooks.EvalHook):
                    h._func = self._with_ema(h._func)
                    hooks_list.insert(i, EMAHook(self.ema, self.cfg.SOLVER.EMA.LOG_PERIOD))
                    break
            else:
                raise RuntimeError("SOLVER.EMA: no EvalHook to evaluate the average with")
        return hooks_list

    def _with_readout(self, func):
        """An evaluation followed by the fixed-set readout; a monitor: a failure
        is logged and never stops the run."""
        def evaluate():
            res = func()
            try:
                values = self.readout.run(self.model, int(self.iter))
                self.storage.put_scalars(smoothing_hint=False, **values)
                fixed = [k for k in FIXED_KEYS if "fixed/" + k in values]
                bands = sorted(k[len("fixed/"):] for k in values
                               if k.startswith("fixed/") and "_mae_b" in k)
                if fixed or bands:
                    self.storage.put_scalars(smoothing_hint=False,
                                             **{"key/fixed_" + k: values["fixed/" + k] for k in fixed + bands})
                if fixed:
                    logger.info("[fixed] iter %d: %s", self.iter,
                                ", ".join("%s %.3f" % (k, values["fixed/" + k]) for k in fixed))
                fixdet = [k for k in FIXDET_KEYS if "fixdet/" + k in values]
                modal = [k for k in MODALITY_KEYS if k in values]
                if fixdet or modal:
                    self.storage.put_scalars(smoothing_hint=False,
                                             **{"key/fixdet_" + k: values["fixdet/" + k] for k in fixdet},
                                             **{"key/" + k.replace("/", "_"): values[k] for k in modal})
            except Exception:
                logger.exception("[readout] the readout failed; training goes on")
            return res
        return evaluate

    def _with_ema(self, func):
        """An evaluation of the moving average; the training weights come back after it."""
        def evaluate():
            with self.ema.swapped():
                return func()
        return evaluate

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        # hand the training step's cached blocks back before evaluating: the
        # evaluation keeps a tile's 2048 px masks on the GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return super().test(cfg, model, evaluators)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_" + dataset_name)
        e = cfg.BDH_EVAL
        return BDHEvaluator(
            dataset_name, output_dir=output_folder, height_bins=e.HEIGHT_BINS,
            iou_thresh=e.IOU_THRESH, score_thresh_f1=e.SCORE_THRESH_F1,
            score_thresh_mae=e.SCORE_THRESH_MAE, min_recall=e.MIN_RECALL, mae_tol=e.MAE_TOL,
            max_dets=e.MAX_DETS, vol_min_height=e.VOL_MIN_HEIGHT,
            ignore_truncated=e.IGNORE_TRUNCATED, ignore_regions=e.IGNORE_REGIONS)

    @classmethod
    def build_train_loader(cls, cfg):
        dataset = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS)
        return build_detection_train_loader(
            dataset=dataset, mapper=WISHDatasetMapper(cfg, True),
            sampler=TrainingSampler(len(dataset), shuffle=True, seed=cfg.SEED),
            total_batch_size=cfg.SOLVER.IMS_PER_BATCH, num_workers=cfg.DATALOADER.NUM_WORKERS)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=WISHDatasetMapper(cfg, False))

    @classmethod
    def _grad_tag(cls, name):
        for prefix, tag in cls.GRAD_TAGS:
            if name.startswith(prefix):
                return tag
        return "other"

    @classmethod
    def build_optimizer(cls, cfg, model):
        norm_types = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                      torch.nn.SyncBatchNorm, torch.nn.GroupNorm, torch.nn.InstanceNorm1d,
                      torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d, torch.nn.LayerNorm,
                      torch.nn.LocalResponseNorm)
        base_lr = cfg.SOLVER.BASE_LR
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        counts = {}
        for module_name, module in model.named_modules():
            for pname, value in module.named_parameters(recurse=False):
                if not value.requires_grad or value in memo:
                    continue
                memo.add(value)
                lr, wd = base_lr, cfg.SOLVER.WEIGHT_DECAY
                if module_name.startswith("backbone."):            # the optical ResNet-50
                    lr = base_lr * cfg.SOLVER.BACKBONE_MULTIPLIER
                if isinstance(module, norm_types):
                    wd = cfg.SOLVER.WEIGHT_DECAY_NORM
                if isinstance(module, torch.nn.Embedding):
                    wd = cfg.SOLVER.WEIGHT_DECAY_EMBED
                tag = cls._grad_tag(module_name + "." + pname)
                counts[(tag, lr)] = counts.get((tag, lr), 0) + value.numel()
                params.append({"params": [value], "lr": lr, "weight_decay": wd, "grad_tag": tag})
        for (tag, lr), n in sorted(counts.items()):
            logger.info("[WISH-Net] %-13s lr %.2e  %.2f M parameters", tag, lr, n / 1e6)
        clip = cfg.SOLVER.CLIP_GRADIENTS

        class ClippedAdamW(torch.optim.AdamW):
            def step(self, closure=None):
                _log_grad_norms(self.param_groups, clip.CLIP_VALUE)
                if clip.ENABLED and clip.CLIP_VALUE > 0:
                    torch.nn.utils.clip_grad_norm_(
                        itertools.chain(*[g["params"] for g in self.param_groups]), clip.CLIP_VALUE)
                super().step(closure=closure)

        if cfg.SOLVER.OPTIMIZER != "ADAMW" or clip.CLIP_TYPE != "full_model":
            raise ValueError("WISH-Net trains with AdamW and full-model clipping")
        return ClippedAdamW(params, base_lr)


def setup(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    add_wishnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    check_config(cfg)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="wishnet")
    sar_root = cfg.MODEL.WISHNET.SAR.ROOT or os.path.join(cfg.DATASETS.BDH_ROOT, "sar_gf3")
    register_bdh(cfg.DATASETS.BDH_ROOT, cfg.DATASETS.BDH_PREPARED,
                 cfg.INPUT.EVAL_CROP_SIZE or cfg.INPUT.CROP_SIZE, cfg.INPUT.MIN_HEIGHT,
                 cfg.DATASETS.BDH_ANN_DIR, sar_root)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank,
           dist_url=args.dist_url, args=(args,))
