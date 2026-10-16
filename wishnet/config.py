# -*- coding: utf-8 -*-
"""WISH-Net configuration keys.

Everything WISH-Net adds to MaskDINO lives under ``MODEL.WISHNET``; the input,
dataset, evaluation and schedule keys are in INPUT, DATASETS, BDH_EVAL and
SOLVER.  The defaults are the paper's recipe; a configuration chooses the
model with

    MODEL.WISHNET.MODE                 "opt" (optical only) | "fusion" (optical + SAR)
    MODEL.WISHNET.INSAR.ENABLED        the InSAR input of the SAR memory (mode fusion)
    MODEL.WISHNET.ENCODER_EXCHANGE.ENABLED   the two-way exchange at the encoder stage
    MODEL.WISHNET.OPTICAL_PASS.WEIGHT  the optical-only auxiliary pass (mode fusion)
"""
from detectron2.config import CfgNode as CN

# the training buildings per height bin smoothed with a Gaussian of two bins
# (LDS), to the power -0.25, scaled to mean 1 over the training buildings and
# capped at 3 (tools/height_weights.py), from the original split's training
# labels before the review (annotations_fixed/instances_train.json); the runs
# on the reviewed labels kept this table (from those labels it would differ by
# at most 0.02)
BDH_BIN_WEIGHTS = [
    1.0790, 0.9687, 0.8942, 0.8470, 0.8233, 0.8227, 0.8460, 0.8946, 0.9682, 1.0616,
    1.1602, 1.2421, 1.2934, 1.3169, 1.3252, 1.3296, 1.3383, 1.3558, 1.3833, 1.4189,
    1.4605, 1.5075, 1.5609, 1.6192, 1.6780, 1.7339, 1.7913, 1.8612, 1.9533, 2.0660,
    2.1832, 2.2880, 2.3871, 2.5104, 2.6878, 2.9376, 3.0000, 3.0000, 3.0000, 3.0000,
    3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000,
    3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000]


def add_wishnet_config(cfg):
    # ------------------------------------------------------------------ input
    # the training window's side, px of the tile: 2048 = the whole BDH tile
    cfg.INPUT.CROP_SIZE = 2048
    # the evaluation window's side (0 = CROP_SIZE): the prepared crops of that size
    cfg.INPUT.EVAL_CROP_SIZE = 2048
    # tries to find a training window that holds a building
    cfg.INPUT.CROP_RETRY = 10
    # the window is resized by this factor before it goes in, optical and SAR
    # alike, with the annotations scaled to match; the outputs are mapped back to
    # the window's own size, so the evaluation stays at full resolution.  0 =
    # off.  0.5: the whole 2048 tile read at 1024 (0.83 m/px)
    cfg.INPUT.TILE_SCALE = 0.5
    cfg.INPUT.MIN_INST_AREA = 32.0      # px^2 of roof left inside the window
    cfg.INPUT.MIN_INST_FRAC = 0.1       # share of the roof left inside the window
    # a training building whose roof comes within this many px of the window's
    # edge counts as cut, as one with less than 99.5 % of its roof inside the
    # window or its footprint beyond it: it is matched as usual, without height
    # or offset terms.  Training windows only; the evaluation crops keep the
    # truncation they were prepared with
    cfg.INPUT.TRUNC_PX = 1.0
    # radiometric jitter of the optical image (training only): brightness and
    # contrast factors 1 + U(-value, value); 0 = off
    cfg.INPUT.RGB_BRIGHTNESS = 0.10
    cfg.INPUT.RGB_CONTRAST = 0.10
    # training only: the whole sample flipped or rotated, every modality and
    # annotation alike (mapper.GEO_TRANSFORMS): "all8" every flip and 90-degree
    # rotation of the square, drawn uniformly; "" off
    cfg.INPUT.GEO_AUG = "all8"
    # bd_height below this = floor count never recorded: such a building is
    # class / box / mask ground truth without height or offset terms
    cfg.INPUT.MIN_HEIGHT = 3.0

    # --------------------------------------------------------------- datasets
    # BDH: images/{train,val,test}/<tile>.tif and <BDH_ANN_DIR>/instances_train.json;
    # the evaluation crops instances_{val,test}_crop<EVAL_CROP_SIZE>.json in
    # BDH_PREPARED (datasets/README.md)
    cfg.DATASETS.BDH_ROOT = "datasets/bdh"
    cfg.DATASETS.BDH_PREPARED = "datasets/bdh/prepared_v2"
    cfg.DATASETS.BDH_ANN_DIR = "annotations_v2"

    # ------------------------------------------------------------------ model
    cfg.MODEL.WISHNET = CN()
    # "opt":    MaskDINO + height / offset heads, no SAR
    # "fusion": the SAR branch, read by every decoder layer
    cfg.MODEL.WISHNET.MODE = "opt"
    # optical ResNet-50: torchgeo's FMOW_RGB_GASSL weights in detectron2 format
    # (tools/convert_torchgeo_r50.py)
    cfg.MODEL.WISHNET.OPT_BACKBONE_WEIGHTS = "pretrained/R-50_torchgeo_fmow_rgb_gassl.pkl"
    # the optical encoder's (pixel decoder's) Transformer layers under activation
    # checkpointing in training: the same numbers, its activations recomputed in
    # the backward instead of kept (about 0.27 GB per layer and image for a 1024
    # input).  False: off
    cfg.MODEL.WISHNET.OPT_ENCODER_CHECKPOINT = True
    # the optical pass (training only, mode fusion): after the fused decoding the
    # decoder runs once more on the same queries with every SAR step skipped --
    # the optical computation with the same weights and heads -- and WEIGHT times
    # its class, box, height and offset terms are added, every decoder layer's
    # (the pass predicts no masks); the matching queries are scored against the
    # fused pass's matching of the same layer, the denoising queries against
    # their own boxes.  The optical part has to answer on its own and cannot
    # hand its height cues over to the SAR; the SAR side gets no gradient from
    # these terms.  0 = off
    cfg.MODEL.WISHNET.OPTICAL_PASS = CN()
    cfg.MODEL.WISHNET.OPTICAL_PASS.WEIGHT = 0.0
    # recompute the pass's decoder layers in the backward instead of keeping
    # their activations (each layer's value projection of the optical memory is
    # about 90 MB per image)
    cfg.MODEL.WISHNET.OPTICAL_PASS.CHECKPOINT = True
    # the fixed-set readout (optional, a monitor: the training computation is
    # the same either way): at every validation the ground-truth buildings of
    # fixed training and validation tiles (every n-th by file name, 0 = all) are
    # decoded as queries; their height MAE goes to fixed/train_mae,
    # fixed/val_mae, fixed/gap (val - train) and fixed/val_mae_50 (ground truth
    # >= 50 m): heights on a fixed set of buildings, free of the recall
    # (wishnet/evaluation/readout.py)
    cfg.MODEL.WISHNET.FIXED_SET_HEIGHT = CN()
    cfg.MODEL.WISHNET.FIXED_SET_HEIGHT.ENABLED = False
    # per height band of the ground truth as well (3-10, 10-20, 20-30, 30-50,
    # 50-100 and >= 100 m): per band the buildings, the MAE and the signed bias
    cfg.MODEL.WISHNET.FIXED_SET_HEIGHT.BANDS = False
    # the detection readings of the same decodings: the recall of the encoder
    # proposals' masks, the F1 at the best score threshold and the AP50 of the
    # last layer's masks, at IoU 0.5 on the stride-4 grid; with the SAR readout
    # also their changes with the readout's other SAR states
    cfg.MODEL.WISHNET.FIXED_SET_HEIGHT.DETECTION = False
    # the detection readings only at the validations every this many iterations
    # (and the last); 0 = at every validation
    cfg.MODEL.WISHNET.FIXED_SET_HEIGHT.DETECTION_PERIOD = 0
    # the SAR readout of the fixed set (optional, mode fusion): the same
    # decodings with the SAR memory replaced by its mean over the training
    # tiles, and with a zero SAR input; the height MAE changes go to sar_mean/*
    # and sar_zero/*, training against validation tiles
    cfg.MODEL.WISHNET.SAR_MEAN_READOUT = CN()
    cfg.MODEL.WISHNET.SAR_MEAN_READOUT.ENABLED = False
    cfg.MODEL.WISHNET.SAR_MEAN_READOUT.TRAIN_TILES = 32
    cfg.MODEL.WISHNET.SAR_MEAN_READOUT.VAL_TILES = 128
    cfg.MODEL.WISHNET.SAR_MEAN_READOUT.WITH_ZERO = True
    # with INSAR: also GF-3 alone and InSAR alone zeroed (sar_zero_gf3/*, sar_zero_ins/*)
    cfg.MODEL.WISHNET.SAR_MEAN_READOUT.PER_MODALITY = False

    # ---------------------------------------------------------------- SAR
    cfg.MODEL.WISHNET.SAR = CN()
    # folder holding {train,val}/<tile>.tif of the GF-3 HH product
    # (single-look sigma0 in dB on the tile's grid, float32, NaN = no data);
    # "" = BDH_ROOT/sar_gf3
    cfg.MODEL.WISHNET.SAR.ROOT = ""
    # z-score of sigma0 dB: the statistics of the product over the training tiles
    # of BDH-RGB-SAR's geographic split (tools/gf3/tile_summary_gf3.py), kept for
    # the original split
    cfg.MODEL.WISHNET.SAR.MEAN = -8.67
    cfg.MODEL.WISHNET.SAR.STD = 6.79
    # the SAR window is area-resampled to this side before the backbone
    cfg.MODEL.WISHNET.SAR.INPUT_PX = 1024
    # the zeroing evaluations: replace the standardised GF-3 input by zeros
    cfg.MODEL.WISHNET.SAR.ZERO = False
    # SAR speckle resampling, training only, GF-3 only (mapper.speckle_resample):
    # with probability PROB a training sample's GF-3 window gets new speckle --
    # the linear Gaussian local mean with MEAN_SIGMA_M times mean-1 Gamma(ENL,
    # 1/ENL) noise drawn every NATIVE_M_X along x (range) and NATIVE_M_Y along y
    # (azimuth), random phase, enlarged with UPSAMPLE and normalised by its own
    # local mean, so that only fluctuations below MEAN_SIGMA_M change; pixels
    # whose intensity blurred with KEEP_SIGMA_M is above BRIGHT_DB (strong
    # scatterers) keep their own, and the local mean leaves them out
    cfg.MODEL.WISHNET.SAR.SPECKLE = CN()
    cfg.MODEL.WISHNET.SAR.SPECKLE.ENABLED = False
    cfg.MODEL.WISHNET.SAR.SPECKLE.PROB = 0.5
    cfg.MODEL.WISHNET.SAR.SPECKLE.ENL = 0.38
    cfg.MODEL.WISHNET.SAR.SPECKLE.NATIVE_M_X = 2.69
    cfg.MODEL.WISHNET.SAR.SPECKLE.NATIVE_M_Y = 3.10
    cfg.MODEL.WISHNET.SAR.SPECKLE.MEAN_SIGMA_M = 5.0
    cfg.MODEL.WISHNET.SAR.SPECKLE.KEEP_SIGMA_M = 1.0
    cfg.MODEL.WISHNET.SAR.SPECKLE.UPSAMPLE = "bilinear"
    cfg.MODEL.WISHNET.SAR.SPECKLE.BRIGHT_DB = 7.6
    # basic-block ResNet-18 with GroupNorm(32), every stage used; ImageNet
    # weights with the RGB stem kernels summed onto the one SAR band and the
    # batch-norm statistics, re-estimated on GF-3 training windows, folded into
    # the convolutions and the GroupNorm scale and shift (tools/sar_backbone/)
    cfg.MODEL.WISHNET.SAR.DEPTH = 18
    cfg.MODEL.WISHNET.SAR.WEIGHTS = "pretrained/R-18_imagenet_sar1_adabn_gn.pkl"
    # SAR Transformer encoder: MaskDINO's multi-scale deformable encoder over
    # res3-res5 (and the InSAR levels) with pre-norm layers and a final
    # LayerNorm, res2 joined through the FPN lateral at stride 4
    cfg.MODEL.WISHNET.SAR.ENC_LAYERS = 6
    cfg.MODEL.WISHNET.SAR.ENC_DIM_FEEDFORWARD = 1024

    # ----------------------------------------------------------------- InSAR
    # an interferometric stack over the tiles, as channels of one tile per BDH
    # tile (BDH: Sentinel-1A ascending track 142, VV, eight dates 2023-06..08,
    # the seven consecutive 12-day pairs as the real and imaginary parts of the
    # high-passed coherence, the mean coherence, sigma0 and the amplitude
    # dispersion; 128 x 128 px, 6.63 m).  Its own encoder (ResNet-18 res2-res4,
    # a stride-1 stem; outputs at 13 m and 26 m), its two levels joined to the
    # GF-3 levels in one SAR Transformer encoder, read by the decoder's SAR
    # reads with anchors along the InSAR sensor's own direction
    cfg.MODEL.WISHNET.INSAR = CN()
    cfg.MODEL.WISHNET.INSAR.ENABLED = False
    # {train,val}/<GF-3 sar_file_name> and the product's metadata META
    # (channel names, per-channel statistics); "" = BDH_ROOT/s1_insar_asc
    cfg.MODEL.WISHNET.INSAR.ROOT = ""
    cfg.MODEL.WISHNET.INSAR.META = "S1_INSAR_ASC.json"
    # the statistics in META that standardise the channels (train_orig: the
    # original split's training tiles, tools/s1insar/tile_s1insar.py)
    cfg.MODEL.WISHNET.INSAR.NORM_KEY = "train_orig"
    # channel names to use, [] = all
    cfg.MODEL.WISHNET.INSAR.CHANNELS = []
    cfg.MODEL.WISHNET.INSAR.INPUT_PX = 128
    # ImageNet ResNet-18 checkpoint: only its res2-res4 convolution weights are
    # loaded (the stem is new, the GroupNorms start at scale 1, shift 0)
    cfg.MODEL.WISHNET.INSAR.WEIGHTS = "pretrained/R-18_imagenet_sar1_adabn.pkl"
    # direction towards the InSAR sensor, degrees clockwise from north; the
    # anchors of the InSAR levels run along it (Sentinel-1 ascending,
    # right-looking: west)
    cfg.MODEL.WISHNET.INSAR.TOWARD_SENSOR_DEG = 260.3
    # the zeroing evaluations: replace the standardised InSAR input by zeros
    cfg.MODEL.WISHNET.INSAR.ZERO = False

    # -------------------------------------------------------------- fusion
    cfg.MODEL.WISHNET.FUSION = CN()
    # heads of the SAR reads (decoder and exchange)
    cfg.MODEL.WISHNET.FUSION.N_HEADS = 8
    # channel gate sigmoid(W [q; m] + b): b = 0 starts it half open
    cfg.MODEL.WISHNET.FUSION.GATE_BIAS = 0.0
    # the injection's initial size relative to the query (per element)
    cfg.MODEL.WISHNET.FUSION.INJECT_REL = 0.1
    # gate warm-up: for this many iterations the gates (W_g, b_g) and their value
    # projections W_v stay at their initialisation, so g = 0.5 and the injection
    # stays at INJECT_REL of the query while the SAR side learns; the fusion can
    # shut SAR only afterwards.  0 = off
    cfg.MODEL.WISHNET.FUSION.GATE_WARMUP_ITERS = 0

    # ------------------------------------------------------ encoder exchange
    # the two-way exchange at the encoder stage (modeling/exchange.py, mode
    # fusion): after the optical encoder's Transformer and the GF-3 / InSAR
    # encoder, before the optical FPN and the query selection, the optical
    # tokens of s32, s16 and s8 read GF-3 (s32, s16, s8) and InSAR (i32, i16)
    # with the range reads, each written back times its own per-channel
    # coefficient starting at 0; the GF-3 and InSAR tokens of those levels read
    # the optical memory at the mirrored anchors, written back times a
    # per-channel coefficient starting at REVERSE_INIT.  The query selection, the
    # mask features and the decoder then take the memories after the exchange;
    # the optical pass decodes from the optical memory before it.  False: off
    cfg.MODEL.WISHNET.ENCODER_EXCHANGE = CN()
    cfg.MODEL.WISHNET.ENCODER_EXCHANGE.ENABLED = False
    cfg.MODEL.WISHNET.ENCODER_EXCHANGE.REVERSE_INIT = 0.1
    # the block under activation checkpointing in training
    cfg.MODEL.WISHNET.ENCODER_EXCHANGE.CHECKPOINT = True

    # -------------------------------------------------------- range anchors
    # the SAR reads' anchors: metres from the footprint centre along the
    # direction towards the sensor (< 0 the shadow side, 0 the dihedral line,
    # > 0 the layover band); every head samples around each anchor on every
    # level with learned offsets and weights
    cfg.MODEL.WISHNET.RANGE = CN()
    cfg.MODEL.WISHNET.RANGE.ANCHORS_M = [-60.0, -30.0, 0.0, 15.0, 30.0, 50.0, 80.0, 120.0]
    # direction towards the GF-3 sensor, degrees clockwise from north (BDH:
    # descending and right-looking over Beijing, east-south-east)
    cfg.MODEL.WISHNET.RANGE.TOWARD_SENSOR_DEG = 100.2
    # pixel size of the tile's grid (m), onto which the SAR window is resampled;
    # with INPUT.TILE_SCALE the window goes in coarser and the anchors are
    # converted with GSD_M / TILE_SCALE (0.83 m for the whole tile at 1024)
    cfg.MODEL.WISHNET.RANGE.GSD_M = 0.4142

    # -------------------------------------------------------------- height
    cfg.MODEL.WISHNET.HEIGHT = CN()
    cfg.MODEL.WISHNET.HEIGHT.INIT = 25.0          # every query starts here (m)
    cfg.MODEL.WISHNET.HEIGHT.LOSS_SCALE = 10.0    # metres per unit of the L1 term
    cfg.MODEL.WISHNET.HEIGHT.L1_WEIGHT = 2.0
    cfg.MODEL.WISHNET.HEIGHT.DC_WEIGHT = 2.5
    cfg.MODEL.WISHNET.HEIGHT.DC_MIN_SIGMA_FRAC = 0.5
    cfg.MODEL.WISHNET.HEIGHT.BIN_CE_WEIGHT = 0.1
    # the target of the bin cross-entropy: 0 = the one-hot bin of the labelled
    # height; > 0 = SORD's soft ordinal target (Diaz & Marathe, CVPR 2019), a
    # Gaussian of this width (m) around the height, integrated over the bins and
    # renormalised over them (the mass beyond the first and last edge dropped)
    cfg.MODEL.WISHNET.HEIGHT.BIN_CE_SOFT_SIGMA_M = 1.5
    # floor-aligned bins: 3 m wide, centred on 3, 6, ..., 180 m
    cfg.MODEL.WISHNET.HEIGHT.BIN_MIN_M = 3.0
    cfg.MODEL.WISHNET.HEIGHT.BIN_MAX_M = 180.0
    cfg.MODEL.WISHNET.HEIGHT.BIN_STEP_M = 3.0
    cfg.MODEL.WISHNET.HEIGHT.BIN_RESIDUAL_WEIGHT = 1.0
    # the height terms (L1, distribution constraint, bin cross-entropy) of a
    # matched or denoising query -- in the fused decoding and in the optical
    # pass -- times the weight of its ground-truth height's bin: one weight per
    # bin above (tools/height_weights.py).  A height beyond the last bin, which
    # the head cannot reach, keeps weight 1.  The offset and detection terms
    # are not weighted.  []: off
    cfg.MODEL.WISHNET.HEIGHT.BIN_WEIGHTS = list(BDH_BIN_WEIGHTS)
    # with BIN_WEIGHTS: every height term is scaled by its batch's
    # sum(l) / sum(w l), taken without gradient.  The weights then only move the
    # gradient between the buildings; the term's total, and the value logged,
    # stay those of the unweighted term.  False: the weighted sum as it is
    cfg.MODEL.WISHNET.HEIGHT.BIN_WEIGHTS_KEEP_TOTAL = True

    # -------------------------------------------------------------- offset
    # px of the input (0.83 m with INPUT.TILE_SCALE 0.5)
    cfg.MODEL.WISHNET.OFFSET = CN()
    cfg.MODEL.WISHNET.OFFSET.SCALE = 25.0          # MLP(q) * this -> px
    cfg.MODEL.WISHNET.OFFSET.WEIGHT = 2.0
    cfg.MODEL.WISHNET.OFFSET.LOSS_SCALE = 10.0
    cfg.MODEL.WISHNET.OFFSET.BETA = 1.0
    # a labelled offset supervises only when |o| <= MAX_PX_PER_M * h + ABS_FLOOR_PX
    cfg.MODEL.WISHNET.OFFSET.MAX_PX_PER_M = 3.0
    cfg.MODEL.WISHNET.OFFSET.ABS_FLOOR_PX = 15.0

    # ---------------------------------------------------------- evaluation
    cfg.BDH_EVAL = CN()
    cfg.BDH_EVAL.HEIGHT_BINS = [3.0, 10.0, 20.0, 30.0, 50.0, 100.0]
    cfg.BDH_EVAL.IOU_THRESH = 0.5
    cfg.BDH_EVAL.MAX_DETS = 600
    # negative = sweep on this split; positive = frozen (test reuses val's)
    cfg.BDH_EVAL.SCORE_THRESH_F1 = -1.0
    cfg.BDH_EVAL.SCORE_THRESH_MAE = -1.0
    cfg.BDH_EVAL.MIN_RECALL = 0.30
    cfg.BDH_EVAL.MAE_TOL = 0.05
    cfg.BDH_EVAL.VOL_MIN_HEIGHT = 1.5
    # truncated buildings and the buildings the eval json leaves out are
    # ignore regions (COCO iscrowd)
    cfg.BDH_EVAL.IGNORE_TRUNCATED = True
    cfg.BDH_EVAL.IGNORE_REGIONS = True

    # ------------------------------------------------- checkpoints, average
    cfg.SOLVER.CHECKPOINT_MAX_KEEP = 1
    # model_best.pth by the lowest bdh/hungarian/MAE of the validations
    cfg.SOLVER.KEEP_BEST = True
    # model_best_h.pth: the validation the weight-selection rule picks
    # (wishnet/evaluation/selection.py)
    cfg.SOLVER.KEEP_BEST_H = True
    # model_best_height.pth: the average of the validation with the lowest
    # fixed-set height MAE (needs MODEL.WISHNET.FIXED_SET_HEIGHT)
    cfg.SOLVER.KEEP_BEST_HEIGHT = False
    # the exponential moving average of the weights (wishnet/engine/ema.py):
    # the validations, model_best, model_best_h and model_final use it; the
    # periodic checkpoints keep the training weights for a resume.  The decay
    # ramps as min(DECAY, (1 + n) / (WARMUP + n)) over the updates n
    cfg.SOLVER.EMA = CN()
    cfg.SOLVER.EMA.ENABLED = True
    cfg.SOLVER.EMA.DECAY = 0.9998
    cfg.SOLVER.EMA.WARMUP = 10.0
    # ema/rel_dist (||average - weights|| / ||weights||) every this many iterations
    cfg.SOLVER.EMA.LOG_PERIOD = 100


def check_config(cfg):
    """Refuse configurations the code was not written for."""
    g = cfg.MODEL.WISHNET
    if g.MODE not in ("opt", "fusion"):
        raise ValueError("MODEL.WISHNET.MODE must be 'opt' or 'fusion', got %r" % (g.MODE,))
    if not g.RANGE.ANCHORS_M:
        raise ValueError("MODEL.WISHNET.RANGE.ANCHORS_M: at least one anchor")
    if cfg.MODEL.META_ARCHITECTURE != "WISHNet":
        raise ValueError("MODEL.META_ARCHITECTURE must be WISHNet")
    if cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME != "WISHDecoder":
        raise ValueError("MODEL.MaskDINO.TRANSFORMER_DECODER_NAME must be WISHDecoder")
    if cfg.SOLVER.AMP.ENABLED:
        raise ValueError("mixed precision is not supported")
    if g.SAR.DEPTH not in (18, 34):
        raise ValueError("MODEL.WISHNET.SAR.DEPTH: 18 or 34 (basic-block ResNets)")
    if g.INSAR.ENABLED and g.MODE != "fusion":
        raise ValueError("MODEL.WISHNET.INSAR is built for mode fusion (the SAR memory of the decoder)")
    if g.INSAR.ENABLED and g.INSAR.INPUT_PX % 4:
        raise ValueError("MODEL.WISHNET.INSAR.INPUT_PX must be a multiple of 4 (two stride-2 stages)")
    if cfg.INPUT.GEO_AUG not in ("", "all8"):
        raise ValueError("INPUT.GEO_AUG must be '' or 'all8'")
    if not 0 <= cfg.INPUT.TRUNC_PX < 0.5 * cfg.INPUT.CROP_SIZE:
        raise ValueError("INPUT.TRUNC_PX: px from the window's edge, at least 0, got %s" % cfg.INPUT.TRUNC_PX)
    if g.ENCODER_EXCHANGE.ENABLED and g.MODE != "fusion":
        raise ValueError("MODEL.WISHNET.ENCODER_EXCHANGE is built for mode fusion")
    if g.ENCODER_EXCHANGE.ENABLED and (cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS != 3
                                       or len(cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES) != 3):
        raise ValueError("MODEL.WISHNET.ENCODER_EXCHANGE: the optical encoder's three levels s32, s16, s8")
    if g.SAR_MEAN_READOUT.PER_MODALITY and not (g.MODE == "fusion" and g.INSAR.ENABLED
                                                and g.SAR_MEAN_READOUT.ENABLED):
        raise ValueError("SAR_MEAN_READOUT.PER_MODALITY needs mode fusion, INSAR and the SAR readout")
    if g.SAR_MEAN_READOUT.ENABLED and g.MODE != "fusion":
        raise ValueError("MODEL.WISHNET.SAR_MEAN_READOUT reads the SAR of mode fusion")
    if g.FIXED_SET_HEIGHT.DETECTION_PERIOD < 0:
        raise ValueError("MODEL.WISHNET.FIXED_SET_HEIGHT.DETECTION_PERIOD: 0 (every validation) or more")
    if cfg.SOLVER.KEEP_BEST_HEIGHT and not g.FIXED_SET_HEIGHT.ENABLED:
        raise ValueError("SOLVER.KEEP_BEST_HEIGHT needs MODEL.WISHNET.FIXED_SET_HEIGHT.ENABLED")
    if cfg.SOLVER.EMA.ENABLED and not (0.0 < cfg.SOLVER.EMA.DECAY < 1.0 and cfg.SOLVER.EMA.WARMUP >= 0):
        raise ValueError("SOLVER.EMA: DECAY in (0, 1), WARMUP >= 0")
    if g.OPTICAL_PASS.WEIGHT < 0:
        raise ValueError("MODEL.WISHNET.OPTICAL_PASS.WEIGHT must be >= 0")
    if g.OPTICAL_PASS.WEIGHT > 0 and g.MODE != "fusion":
        raise ValueError("MODEL.WISHNET.OPTICAL_PASS is for mode fusion")
    if g.HEIGHT.BIN_CE_SOFT_SIGMA_M < 0:
        raise ValueError("MODEL.WISHNET.HEIGHT.BIN_CE_SOFT_SIGMA_M: 0 (one-hot) or a width > 0 (m)")
    if len(g.HEIGHT.BIN_WEIGHTS):
        n_bins = int(round((g.HEIGHT.BIN_MAX_M - g.HEIGHT.BIN_MIN_M) / g.HEIGHT.BIN_STEP_M)) + 1
        if len(g.HEIGHT.BIN_WEIGHTS) != n_bins or min(g.HEIGHT.BIN_WEIGHTS) <= 0:
            raise ValueError("MODEL.WISHNET.HEIGHT.BIN_WEIGHTS: one positive weight per height bin, %d, "
                             "got %d" % (n_bins, len(g.HEIGHT.BIN_WEIGHTS)))
    if g.HEIGHT.BIN_WEIGHTS_KEEP_TOTAL and not len(g.HEIGHT.BIN_WEIGHTS):
        raise ValueError("MODEL.WISHNET.HEIGHT.BIN_WEIGHTS_KEEP_TOTAL needs MODEL.WISHNET.HEIGHT.BIN_WEIGHTS")
    sp = g.SAR.SPECKLE
    if sp.ENABLED:
        if g.MODE != "fusion":
            raise ValueError("MODEL.WISHNET.SAR.SPECKLE is for mode fusion")
        if not (0.0 < sp.PROB <= 1.0 and sp.ENL > 0 and sp.NATIVE_M_X > 0 and sp.NATIVE_M_Y > 0
                and sp.MEAN_SIGMA_M > 0 and sp.KEEP_SIGMA_M > 0):
            raise ValueError("MODEL.WISHNET.SAR.SPECKLE: PROB in (0, 1]; ENL, NATIVE_M_X, NATIVE_M_Y, MEAN_SIGMA_M, "
                             "KEEP_SIGMA_M > 0")
        if sp.UPSAMPLE not in ("nearest", "bilinear"):
            raise ValueError("MODEL.WISHNET.SAR.SPECKLE.UPSAMPLE must be 'nearest' or 'bilinear'")
