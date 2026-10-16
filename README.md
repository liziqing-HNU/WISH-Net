# WISH-Net

**W**eakly-aligned **I**nstance-level multi-**S**ource **H**eight: building-level
height estimation from VHR optical imagery, GF-3 SAR and multitemporal
Sentinel-1 InSAR.

WISH-Net detects buildings in a 0.41 m optical tile and estimates the height of
each from the optical image, a GF-3 backscatter tile and a 17-channel
Sentinel-1 InSAR tile of the same ground.  The SAR inputs are only weakly
aligned with the optical image (registration and DEM errors, and layover and
shadow displaced along the range direction), so every building reads them
inside the detector, along each sensor's range direction and with learned
sampling offsets.  It is built on MaskDINO (detectron2):

- optical ResNet-50 (torchgeo, fMoW GASSL) and the MaskDINO pixel decoder;
- SAR branch: GF-3 ResNet-18 (GroupNorm) and InSAR ResNet-18 (res2-res4,
  stride-1 stem) in one joint deformable encoder;
- a two-way exchange at the encoder stage between the optical memory and the
  SAR levels, read along each sensor's range direction;
- in every decoder layer a SAR step: a range-anchored deformable read of the
  SAR memory around the query's footprint and a gated injection into the query;
- height as a distribution over 3 m bins (soft ordinal targets, a distribution
  constraint and an L1 term, weighted per height bin) and the roof-to-footprint
  offset; in training an optical-only auxiliary pass from the memory before
  the exchange.

## Repository

| path | contents |
|---|---|
| `train_net.py` | training and evaluation (detectron2 launcher) |
| `wishnet/` | configuration, data loading, model, losses, moving average, hooks, evaluation |
| `configs/bdh/` | the four models of the paper (below) |
| `tools/run.sh` | one model end to end: train, choose the weights, evaluate, zeroing |
| `tools/evaluate.py` | whole-tile evaluation with the up-down flip averaging and input zeroing |
| `tools/paired_compare.py`, `tools/make_table.py` | paired per-building gains with tile-bootstrap intervals; the results table |
| `tools/labels/` | the original split, the evaluation crops and the reviewed labels (labels v2) |
| `tools/pretrained/` | the pretrained backbones |
| `tools/gf3/`, `tools/s1insar/` | the GF-3 and Sentinel-1 InSAR processing chains |
| `tools/height_weights.py`, `tools/model_cost.py` | the height-bin weights; parameters and inference time |
| `tests/` | checks of this code and of its equivalence with the original implementation |
| `datasets/` | the label corrections, md5 sums, the data layout (`datasets/README.md`) |
| `maskdino/` | MaskDINO, unchanged (its license: `maskdino/LICENSE`) |

## Installation

See [INSTALL.md](INSTALL.md): Python 3.10, PyTorch 2.4.1 with CUDA 12.1,
detectron2 0.6 built from source and MaskDINO's deformable-attention operator.

## Data and pretrained weights

See [datasets/README.md](datasets/README.md).  The dataset folder
`datasets/bdh` holds BDH's original train / val split of the optical tiles, the
GF-3 and InSAR tiles of the same ground and the label files; every label file
and every tile the runs read is listed with its md5 (`datasets/bdh_md5.txt`,
`datasets/bdh_tiles_md5.txt`).  Two GF-3 statistics come from the training
tiles of the dataset's geographic split, on which the work began: the mean and
standard deviation that standardise the GF-3 input, and the batch-norm
statistics of the GF-3 backbone's initialisation (made again from those tiles,
bit-identical).  The three pretrained backbones go to `pretrained/`
(`tools/pretrained/`, md5 in `pretrained/md5.txt`).

## The models of the paper

All four share the recipe (`configs/bdh/Base-WISHNet.yaml`) and the seed 3408;
each config adds one change to the one before.

| paper | config |
|---|---|
| Optical only | `configs/bdh/opt.yaml` |
| + GF-3 | `configs/bdh/gf3.yaml` |
| + InSAR | `configs/bdh/gf3_insar.yaml` |
| + exchange (WISH-Net) | `configs/bdh/wishnet.yaml` |

Training: AdamW, learning rate 1e-4 (the optical ResNet-50 at 0.1x), weight
decay 0.05, gradient clipping at 1.0 over the whole model, two tiles per batch,
the rate lowered tenfold at 20k steps, 22k steps; the moving average of the
weights (decay 0.9998) is validated every 2000 steps.  One run takes about ten
hours on one RTX 4090 D.

## Training and evaluation

```bash
CFG=configs/bdh/wishnet.yaml bash tools/run.sh
```

`tools/run.sh` trains (resuming an interrupted run), validates the three
candidate weights (the lowest validation height MAE, the weight-selection rule
over the validations, the last), chooses among them by the rule (drop those
whose F1 is more than one point below the highest, then the lowest best-F1
MAE), evaluates the chosen weights on the validation split with the up-down
flip averaging, and for the fusion models runs the zeroing diagnostics (GF-3,
InSAR and both set to zero, at the full input's thresholds, without flip
averaging).  Everything goes to the config's `OUTPUT_DIR`; `final_val/orig` is
the model as it is (F1, mAP50, the best-F1 threshold), `final_val/avg` the same
detections with the flip-averaged heights (MAE, RMSE, per height bin).

The evaluation protocol: whole 2048 px tiles; COCO segm matching at IoU 0.5;
buildings cut by the tile edge and buildings without a recorded floor count
are ignore regions; F1, the height MAE and RMSE at the score threshold of the
best F1, swept on the split; mAP50; and the MAE of the BDH protocol (no score
threshold, greedy matching), flip-averaged in the tables and unflipped where
it is set against published numbers.

## Reproducing the tables

With the four runs in `output/{opt,gf3,gf3_insar,wishnet}`:

```bash
# the ladder: paired gains on the buildings both models detect, flip-averaged heights
python tools/paired_compare.py --gt datasets/bdh/prepared_v2/instances_val_crop2048.json \
    --run OPT=output/opt/final_val/avg --run GF3=output/gf3/final_val/avg \
    --run INSAR=output/gf3_insar/final_val/avg --run WISH-Net=output/wishnet/final_val/avg \
    --ref OPT --pair GF3,INSAR --pair INSAR,WISH-Net --out output/paired_ladder
python tools/make_table.py --run OPT=output/opt/final_val --run GF3=output/gf3/final_val \
    --run INSAR=output/gf3_insar/final_val --run WISH-Net=output/wishnet/final_val \
    --paired output/paired_ladder/analysis.json
# zeroing: the cost of a zeroed input, unflipped (gain of the full input over the zeroed one)
python tools/paired_compare.py --gt datasets/bdh/prepared_v2/instances_val_crop2048.json \
    --run ZERO=output/wishnet/zero_val/orig --run FULL=output/wishnet/final_val/orig \
    --ref ZERO --out output/paired_zero
```

`--exclude-gt datasets/bdh/prepared_v2/instances_val_crop2048_model_sourced_ids.txt`
leaves out the 44 validation buildings whose reviewed height or offset was taken
from a model's prediction (`--only-gt` keeps only them).

The other runs of the paper -- the network with a SAR query stream and its run
with the default seed (ladder), and the comparison of interferometric inputs
(the S1 stack, one S1 pair, one GF-3 pair at 6.63 m and at 3.31 m) -- were made
with the original implementation; this repository holds the final design and
its ladder only.

## Checkpoints and the working name

WISH-Net was developed under the working name GFOPTNet.  The rename changed
names only: the package (`wishnet`), the classes and registry names
(`WISHNet`, `WISHDecoder`, `WISHDatasetMapper`, `WISHCriterion`), the
configuration namespace (`MODEL.WISHNET`), file names and messages.  No
parameter or buffer name contains any of them, so checkpoints written before
the rename load strictly as they are -- no tensor missing, extra or reshaped --
and no key mapping is needed.  A configuration file written for the working
name needs its `MODEL.GFOPT` section renamed to `MODEL.WISHNET` and the
registry names `GFOPTNet` and `GFOPTDecoder` replaced by `WISHNet` and
`WISHDecoder`.  The rename was checked like the port below (`tests/README.md`).

## Equivalence with the original implementation

This repository is the final design ported from the implementation the paper's
runs were trained with, without its experiment switches.  The module and
parameter names are kept, so the paper's weights load as they are (strictly:
no tensor missing, extra or reshaped), and the configurations differ only in
paths, the 22k stop and two optional readouts.  With the same seed the two
give identical initial weights, and their forward passes are bit-identical:
with the paper's WISH-Net weights the whole-tile evaluation of the validation
split is the same detection for detection and metric for metric (167 850
detections, 134 metrics), with and without the flip averaging.  On the CPU
with one thread and deterministic algorithms, training is deterministic, and
every loss term, every gradient and every parameter after each step is
bit-identical between the two codes (12 steps); with several threads two runs
of the same code already part after a few steps.  On the GPU, MaskDINO's
deformable attention accumulates its gradients with atomic additions, so two
trainings of the same code differ from the second step on.  Two trainings
with each code from the same weights, 200 steps on the real data (a 12 GB GPU:
one tile per batch, the GF-3 window read at 512 px), give bit-identical losses
at the first step in all four, and over the other 199 steps the median
difference of the total loss between a run of one code and a run of the other
(0.11 to 0.98) is no larger than between the two runs of the same code (0.80
and 1.10); the parameters after the 200 steps differ by at most 0.042, across
the codes and within them alike (`tests/equivalence/`, `tests/README.md`).

The paper's runs were configured with 24000 steps and stopped near 22k; a few
ran some tens of steps longer (for example the run with the SAR query stream
and seed 3408 stopped at 22028).  Every weight the paper uses was chosen by
validation at step 21999 or earlier, so this does not affect them.  This
repository ends training at 22000 steps; the learning rate of every step
0-21999 (bit for bit) and the steps of the validations and checkpoints are the
same as under the stop (`tests/check_schedule.py`).

## Citation

To be added with the paper.

## License

To be decided.  `maskdino/` is MaskDINO (IDEA Research), under the Apache
License 2.0 ([maskdino/LICENSE](maskdino/LICENSE)).
