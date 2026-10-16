# Data

The tiles and label files are not part of this repository.  This page gives
the layout the code reads, how each file is made, and the md5 sums of the
files the paper's runs read: the label and metadata files (`bdh_md5.txt`) and
the 2793 tiles (`bdh_tiles_md5.txt`), both relative to `datasets/bdh`:

```bash
cd datasets/bdh && md5sum -c ../bdh_md5.txt && md5sum -c ../bdh_tiles_md5.txt
```

## Sources

- **Optical tiles and labels**: BDH (the building detection and height dataset
  of BDH-Net), 931 RGB tiles of 2048 x 2048 px at 0.41 m over Beijing, with
  each building's roof polygon, footprint polygon and floor-count height.
- **GF-3**: two Gaofen-3 UFS scenes, HH, single-look complex (L1A), of
  2024-04-06 (`GF3_SYC_UFS_040315_E116.4_N39.9_20240406_L1A_DH_L10006917544`)
  and 2024-04-01 (`GF3_SYC_UFS_040243_E116.1_N40.0_20240401_L1A_DH_L10006915245`),
  calibrated to sigma0 in dB and resampled onto every tile's grid
  (`tools/gf3/`).
- **Sentinel-1 InSAR**: Sentinel-1A ascending, relative orbit 142, IW VV, eight
  dates 2023-06-06 to 2023-08-29; seven 12-day pairs plus mean coherence, mean
  sigma0 and amplitude dispersion, 17 channels on 128 x 128 px per tile (6.63 m)
  (`tools/s1insar/`).

The tiles were packaged per tile in a geographically split dataset
(BDH-RGB-SAR: train / val / test of 561 / 186 / 184 tiles in 6 x 6-tile
blocks, `metadata/splits.csv` keeping BDH's original split of each tile).  The
paper uses BDH's original random split, 558 training and 373 validation tiles,
no test split.

## Layout of `datasets/bdh`

```
datasets/bdh/
  images/{train,val}/<tile>.tif               RGB, 2048 x 2048
  sar_gf3/{train,val}/<tile>_sar.tif          GF-3 HH sigma0 (dB), float32, NaN = no data, the tile's grid
  s1_insar_asc/{train,val}/<tile>_sar.tif     17 x 128 x 128, float32, NaN = no data
  s1_insar_asc/S1_INSAR_ASC.json              channel names, dates, geometry, normalisation statistics
  annotations_fixed/instances_{train,val}.json   BDH's labels on the original split
  prepared/instances_{train,val}_crop2048{,_ignore}.json   evaluation crops and ignore regions
  annotations_v2/instances_{train,val}.json   the reviewed labels (training reads the train file)
  prepared_v2/instances_{train,val}_crop2048{,_ignore}.json   the reviewed evaluation crops
```

The configs read `DATASETS.BDH_ROOT datasets/bdh`, `DATASETS.BDH_ANN_DIR
annotations_v2` and `DATASETS.BDH_PREPARED datasets/bdh/prepared_v2`; the
GF-3 and InSAR folders default to `sar_gf3` and `s1_insar_asc` under the root
(`MODEL.WISHNET.SAR.ROOT`, `MODEL.WISHNET.INSAR.ROOT`).

## Making the files

```bash
# 1. the original split: annotation files merged and re-split, the tiles linked
python tools/labels/make_orig_split.py --src <BDH-RGB-SAR> --dst datasets/bdh
# 2. the evaluation crops, their ignore regions and the reviewed labels, checked by md5
bash tools/labels/make_labels_v2.sh datasets/bdh
```

- **Evaluation crops** (`prepare_bdh.py`): one crop per tile, the whole tile;
  the annotations with a recorded height of at least 3 m whose visible part
  passes the gates (32 px^2, a tenth of the roof); each with its footprint ring,
  roof-to-footprint offset, the share of the roof inside (`clip_frac`) and
  `truncated` (less than 99.5 % of the roof inside the tile, the roof within
  1 px of its edge, or the footprint reaching beyond it).
- **Ignore regions** (`make_ignore_regions.py`): the buildings the crop leaves
  out -- no recorded floor count, or below the gates.  With the truncated
  buildings they are ignore regions of the evaluation (neither true nor false
  positives, never scored for height).
- **Reviewed labels** (`apply_label_changes.py`, `bdh_labels_v2_changes.json`):
  the corrections of a review of the labels -- heights (the reviewer's, derived
  from the offset, a model's prediction the reviewer took, or a published
  height), roof-to-footprint offsets, roofs and footprints swapped, buildings
  excluded from height supervision and evaluation, one cut building, one
  duplicate -- 255 training and 313 validation buildings.  44 validation
  buildings have a height or offset taken from a model's prediction; their ids
  are written to `prepared_v2/instances_val_crop2048_model_sourced_ids.txt`.
- The **GF-3** and **InSAR** tiles: `tools/gf3/README.md`,
  `tools/s1insar/README.md`.

## Normalisation

- GF-3: `MODEL.WISHNET.SAR.MEAN -8.67`, `SAR.STD 6.79` (dB).
- InSAR: per channel, the `train_orig` statistics of `S1_INSAR_ASC.json` (the
  original split's 558 training tiles, pixels where every channel has data);
  no-data pixels are set to zero, the training mean, after standardisation.

Two GF-3 statistics were taken from the training tiles of the geographic split
(BDH-RGB-SAR's 561), before the paper moved to the original split, and kept:

- `SAR.MEAN` / `SAR.STD`: the product's mean and standard deviation over those
  tiles (`tools/gf3/tile_summary_gf3.py`: -8.6658, 6.7891);
- the batch-norm statistics of the GF-3 ResNet-18 (below): re-estimated on
  windows of those tiles (`convert_r18_imagenet.py --adabn
  <BDH-RGB-SAR>/sar_gf3/train`), they are bit-identical to the paper's file;
  re-estimated on the original split's training tiles they differ by up to
  1.6e2 (relative).

## Pretrained backbones

```bash
python tools/pretrained/convert_torchgeo_r50.py resnet50_fmow_rgb_gassl-da43d987.pth \
    pretrained/R-50_torchgeo_fmow_rgb_gassl.pkl
python tools/pretrained/convert_r18_imagenet.py --src resnet18-f37072fd.pth \
    --out pretrained/R-18_imagenet_sar1_adabn.pkl --adabn <BDH-RGB-SAR>/sar_gf3/train --n 400
python tools/pretrained/fold_bn_to_gn.py --src pretrained/R-18_imagenet_sar1_adabn.pkl \
    --out pretrained/R-18_imagenet_sar1_adabn_gn.pkl
```

`pretrained/md5.txt` gives the md5 of the paper's three files and of the two
downloaded checkpoints.  The converted files were written by Python 3.10
(pickle protocol 4); written by a Python with another default protocol their
md5 differs while every tensor is the same, which `tests/compare_pretrained.py`
checks tensor by tensor.

The optical ResNet-50 is torchgeo's `ResNet50_Weights.FMOW_RGB_GASSL`
(huggingface.co/torchgeo/resnet50_fmow_rgb_gassl), the SAR ResNet-18s start
from torchvision's ImageNet ResNet-18 (`resnet18-f37072fd.pth`).  The GF-3
branch reads the ResNet-18 with batch-norm statistics re-estimated on GF-3
training windows and folded into its GroupNorms; the InSAR branch reads only
the res2-res4 convolution weights, which the re-estimation does not change.

## Other regions

A region in the same layout runs with its own `DATASETS.BDH_ROOT`.  The scene
geometry enters through `MODEL.WISHNET.RANGE.TOWARD_SENSOR_DEG` (direction
towards the GF-3 sensor, degrees clockwise from north; 100.2 over Beijing),
`MODEL.WISHNET.INSAR.TOWARD_SENSOR_DEG` (towards the InSAR sensor; 260.3) and
`MODEL.WISHNET.RANGE.GSD_M` (the tile's pixel size, 0.4142 m); the GF-3
statistics through `SAR.MEAN` / `SAR.STD`.  An InSAR product with other
channels names them in its metadata file (`MODEL.WISHNET.INSAR.META`, with the
statistics under `NORM_KEY`); `MODEL.WISHNET.INSAR.CHANNELS` selects channels by
name.
