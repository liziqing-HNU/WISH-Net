# GF-3 chain: L1A SLC to sigma0 (dB) tiles

Two Gaofen-3 UFS single-look complex scenes (L1A, HH) are calibrated to sigma0,
geocoded backward onto the common BDH SAR grid over a ground height model,
mosaicked, converted to dB and resampled onto the grid of every BDH tile.

**Not run in this release.** GDAL and the GF-3 data were not available where
this release was prepared: the scripts were byte-compiled, and their
command-line handling was exercised on small synthetic inputs with GDAL replaced
by a stub; no product was made.  The commands below are reconstructed from the
scripts' docstrings and from the metadata the original runs recorded.  A value
whose exact command line was never recorded is marked *reconstructed*, with its
source.  The code is that of the original runs; only hard-coded server paths
became command-line arguments.

Recorded metadata cited below (outputs of the original runs, not part of this
release):

| file | written by | md5 |
|---|---|---|
| `SAR_BANDS_gf3.json` | `mosaic_gf3.py`; `train_statistics` by `tile_summary_gf3.py`; `dem`, `registration_check` and `geometry_measured_single_look` were added afterwards (no script of this chain writes them) | `07eac3e263cb523c362d44848679d7a6` |
| `tile_summary_gf3.json` | `tile_summary_gf3.py` | `e5af206c559b34f3b17d7713f3286ddd` |

## Scripts

| script | role | needs |
|---|---|---|
| `make_pseudo_dtm.py` | ground height model from the DSM | NumPy, SciPy |
| `gf3_common.py` | product metadata, orbit fit, range-Doppler geometry, RPC, calibration; imported by `geocode_gf3.py` | NumPy |
| `geocode_gf3.py` | one scene: calibrated sigma0, geocoded onto the grid | GDAL, SciPy |
| `mosaic_gf3.py` | the two scenes: mosaic, dB product, product json | GDAL, SciPy |
| `make_tiles_gf3.py` | the product on every tile's grid | OpenCV, tifffile with imagecodecs (floating-point predictor) |
| `tile_summary_gf3.py` | per-tile coverage and source scene, per-split statistics | tifffile |
| `check_registration.py` | offset of a geocoded scene against the S1 mosaics | GDAL, SciPy |
| `registration_check_gf3.py` | wall-base profile of a mosaic on the building footprints | SciPy, tifffile; matplotlib (`profile-map`, `compare`), OpenCV (`compare`), GDAL (`export`) |

## Inputs

Placeholders: `<raw>` the unpacked products, `<dem>` the height models,
`<gf3>` the work folder, `<tiles>` the tile folder, `<BDH-RGB-SAR>` the dataset.

- **GF-3 products**, unpacked; `--scene` is the product path without extension
  (`<scene>.meta.xml`, `.rpc`, `.tiff`, `.incidence.xml` are read):
  `GF3_SYC_UFS_040315_E116.4_N39.9_20240406_L1A_DH_L10006917544` (2024-04-06,
  main scene) and `GF3_SYC_UFS_040243_E116.1_N40.0_20240401_L1A_DH_L10006915245`
  (2024-04-01, fills the north-west corner) (`SAR_BANDS_gf3.json`
  `scenes.<date>.scene`).
- **DSM** `<dem>/beijing.dem.wgs84` with its `.vrt`: Copernicus GLO-30 in
  ellipsoidal heights, ISCE-style float32, 1 arcsec over 115-118 E, 39-41 N
  (7200 x 10800, `SHAPE` in `make_pseudo_dtm.py`).  `make_pseudo_dtm.py` writes
  its `.vrt` by copying the DSM's and replacing the DSM's file name by the
  absolute path of `--out` (the original script had the name
  `beijing.dem.wgs84` written in; this is the one change beyond paths made
  arguments).  The conversion of GLO-30 to this file is not part of the
  chain.
- **Grid json** `<grid json>`: any json whose `grid` holds `width` 18555,
  `height` 18809 and `geotransform_3857` [12934210.89940096, 2.15993024, 0,
  4870303.58215232, 0, -2.15993024] (EPSG:3857, 1.657 m on the ground), the
  grid of the BDH Sentinel-1 products.  The original runs read the S1
  ascending product's `sar_3ch_scales.json`; `SAR_BANDS_gf3.json` carries the
  same block (`grid`).
- **Dataset** `<BDH-RGB-SAR>/annotations_fixed/instances_{train,val,test}.json`
  (the geographic split, 561 / 186 / 184 tiles): per image `geotransform_3857`,
  `width`, `height`, `sar_file_name`.
- For `check_registration.py` only: the BDH S1 descending and ascending
  3-channel uint8 mosaics on the same grid (band 3: VV in dB) with their json
  (`encoding.scales.vv_db`), products of the earlier BDH SAR processing, not
  part of this release.

## Steps

```bash
# 1. ground height model
#    (recorded: python make_pseudo_dtm.py --out .../beijing.pdtm.wgs84, the DSM path a script default)
python tools/gf3/make_pseudo_dtm.py --dsm <dem>/beijing.dem.wgs84 --out <dem>/beijing.pdtm.wgs84

# 2. each scene: calibrated sigma0 on the grid (reconstructed, see Parameters)
python tools/gf3/geocode_gf3.py --scene <raw>/GF3_SYC_UFS_040315_E116.4_N39.9_20240406_L1A_DH_L10006917544 \
    --grid <grid json> --dem <dem>/beijing.pdtm.wgs84.vrt --out <gf3>/geo \
    --doppler rpc --shift-az-m 16.67 --shift-rg-m 11.77
python tools/gf3/geocode_gf3.py --scene <raw>/GF3_SYC_UFS_040243_E116.1_N40.0_20240401_L1A_DH_L10006915245 \
    --grid <grid json> --dem <dem>/beijing.pdtm.wgs84.vrt --out <gf3>/geo \
    --doppler rpc --shift-az-m 16.72 --shift-rg-m 12.38

# 3. mosaic and product
#    (recorded: python mosaic_gf3.py --single hh_sl_db --float --out <dir>, --geo and --grid script defaults)
python tools/gf3/mosaic_gf3.py --geo <gf3>/geo --grid <grid json> --single hh_sl_db --float \
    --out <gf3>/gf3_product

# 4. the product and the per-pixel source scene on every tile's grid (reconstructed)
python tools/gf3/make_tiles_gf3.py --mosaic <gf3>/gf3_product/gf3_hh_sl_db.tif \
    --meta <gf3>/gf3_product/SAR_BANDS_gf3.json --root <BDH-RGB-SAR> --out <tiles>/sar_gf3
python tools/gf3/make_tiles_gf3.py --mosaic <gf3>/gf3_product/gf3_source.tif \
    --meta <gf3>/gf3_product/SAR_BANDS_gf3.json --root <BDH-RGB-SAR> --out <tiles>/sar_gf3_source \
    --interp nearest

# 5. per-tile summary and training statistics (recorded: python tile_summary_gf3.py --tiles .../tiles)
python tools/gf3/tile_summary_gf3.py --tiles <tiles>
```

| step | writes |
|---|---|
| 1 | `<dem>/beijing.pdtm.wgs84` (float32, the DSM's grid) and its `.vrt` |
| 2 | `<gf3>/geo/<date>_sigma0.tif` (linear sigma0 on the grid window of the scene, NaN = no data), `<date>_geometry.tif`, `<date>_geocode.json` |
| 3 | `<gf3>/gf3_product/gf3_hh_sl_db.tif` (the product: single-look sigma0 HH in dB, float32, NaN = no data), `gf3_sigma0_mosaic.tif` (linear), `gf3_source.tif` (1 = 2024-04-06, 2 = 2024-04-01), `SAR_BANDS_gf3.json` |
| 4 | `<tiles>/sar_gf3/{train,val,test}/<sar_file_name>` (2048 x 2048, float32 dB), `<tiles>/sar_gf3_source/...`, `tile_coverage.csv` and a copy of the json in each |
| 5 | `<tiles>/tile_summary_gf3.json`; `train_statistics` added to `<tiles>/sar_gf3/SAR_BANDS_gf3.json` |

Step 4 of the original processing is not recorded as a command; the docstring
example of `make_tiles_gf3.py` cuts the earlier 3-channel uint8 candidate
(`gf3_3ch_uint8.tif`), while the tiles of the paper are the float product
`gf3_hh_sl_db.tif` (`SAR_BANDS_gf3.json` `file`, `encoding.dtype` float32).
The source maps are read by step 5 (`tile_summary_gf3.py` docstring).

## Registration checks

The constant corrections of step 2 and the choice of the ground model come
from two measurements.

```bash
# offset GF-3 minus S1 per 1.7 km block (scene geocoded without corrections: the defaults 0)
python tools/gf3/check_registration.py --gf3 <gf3>/geo/20240406_sigma0.tif \
    --geocode-json <gf3>/geo/20240406_geocode.json \
    --s1-desc <s1 desc mosaic> <its json> --s1-asc <s1 asc mosaic> <its json> --out-json <file>
# wall-base profile of a mosaic on the footprint edges facing the sensor
python tools/gf3/registration_check_gf3.py profile --layers "GF-3=<gf3>/gf3_product/gf3_sigma0_mosaic.tif" \
    --u 0.9836,0.1801 --root <BDH-RGB-SAR> --grid <grid json> --summary <tiles>/tile_summary_gf3.json \
    --product-json <gf3>/gf3_product/SAR_BANDS_gf3.json --out <gf3>/registration_check
```

- `check_registration.py` splits the block offsets into along-track and
  towards-sensor parts.  Against the S1 descending mosaic, whose geometry is
  nearly that of GF-3, the uncorrected scenes sat +16.2 m along track and
  23.4 m away from the sensor (2024-04-06), +17.3 m and 32.4 m (2024-04-01);
  in radar coordinates both are about 16.7 m in azimuth and 12 m in slant
  range, a constant of the products' timing (experiment notes,
  `docs/gf3_data/README.md` of the MaskDINO-BDH repository).  The conversion
  to the values passed in step 2 is not scripted and was not recorded.
- `registration_check_gf3.py profile` stacks the sensor-facing walls (67 552,
  `make_pseudo_dtm.py` docstring); for a registered product the wall-ground
  dihedral peaks at 0 m.  Wall-base peak, all walls: uncorrected -32.8 m
  (experiment notes); corrected, geocoded on the DSM -11.1 m
  (`registration_check.dsm_version_for_comparison_m`); corrected, on the
  ground model +0.3 m, train / val / test +1.0 / +0.4 / -1.4 m
  (`registration_check.wall_base_peak_m`), per 1.7 km block median -0.19 m,
  IQR [-2.29, 2.63] m (`registration_check.per_1.7km_block`).  `--u` is the
  2024-04-06 scene's unit vector towards the sensor
  (`scenes.20240406.geometry.unit_px_towards_sensor`).
- `profile-map`, `compare` and `export` (peak per block, image pairs with a
  swipe viewer, QGIS layers) take the same arguments, the last two also
  `--tiles <split>:<image id> ...`.  The viewer's labels are in Chinese, as in
  the original script.

## Parameters of the paper's data

Keys refer to `SAR_BANDS_gf3.json` unless named otherwise.

| parameter | value | source |
|---|---|---|
| scenes, priority | 2024-04-06 first, 2024-04-01 fills the rest (5.6 % of the grid) | `priority`, `scenes.20240401.pixels_in_mosaic_fraction`; `ORDER` in `mosaic_gf3.py` |
| polarisation | HH | `bands`; `GF3Meta(pol="HH")` in `gf3_common.py` |
| calibration | sigma0 = (I^2 + Q^2) (QualifyValue / 32767)^2 / 10^(K / 10), K = CalibrationConst = 18.04 dB; QualifyValue 7227.43 (04-06), 2265.08 (04-01) | `GF3Meta.sigma0_gain`; `scenes.<date>.calibration` |
| ground model | DSM local minimum over 25 px (750 m), Gaussian sigma 8 px (240 m); blended linearly back to the DSM as the smoothed ground relief over 750 m rises from 25 to 50 m; never above the DSM; the script defaults | `make_pseudo_dtm.py`; `dem.method` |
| height model of geocoding | the ground model, `--dem <dem>/beijing.pdtm.wgs84.vrt` (reconstructed) | `dem.file`; `make_pseudo_dtm.py` docstring |
| Doppler convention | `--doppler rpc` (reconstructed; the default) | `scenes.<date>.doppler_convention` |
| along-track correction | `--shift-az-m` 16.67 (04-06), 16.72 (04-01) (reconstructed) | `scenes.<date>.shift_az_m` |
| slant-range correction | `--shift-rg-m` 11.77 (04-06), 12.38 (04-01) (reconstructed) | `scenes.<date>.shift_rg_m` |
| orbit fit | degree 7 over the scene +- 40 s | `Orbit` defaults in `gf3_common.py` |
| geocoding nodes | every 4 grid px (`STEP`), bilinear in between; intensity sampled bilinearly | `geocode_gf3.py` |
| mosaic | no feathering; dB with a floor of -35 dB; `--single hh_sl_db --float` | `mosaic_gf3.py`; `encoding.floor_db`; recorded command |
| product | `gf3_hh_sl_db.tif`, float32 dB, NaN = no data, 99.95 % of the grid valid | `file`, `encoding`, `valid_fraction` |
| tiles | bilinear in dB (`--interp linear`, the default) onto each tile's `geotransform_3857`, float32, DEFLATE with the floating-point predictor (reconstructed command) | `make_tiles_gf3.py`; `file` |
| statistics | mean -8.6658 dB, std 6.7891 dB over the 561 training tiles of the geographic split; the model's `MODEL.WISHNET.SAR.MEAN` -8.67, `SAR.STD` 6.79; not recomputed for the original split | `train_statistics`; `tile_summary_gf3.json` `splits.train`; `wishnet/config.py` |

## Original split

The paper uses BDH's original train / val split.  `tools/labels/make_orig_split.py`
links the tiles of `<BDH-RGB-SAR>/sar_gf3/<geographic split>/` into
`datasets/bdh/sar_gf3/{train,val}/` and the folder's json files into
`datasets/bdh/sar_gf3/`; place or link `<tiles>/sar_gf3` there first:

```bash
python tools/labels/make_orig_split.py --src <BDH-RGB-SAR> --dst datasets/bdh --sar-dir sar_gf3 s1_insar_asc
```
