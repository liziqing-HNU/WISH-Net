# Sentinel-1 InSAR chain: ISCE2 stack to 17-channel tiles

Eight Sentinel-1A IW SLC acquisitions (ascending, relative orbit 142, VV),
coregistered by ISCE2 topsStack, are turned burst by burst in radar geometry
into the locally high-passed complex coherence of the seven sequential 12-day
pairs, flattened against the ground height model of the GF-3 chain, plus the
mean coherence, the mean sigma0 and the amplitude dispersion; the bursts are
geocoded onto one lattice over the area of interest, mosaicked and averaged
onto the grid of every BDH tile.

**Not run in this release.** ISCE2, GDAL and the Sentinel-1 data were not
available where this release was prepared: the scripts were byte-compiled, and
their command-line handling was exercised on small synthetic inputs with GDAL
and `s1a_isce_utils` replaced by stubs (`make_s1insar.py --dry-run` gives the
recorded lattice, 9278 x 9405); no product was made.  The commands below are
reconstructed from the scripts' docstrings and from the metadata the original
runs recorded.  A value whose exact command line was never recorded is marked
*reconstructed*, with its source.  The code is that of the original runs; only
hard-coded server paths became command-line arguments.

Recorded metadata cited below: `S1_INSAR_ASC.json` (written by `tile_s1insar.py`:
the mosaic json of `make_s1insar.py` with the tiles and the normalisation
statistics added; md5 `44096977c82cde414ec6ed55b4dbc107`; not part of this
release, the dataset folder holds it as `datasets/bdh/s1_insar_asc/S1_INSAR_ASC.json`).

## Scripts

| script | role | needs |
|---|---|---|
| `make_s1insar.py` | the stack: per-burst features in radar geometry, geocoding, AOI mosaic and json | ISCE2 (`s1a_isce_utils` of topsStack), GDAL (Python bindings and `gdalwarp`), NumPy, SciPy |
| `tile_s1insar.py` | the mosaic averaged onto every tile, normalisation statistics | GDAL, NumPy |

## Inputs

Placeholders: `<zips>` the SAFE zips, `<work>` the stack, `<dem>` the height
models, `<s1insar>` the output folder, `<BDH-RGB-SAR>` the dataset.

- **SAFE zips** of the eight dates, `<zips>/S1A_IW_SLC__1SDV_<date>T*.zip`:
  20230606, 20230618, 20230630, 20230712, 20230724, 20230805, 20230817,
  20230829 (`TRACKS["asc"]`; `S1_INSAR_ASC.json` `dates`).  The annotation
  (orbit state vectors, sigmaNought calibration vectors) is read from them.
- **Stack** `<work>`: `reference/<IW>.xml`, `coreg_secondarys/<date>/<IW>/`
  (`burst_NN.slc.vrt`, `range_NN.off.vrt`) and `geom_reference/<IW>/`
  (`lat`, `lon`, `hgt`, `los` `_NN.rdr.vrt`), swaths IW2 and IW3.
- **Ground height model** `<dem>/beijing.pdtm.wgs84.vrt`, made by
  `tools/gf3/make_pseudo_dtm.py` (`S1_INSAR_ASC.json` `source.ground_dem`).
- **Dataset** `<BDH-RGB-SAR>`: `annotations_fixed/instances_{train,val,test}.json`
  (the geographic split; per image `geotransform_3857`, `width`, `height`,
  `file_name`, `sar_file_name`, `gsd_m`) and `metadata/tiles_index.csv` (`split`:
  `bdh_train` / `bdh_val`, BDH's original split).

## Steps

### 0. Stack (ISCE2; not part of this release; reconstructed)

```bash
stackSentinel.py -s <zips> -o <orbits> -a <aux> -d <dem>/dem_isce.dem.wgs84 -w <work> \
    -b '<S N W E>' -W slc -C geometry -p vv
# then the run files run_01_unpack_topo_reference, run_02_unpack_secondary_slc,
# run_04_fullBurst_geo2rdr and run_05_fullBurst_resample of <work>/run_files
```

The command line of the ascending stack, its bounding box and its orbit files
were not recorded.  Sources: the docstring of `make_s1insar.py` (coregistered
onto 20230606 by topsStack, `reference`, `coreg_secondarys`, `geom_reference`,
run_02/04/05, no ESD; the secondaries flattened against the DEM of the topo
step, Copernicus GLO-30 in ellipsoidal heights, `dem_isce`); `S1_INSAR_ASC.json`
`source.dem_of_flattening` (`dem_isce.dem.wgs84`, GLO-30 DSM); the options
`-W slc -C geometry -p vv` and the four run files as recorded for the
descending S1B stack, the `desc` track of `make_s1insar.py`
(`tools_bdh/sar_desc/run_isce_desc.sh` of the MaskDINO-BDH repository).
`reference/` and `geom_reference/` are written by run_01.  The stack must
cover the area of interest, 116.19-116.55 E, 39.75-40.03 N (`AOI`).

### 1. Features and mosaic

```bash
# recorded: python tools/s1insar/make_s1insar.py --out <s1insar> in an ISCE2 environment,
# the stack, zips and ground model being constants of the script then
python tools/s1insar/make_s1insar.py --work <work> --zips <zips> [--old-zips <dir>/] \
    --pdtm <dem>/beijing.pdtm.wgs84.vrt --topsstack <isce2>/topsStack --out <s1insar>
```

`--topsstack` is the topsStack folder of the ISCE2 installation (default
`$ISCE_STACK/topsStack`; for a conda ISCE2, `$CONDA_PREFIX/share/isce2/topsStack`).
`--old-zips`: the zip folder named inside the stack's burst VRTs, rewritten to
`--zips` (default `--zips` itself); the paper's stack named an earlier folder
(`TRACKS["asc"]["old_zips"]` of the original script).  No `--orbits`: the
ascending baselines come from the SAFE annotation orbits (`TRACKS["asc"]`;
the json has no `source.orbits`).  `--dry-run` prints the constants and the
bursts on the area of interest.

Writes `<s1insar>/s1insar_asc_mosaic.tif` (17 bands, float32, NaN = no data)
and `s1insar_asc_mosaic.json` (channels, dates, pairs, per-burst baselines,
heights of ambiguity, incidence, line-of-sight azimuth and checks).

### 2. Tiles

```bash
# recorded: python tools/s1insar/tile_s1insar.py with its defaults
python tools/s1insar/tile_s1insar.py --mosaic <s1insar>/s1insar_asc_mosaic.tif --root <BDH-RGB-SAR>
```

Writes `<BDH-RGB-SAR>/s1_insar_asc/{train,val,test}/<sar_file_name>` (17 x 128 x
128, float32, NaN = no data; named like the GF-3 tile), `S1_INSAR_ASC.json` and
`tile_quality.csv` (`--out` defaults to `<root>/s1_insar_<track>`).

## Parameters of the paper's data

Keys refer to `S1_INSAR_ASC.json` unless named otherwise; constants are those
of `make_s1insar.py` and `tile_s1insar.py`.

| parameter | value | source |
|---|---|---|
| track | `--track asc` (the default): Sentinel-1A, relative orbit 142, ascending, IW, VV; swaths IW2, IW3 | `TRACKS["asc"]`; `product` |
| dates, pairs | 8 dates 2023-06-06 to 2023-08-29, reference 20230606; 7 sequential 12-day pairs | `dates`, `reference`, `pairs` |
| bursts on the area | IW2_01-03, IW3_02-04 | `bursts` |
| area of interest | 116.19-116.55 E, 39.75-40.03 N | `AOI` |
| flattening | ISCE's, against `dem_isce.dem.wgs84` (GLO-30 DSM); re-referenced to the ground model, `exp(j K (h_DSM - h_ground))` | docstring; `source.dem_of_flattening`, `source.ground_dem`, `phase_model` |
| baselines | Bp at the burst window centre from the SAFE annotation orbits; sign of K +1 (`--sign`, default) | `TRACKS["asc"]`; `sign`; `bursts.<burst>.bperp_m` |
| coherence window | 5 range x 2 azimuth samples (about 17 x 28 m) | `WIN`; `window_looks` |
| high-pass | Gaussian, sigma 400 m on the ground | `SIGMA_M`; `highpass_sigma_m` |
| sigma0 | VV, the annotation's sigmaNought LUT, no noise removal, power mean of the 8 dates, dB | `channel_notes.sigma0_vv_db` |
| geolocation | moved from the DSM surface to the ground model, (h_DSM - h_ground) cot(theta) towards the sensor | docstring; `geometry.geocoding_surface` |
| lattice | EPSG:3857, 8 x 0.53998256 = 4.31986048 units (3.31 m), 9278 x 9405, origin (12934210.89940096, 4870305.74208256); gdalwarp `-geoloc`, bilinear; 99.993 % covered | `grid`, `coverage_of_aoi_lattice` |
| channels | `re_<i>_<j>`, `im_<i>_<j>` of the 7 pairs (`re_0606_0618` ... `im_0817_0829`), `coh_mean`, `sigma0_vv_db`, `amp_dispersion_vv` | `channels` |
| tiles | 128 x 128 px over each tile's extent (6.63 m), GDAL average, NaN = no data; 561 / 186 / 184 tiles | `--px` default; `tiles` |
| statistics | per channel over the pixels where every channel has data: `train_geo` (the 561 geographic training tiles, 9 191 424 px) and `train_orig` (the 558 `bdh_train` tiles of `metadata/tiles_index.csv`, 9 142 272 px); the model reads `train_orig` | `normalisation`; `MODEL.WISHNET.INSAR.NORM_KEY` in `wishnet/config.py` |

Both pixel counts are the tile count times 128 x 128: every pixel of the
training tiles has data in all channels.  The 558 `bdh_train` tiles of
`metadata/tiles_index.csv` are the `orig_split` training tiles of
`metadata/splits.csv` that `make_orig_split.py` uses (checked on the dataset's
metadata).  `--track desc` (Sentinel-1B, relative orbit 47, 2021) is kept in
the script but not used for the paper's data; it needs `--orbits` (POEORB, as
in its coregistration).

## Original split

The paper uses BDH's original train / val split.  `tools/labels/make_orig_split.py`
links the tiles of `<BDH-RGB-SAR>/s1_insar_asc/<geographic split>/` into
`datasets/bdh/s1_insar_asc/{train,val}/` and `S1_INSAR_ASC.json` into
`datasets/bdh/s1_insar_asc/`:

```bash
python tools/labels/make_orig_split.py --src <BDH-RGB-SAR> --dst datasets/bdh --sar-dir sar_gf3 s1_insar_asc
```
