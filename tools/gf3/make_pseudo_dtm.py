# -*- coding: utf-8 -*-
"""A bare-earth height model for geocoding the BDH SAR, from the DSM.

Why: Copernicus GLO-30 is a DSM.  Over Beijing's built-up blocks it sits a
median 6.7 m (p90 12.4 m) above the ground, so a geocoder that takes it as the
terrain places every ground-level scatterer -- the wall-ground dihedral of each
building, roads between the blocks -- dz / tan(theta) too far from the sensor.
Stacked over 67 552 walls facing the sensor, the dihedral of the DSM-geocoded
GF-3 product peaks 11.1 m behind the annotated footprint edge; geocoded on this
model with the same constant corrections it peaks at +0.3 m (train / val /
test +1.0 / +0.4 / -1.4 m).  The existing S1 products (ISCE, same DSM) show the
same bias: descending -6.5 m, ascending -10.6 m (registration_check_gf3.py profile).

What: the AOI is the flat North China plain, so the ground is the DSM's local
minimum over ~750 m (every window holds streets, courtyards or water), smoothed
(sigma 240 m).  Where the ground surface itself has relief (> 25 m over 750 m:
the Western Hills, 0.35 % of the AOI) the model blends back to the DSM, and it
is never above the DSM.  Written as ISCE-style float32 (same grid, ellipsoidal
heights as the DSM) with a .vrt, so geocode_gf3.py --dem takes it directly.

    python tools/gf3/make_pseudo_dtm.py --dsm <dem>/beijing.dem.wgs84 --out <dem>/beijing.pdtm.wgs84
"""
import argparse
import os

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter

SHAPE = (7200, 10800)                 # 1 arcsec, lon 115-118, lat 39-41


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsm", required=True,
                    help="Copernicus GLO-30 DSM, ISCE-style float32 on SHAPE (ellipsoidal heights), with its "
                         ".vrt next to it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=25, help="minimum filter, px (1 arcsec ~ 30 m)")
    ap.add_argument("--sigma", type=float, default=8.0, help="smoothing, px")
    a = ap.parse_args()
    dsm = np.fromfile(a.dsm, np.float32).reshape(SHAPE)
    ground = gaussian_filter(minimum_filter(dsm, a.window), a.sigma)
    relief = gaussian_filter(maximum_filter(ground, a.window) - minimum_filter(ground, a.window), a.sigma)
    w = np.clip((relief - 25.0) / 25.0, 0, 1)        # 0 in the plain, 1 in hills
    dtm = np.minimum((1 - w) * ground + w * dsm, dsm).astype(np.float32)
    dtm.tofile(a.out)
    # the DSM's .vrt with its raster (the DSM's file name) replaced by the output's path
    vrt = open(a.dsm + ".vrt").read().replace(os.path.basename(a.dsm), os.path.abspath(a.out)).replace(
        'relativeToVRT="1"', 'relativeToVRT="0"')
    open(a.out + ".vrt", "w").write(vrt)
    r0, r1 = int((41.0 - 40.00) * 3600), int((41.0 - 39.73) * 3600)
    c0, c1 = int((116.19 - 115.0) * 3600), int((116.55 - 115.0) * 3600)
    d = (dsm - dtm)[r0:r1, c0:c1]
    print("AOI: DSM - DTM median %.1f  p75 %.1f  p90 %.1f m; hills blended back on %.2f%% of the AOI"
          % (*np.percentile(d, [50, 75, 90]), 100 * (w[r0:r1, c0:c1] > 0).mean()))


if __name__ == "__main__":
    main()
