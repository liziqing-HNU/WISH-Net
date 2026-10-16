# -*- coding: utf-8 -*-
"""Dump effective configurations, flattened, as JSON, for compare_configs.py.
Run from the root of the checkout whose configurations are dumped (this code,
or another checkout as in run_side.py), so that its own package is imported.

    python <this repo>/tests/equivalence/dump_config.py <out.json> <config> [<config> ...]
"""
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import yaml
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from maskdino import add_maskdino_config


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, prefix + str(k) + "."))
        else:
            out[prefix + str(k)] = v
    return out


def base_cfg():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    if importlib.util.find_spec("wishnet") is not None:
        from wishnet import add_wishnet_config
        add_wishnet_config(cfg)
        return cfg
    # the code under its earlier name; the original implementation's second
    # dataset adds keys of its own
    from gfoptnet import add_gfoptnet_config
    add_gfoptnet_config(cfg)
    if importlib.util.find_spec("gfoptnet.data.register_dfc") is not None:
        from gfoptnet.data.register_dfc import add_dfc_config
        add_dfc_config(cfg)
    return cfg


def main():
    out = sys.argv[1]
    res = {}
    for f in sys.argv[2:]:
        cfg = base_cfg()
        cfg.merge_from_file(f)
        res[os.path.basename(f)] = flatten(yaml.safe_load(cfg.dump()))
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print("wrote", out, {k: len(v) for k, v in res.items()})


if __name__ == "__main__":
    main()
