# -*- coding: utf-8 -*-
"""Print configuration values of a configuration file with overrides, one
per line (for shell scripts).

    python tools/cfg_get.py <config file> KEY [KEY ...] [-- config overrides ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from maskdino import add_maskdino_config

from wishnet.config import add_wishnet_config


def main():
    args = sys.argv[1:]
    opts = []
    if "--" in args:
        i = args.index("--")
        args, opts = args[:i], args[i + 1:]
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    add_wishnet_config(cfg)
    cfg.merge_from_file(args[0])
    cfg.merge_from_list(opts)
    for key in args[1:]:
        v = cfg
        for part in key.split("."):
            v = getattr(v, part)
        print(v)


if __name__ == "__main__":
    main()
