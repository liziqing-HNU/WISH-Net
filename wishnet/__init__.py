# -*- coding: utf-8 -*-
"""WISH-Net: building instances and heights from VHR optical imagery, GF-3 SAR
and Sentinel-1 InSAR, on MaskDINO.

Importing the package registers the meta architecture (WISHNet) and the
transformer decoder (WISHDecoder) with detectron2 / MaskDINO.
"""
from .config import add_wishnet_config, check_config
from .data import WISHDatasetMapper, register_bdh
from .evaluation import BDHEvaluator
from . import modeling  # noqa: F401  (registers WISHNet and WISHDecoder)

__all__ = ["add_wishnet_config", "check_config", "WISHDatasetMapper", "register_bdh", "BDHEvaluator"]
