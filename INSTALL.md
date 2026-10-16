# Installation

The paper's runs used Linux, Python 3.10, PyTorch 2.4.1 with CUDA 12.1
(torchvision 0.19.1), detectron2 0.6 built from source, numpy 1.26.4, on
RTX 3090, 4090 and 4090 D GPUs (24 GB); the CUDA extensions were compiled with
the CUDA 12.8 toolkit.  Other versions of PyTorch and CUDA work as long as
detectron2 and the extension are compiled for them.

```bash
conda create -n wishnet python=3.10 -y
conda activate wishnet
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# detectron2 0.6 from source
git clone https://github.com/facebookresearch/detectron2.git
pip install --no-build-isolation -e detectron2

# MaskDINO's multi-scale deformable attention (CUDA)
cd maskdino/modeling/pixel_decoder/ops
python setup.py build install
cd -
python -c "import MultiScaleDeformableAttention, detectron2; print('ok')"
```

`CUDA_HOME` must point to a CUDA toolkit whose compiler supports your GPU
(`TORCH_CUDA_ARCH_LIST`, e.g. `8.6` for the 3090, `8.9` for the 4090).

The data chains need more: `tools/gf3/` GDAL, scipy, OpenCV and tifffile;
`tools/s1insar/` additionally ISCE2 (topsStack) and the `gdalwarp` command.
Their READMEs list them.

## Check

```bash
python tests/check_schedule.py      # the learning-rate schedule (detectron2, fvcore)
python tests/test_flip.py           # the flip averaging, on a fake dataset (CPU)
```

`tests/README.md` describes these checks and the equivalence checks against the original
implementation.
