# Tests

## Checks of this code

```bash
python tests/check_schedule.py   # MAX_ITER 22000 against the paper runs' 24000 stopped at 22000
python tests/test_flip.py        # the up-down flip averaging (CPU, fake dataset)
```

`check_schedule.py` builds detectron2's learning-rate schedule for both and
compares the rate of every step 0-21999 bit for bit, base rate and the optical
backbone's, and the steps of the validations, the periodic checkpoints and
`model_final`.  `test_flip.py` checks that a mapped validation sample flipped
twice is the sample again, that the second pass is written back to the
original frame, and the pairing and averaging on synthetic detections.

## Equivalence checks (`equivalence/`)

These tools compare this code with another checkout: the original
implementation the paper's runs were trained with (its training script
`train_gfoptnet.py`, its configuration under `MODEL.GFOPT`), or an earlier
version of this code.  Each one runs from the root of the checkout it tests
and imports that checkout's own package (`wishnet`, or `gfoptnet` under the
working name).  The check builds each model with both codes from the same
seed and compares:

1. the state_dict (names, shapes, initial values);
2. the evaluation outputs on the validation tiles from the same weights;
3. every loss term of N training steps and the parameters after them.

```bash
python tests/equivalence/make_fake_bdh.py <fake root> --size 512     # or the real dataset
# in the other checkout:
python <this repo>/tests/equivalence/run_side.py --side ref --mode init \
    --config-file <its config> --out <dir>/ref <overrides> [its own settings, below]
# in this checkout:
python tests/equivalence/run_side.py --side this --mode init \
    --config-file configs/bdh/<model>.yaml --out <dir>/this <overrides>
# both sides then --mode run --weights <dir>/ref/init.pt --steps N [--eval-tiles 0], and
python tests/equivalence/compare.py <dir>/ref <dir>/this [--noise <dir>/this_2]
```

The configurations are compared by `dump_config.py` (run in each checkout) and
`compare_configs.py` (which maps the working name's namespace and registry
names onto this code's), checkpoints by `check_weights.py`, and whole
evaluations (folders of `tools/evaluate.py`, or the original implementation's)
by `compare_eval.py`.

The forward is deterministic; training is not, run to run.  On a CPU with
several threads some reductions of the backward vary between runs; with
`run_side.py --threads 1 --deterministic` (one thread,
`torch.use_deterministic_algorithms(True)`) CPU training is deterministic, and
`--checksums` records a bit-exact checksum of every gradient and every
parameter after each step, which `compare.py` compares step by step.  On a GPU
MaskDINO's deformable attention accumulates its gradients with atomic
additions: two trainings of the same code differ from the second step on.
There the initial values and the first step's losses are compared bit for
bit, and over the later steps the other side's difference from one run of
this code is set against a second run of this code (`compare.py --noise`: the
largest differences of the total loss and of the final parameters at most
three times those of the second run).

The two sides must see the same data and paths (`<overrides>`: the dataset
and pretrained-weight paths).  Two settings of the original implementation
make it draw different random numbers or read different SAR windows, and are
set on its side only:

- `MODEL.GFOPT.MONITOR_PERIOD 0`: its SAR monitor decodes once more at step 0
  and every 100 steps, drawing denoising noise;
- `MODEL.GFOPT.SAR.CONTEXT_PX 256` on tiles smaller than its context window
  (the fake dataset's 512 px tiles); the real 2048 tiles need nothing.

Against it, `compare_configs.py` takes the differences by design with
`--allow` (its header lists them).  On a CPU `run_side.py` keeps MaskDINO's
`.cuda()` and `.to("cuda")` on the CPU.

## Results against the original implementation

### On the CPU (fake dataset)

- The four models (`opt`, `gf3`, `gf3_insar`, `wishnet` against the original
  implementation's runs of the paper): initial values, evaluation outputs and
  two training steps bit for bit.
- WISH-Net, one thread and deterministic algorithms, 12 steps: every loss term
  (311), every gradient (829 tensors) and every parameter after each step (865
  tensors) bit for bit.
- WISH-Net, 16 threads, 200 steps: bit for bit up to step 6; from step 7 the
  losses part in the last digits (`loss_mask_dn` 0.39221954 against 0.39221945)
  and the runs drift apart.  Two runs of this code part at the same step with
  the same two values, each run taking one of them: the multi-threaded backward
  varies from run to run, the two codes do not differ.

### On the real data

One RTX 5070 (12 GB; PyTorch 2.7.1, CUDA 12.8), the paper's data
(every label file and tile checked against `datasets/bdh_md5.txt` and
`datasets/bdh_tiles_md5.txt`), `CUDNN_BENCHMARK False` on both sides:

| check | result |
|---|---|
| configurations, four models | only paths, `SOLVER.MAX_ITER` and the two optional readouts differ |
| the paper's WISH-Net weights (`model_best_h.pth`) loaded strictly | no tensor missing, extra or reshaped |
| whole-tile evaluation of the 373 validation tiles, against the original implementation's evaluation | 167 850 detections and 134 metrics, none different |
| the same with the flip averaging, against its flip-averaging tool | orig and avg: 167 850 detections and 134 metrics, none different |
| learning-rate schedule, steps 0-21999, the training machines' detectron2 and fvcore | bit for bit |
| label tools (`make_labels_v2.sh`) on a training machine | the 12 label files bit for bit |

Training, 200 steps of WISH-Net from the same initial weights, one tile per
batch and the GF-3 window read at 512 px (`SOLVER.IMS_PER_BATCH 1`,
`MODEL.WISHNET.SAR.INPUT_PX 512`; the full setting needs 19.4 GiB), two runs of
each code, run alternately (`compare_runs.py`).  Every pair is bit-identical at
step 0 and differs from step 1 on.  Over steps 1-199:

| pair | total loss, median difference | largest | parameters after 200 steps, largest difference |
|---|---|---|---|
| this code, run a and run b | 1.098 | 34.96 | 0.0151 |
| original implementation, run a and run b | 0.805 | 11.88 | 0.0423 |
| this code a, original implementation a | 0.105 | 3.98 | 0.0422 |
| this code a, original implementation b | 0.940 | 11.68 | 0.0147 |
| this code b, original implementation a | 0.980 | 35.30 | 0.0421 |
| this code b, original implementation b | 0.495 | 27.76 | 0.0132 |

The rule registered before these runs: W, the larger within-code median
(1.098); the codes differ systematically on the GPU if every cross-code median
exceeds 3 W.  The cross-code medians are 0.10, 0.86, 0.89 and 0.45 W, and the
cross-code parameter differences are no larger than the within-code ones:
no systematic difference.

An earlier comparison with three runs (the original implementation once, this
code twice, from initial weights made the same way) gave a median difference
of the total loss of 1.88 and 2.02 between the original implementation's run
and the two runs of this code and 0.47 between the latter, largest
differences 26.28, 32.09 and 11.07, and parameter differences of 0.0117 (the
original implementation and the first run) and 0.0425 (the two runs of this
code).  With three runs one of them is always the farthest; the 2 x 2
comparison above, with no systematic difference, puts it down to chance.  The
denoising loss terms are missing in the steps whose tile holds more buildings
than `MaskDINO.DN_NUM` (249 to 312 terms per step).

## The rename to WISH-Net

The code under the working name (package `gfoptnet`, namespace `MODEL.GFOPT`)
against the renamed code, each side with its own tools, on the CPU (fake
dataset, one thread and deterministic algorithms), for the four models:

| check | result |
|---|---|
| configurations (`dump_config.py`, `compare_configs.py --same-keys`) | 390 keys each, the same after the mapping; only `OUTPUT_DIR` differs (a path) |
| state_dict | the same names (1190 for WISH-Net), shapes and initial values; no name holds the network's name |
| evaluation outputs from the same weights | bit for bit |
| two training steps | every loss term, every gradient and every parameter after each step bit for bit (WISH-Net: 311 terms, 829 gradients, 865 parameters) |

On the GPU (one RTX 5070, `CUDNN_BENCHMARK False`), the paper's data:

| check | result |
|---|---|
| the paper's WISH-Net weights (`model_best_h.pth`) loaded strictly by the renamed code (`check_weights.py`) | 1190 tensors, none missing, extra or reshaped |
| whole-tile evaluation of the first 20 validation tiles with the flip averaging (`tools/evaluate.py --flip --n 20`), working name against renamed (`compare_eval.py`) | orig and avg: 9000 detections and 134 metrics, none different |
