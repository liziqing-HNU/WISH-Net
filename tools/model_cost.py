# -*- coding: utf-8 -*-
"""Model cost for the paper: parameters per part and the inference time of
one whole validation tile.

Parameters, counted once each (MaskDINO keeps one Parameter under several
names, e.g. the box head as _bbox_embed, bbox_embed.<i> and
decoder.bbox_embed.<i>), each in the first part of PARTS any of its names
falls in:

    heads                 class, mask, box, height and offset heads, the
                          proposal chain's own heads
    decoder: SAR fusion   the decoder's SAR reads and gated fusion
    decoder               the rest of the decoder: layers, query selection,
                          denoising embedding, norms
    optical backbone      ResNet-50
    optical pixel decoder the optical encoder (MaskDINOEncoder)
    GF-3 branch           its ResNet-18
    InSAR branch          its ResNet-18 and the projections into the encoder
    joint SAR encoder     the deformable encoder of the GF-3 (and InSAR) levels
    exchange              the two-way exchange at the encoder stage
    other                 anything else (listed by name; none expected)

Time: the model built from the configuration (with --weights loaded, else as
initialised: the cost does not depend on the weights), eval mode, no
gradient; the first --warmup + --n validation tiles of DATASETS.TEST through
the test loader, batch 1, the whole-tile evaluation settings
(INPUT.EVAL_CROP_SIZE 2048, BDH_EVAL.MAX_DETS 600) applied before the
overrides; only the model call is timed (CUDA synchronised before and after):
the whole tile's inference -- with INPUT.TILE_SCALE 0.5 the 2048 tile
resampled to one 1024 input, a single forward -- not the loading and mapping
of the tile.  Reported: the
median and quartiles over the --n tiles after the --warmup ones, the peak GPU
memory allocated over all the tiles run (the weights included), the GPU's
name, and the other compute processes nvidia-smi lists on that GPU just
before the timing (any there: the times are not clean, a warning says so).
FLOPs are not reported: the multi-scale deformable attention is a custom
operator that the usual counters (fvcore, thop) do not count.

    python tools/model_cost.py --config-file configs/bdh/wishnet.yaml --out <new folder> \\
        [--weights <w>] [--n 40] [--warmup 10] [config overrides ...]

The last lines are the table.  --out, a new or empty folder, gets it as
cost.json, and work/ (the effective config.yaml and the log of the setup).
"""
import argparse
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHAIN_EVAL = ["INPUT.EVAL_CROP_SIZE", "2048", "BDH_EVAL.MAX_DETS", "600"]
P = r"^sem_seg_head\.predictor\."
# (part, pattern), in priority order: a parameter goes to the first part any of its names matches
PARTS = (
    ("heads", P + r"(class_embed|mask_embed|_bbox_embed|bbox_embed|decoder\.bbox_embed|height_embed|offset_embed"
              r"|proposal_heads)\b"),
    ("decoder: SAR fusion", P + r"(sar_read|sar_fuse)\b"),
    ("decoder", P),
    ("optical backbone", r"^backbone\."),
    ("optical pixel decoder", r"^sem_seg_head\.pixel_decoder\."),
    ("GF-3 branch", r"^sar_branch\.backbone\."),
    ("InSAR branch", r"^sar_branch\.(insar|insar_proj)\."),
    ("joint SAR encoder", r"^sar_branch\.encoder\."),
    ("exchange", r"^exchange\."),
)


def all_named_parameters(module, prefix=""):
    """Every (name, Parameter) under every path to it -- a shared one under each of
    its names (named_parameters gives one name only, and its remove_duplicate
    argument is newer than some of the servers' torch)."""
    for n, p in module._parameters.items():
        if p is not None:
            yield prefix + n, p
    for n, m in module._modules.items():
        if m is not None:
            yield from all_named_parameters(m, prefix + n + ".")


def count_parts(model):
    """-> ({part: parameters}, {other parameter name: numel}); every Parameter once."""
    names = {}
    for n, p in all_named_parameters(model):
        names.setdefault(id(p), (p, []))[1].append(n)
    counts = {part: 0 for part, _ in PARTS}
    other = {}
    for p, ns in names.values():
        part = next((part for part, pat in PARTS if any(re.match(pat, n) for n in ns)), None)
        if part is None:
            other[ns[0]] = p.numel()
        else:
            counts[part] += p.numel()
    return counts, other


def time_model(model, batches, warmup, cuda):
    """Seconds of each model call after the first ``warmup`` ones (batch 1, no gradient)."""
    import torch
    times = []
    with torch.no_grad():
        for i, batch in enumerate(batches):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(batch)
            if cuda:
                torch.cuda.synchronize()
            if i >= warmup:
                times.append(time.perf_counter() - t0)
    return times


def other_gpu_processes(smi_csv, mine_uuid, pid):
    """nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory (csv, noheader,
    nounits) -> ["pid <p> <MiB> MiB"] of the processes on this GPU other than
    ``pid``; every GPU's when ``mine_uuid`` is None."""
    bare = lambda u: u.strip().lower().replace("gpu-", "")
    rows = [[x.strip() for x in line.split(",")] for line in smi_csv.splitlines()]
    return ["pid %s %s MiB" % (p, m) for u, p, m in (r for r in rows if len(r) == 3)
            if p != str(pid) and (mine_uuid is None or bare(u) == bare(mine_uuid))]


def gpu_processes_now(torch):
    """other_gpu_processes for the current CUDA device; None if nvidia-smi cannot tell."""
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    try:            # torch >= 2.0; without it every GPU's processes count
        mine = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    except Exception:
        mine = ""
    if not re.fullmatch(r"(gpu-)?[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", mine.lower()):
        mine = None
    return other_gpu_processes(r.stdout, mine, os.getpid())


def quartiles(v):
    v = sorted(v)
    q = statistics.quantiles(v, n=4) if len(v) > 1 else [v[0]] * 3
    return statistics.median(v), q[0], q[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weights", default="")
    ap.add_argument("--n", type=int, default=40, help="tiles timed")
    ap.add_argument("--warmup", type=int, default=10, help="tiles run first, not timed")
    ap.add_argument("--out", required=True, help="a new or empty folder")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    if a.n < 1 or a.warmup < 0:
        sys.exit("--n at least 1, --warmup at least 0")
    if os.path.exists(a.out) and os.listdir(a.out):
        sys.exit("%s is not empty: --out must be a new folder" % a.out)
    import itertools

    import torch
    from detectron2.checkpoint import DetectionCheckpointer

    import train_net as T

    work = os.path.join(a.out, "work")
    args = argparse.Namespace(config_file=a.config_file, opts=CHAIN_EVAL + list(a.opts) + ["OUTPUT_DIR", work],
                              eval_only=True, resume=False, num_gpus=1, machine_rank=0, num_machines=1,
                              dist_url="auto")
    cfg = T.setup(args)
    model = T.Trainer.build_model(cfg)
    if a.weights:
        DetectionCheckpointer(model).load(a.weights)
    model.eval()
    counts, other = count_parts(model)
    total = sum(counts.values()) + sum(other.values())
    cuda = torch.cuda.is_available() and next(model.parameters()).is_cuda
    loader = T.Trainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])
    batches = itertools.islice(iter(loader), a.warmup + a.n)
    others = gpu_processes_now(torch) if cuda else None
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    times = time_model(model, batches, a.warmup, cuda)
    if len(times) < a.n:
        sys.exit("only %d tiles timed of %d (the split has fewer than --warmup + --n)" % (len(times), a.n))
    med, q1, q3 = quartiles(times)
    res = {"config": a.config_file, "weights": a.weights or None, "opts": CHAIN_EVAL + list(a.opts),
           "gpu": torch.cuda.get_device_name() if cuda else "cpu", "torch": torch.__version__,
           "params": counts, "params_other": other, "params_total": total,
           "time_s_per_tile": {"median": med, "q25": q1, "q75": q3, "n": len(times), "warmup": a.warmup},
           "peak_memory_gib": torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None,
           "other_gpu_processes": others,
           "flops": "not reported: the multi-scale deformable attention is a custom operator the counters miss"}
    print("[cost] %s on %s" % (a.config_file, res["gpu"]))
    for part, _ in PARTS:
        if counts[part]:
            print("[cost]   %-22s %12s  %6.2f M" % (part, "{:,}".format(counts[part]), counts[part] / 1e6))
    for n, k in other.items():
        print("[cost]   other: %s %d" % (n, k))
    print("[cost]   %-22s %12s  %6.2f M" % ("total", "{:,}".format(total), total / 1e6))
    print("[cost]   time per whole tile: median %.3f s (quartiles %.3f-%.3f, %d tiles after %d warm-up), "
          "peak memory %s" % (med, q1, q3, len(times), a.warmup,
                              "%.2f GiB" % res["peak_memory_gib"] if cuda else "-"))
    if others:
        print("[cost]   WARNING: other processes on this GPU while timing (%s): the times are not clean"
              % "; ".join(others))
    elif cuda and others is None:
        print("[cost]   nvidia-smi could not tell whether the GPU was free")
    with open(os.path.join(a.out, "cost.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
