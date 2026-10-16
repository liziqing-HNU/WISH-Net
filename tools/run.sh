#!/bin/bash
# One model end to end on one GPU, as in the paper:
#
#   1. train (an interrupted training resumes from last_checkpoint);
#   2. whole-tile validation of the candidates model_best (the lowest Hungarian
#      height MAE of the validations), model_best_h (the weight-selection rule
#      over the validations) and model_final; the same rule
#      (wishnet/evaluation/selection.py: drop F1 more than 1 point below the
#      highest, then the lowest best-F1 MAE) picks the run's weights;
#   3. the chosen weights on the validation split with the up-down flip
#      averaging (tools/evaluate.py --flip): final_val/orig/ is the model as it
#      is (F1, mAP50, the best-F1 threshold swept on the split), final_val/avg/
#      the same detections with the flip-averaged heights (MAE, RMSE, per band,
#      the BDH-protocol MAE of the tables);
#   4. mode fusion: the zeroing diagnostics, without the flip averaging as in
#      the paper, at the full input's frozen thresholds -- GF-3 and InSAR
#      together (zero_val/) and, with InSAR, each alone (zero_gf3_val/,
#      zero_insar_val/); tools/paired_compare.py pairs each with final_val/orig/.
#
#   CFG=configs/bdh/wishnet.yaml bash tools/run.sh [config overrides ...]
#
# PY (default python) is the interpreter of the training and evaluation; the
# periodic checkpoints (model_00*.pth) are removed at the end unless KEEP=1.
# Everything goes to the configuration's OUTPUT_DIR: the training, eval_<candidate>/,
# choice.txt and choice_reason.txt (the weights and why), final_val/ and the
# zero_*val/ folders, run.log.  Finished steps are skipped when the script runs again.
set -eo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:?CFG=<config file>}
PY=${PY:-python}
export PYTHONUNBUFFERED=1
EXTRA=("$@")
mapfile -t cfg < <($PY tools/cfg_get.py "$CFG" OUTPUT_DIR MODEL.WISHNET.MODE MODEL.WISHNET.INSAR.ENABLED \
                   -- "${EXTRA[@]}" | tail -n 3)
[ ${#cfg[@]} -eq 3 ] || { echo "cannot read the configuration $CFG" >&2; exit 1; }
RUN=${cfg[0]%$'\r'} MODE=${cfg[1]%$'\r'} INSAR=${cfg[2]%$'\r'}
mkdir -p "$RUN"
say () { echo "### $(date '+%F %T') $*" | tee -a "$RUN/run.log"; }

# ------------------------------------------------------------------ train
if [ ! -f "$RUN/.train_done" ]; then
  resume=()
  [ -s "$RUN/last_checkpoint" ] && resume=(--resume)
  say "train $CFG ${resume[*]}"
  $PY train_net.py --num-gpus 1 "${resume[@]}" --config-file "$CFG" "${EXTRA[@]}" > "$RUN/train.log" 2>&1 \
    || { say "training FAILED (see $RUN/train.log)"; exit 1; }
  [ -s "$RUN/model_final.pth" ] || { say "training ended without model_final.pth (see $RUN/train.log)"; exit 1; }
  touch "$RUN/.train_done"
fi

# ------------------------------------------------------- choose the weights
V=inference_bdh_val/bdh_metrics.json
if [ ! -s "$RUN/choice.txt" ]; then
  cands=()
  for c in model_best model_best_h model_final; do
    [ -s "$RUN/$c.pth" ] || continue
    d=$RUN/eval_$c
    if [ ! -s "$d/$V" ]; then
      $PY train_net.py --num-gpus 1 --eval-only --config-file "$CFG" "${EXTRA[@]}" \
          MODEL.WEIGHTS "$RUN/$c.pth" OUTPUT_DIR "$d" > "$d.log" 2>&1 || say "validation of $c FAILED (see $d.log)"
    fi
    [ -s "$d/$V" ] && cands+=("$c=$d/$V")
  done
  [ ${#cands[@]} -gt 0 ] || { say "no candidate validated (see $RUN/eval_*.log)"; exit 1; }
  pick=$($PY tools/select_weights.py --reason-file "$RUN/choice_reason.txt" "${cands[@]}") \
    || { say "no weights chosen: $(cat "$RUN/choice_reason.txt")"; exit 1; }
  echo "$RUN/$pick.pth" > "$RUN/choice.txt"
  say "weights $pick: $(cat "$RUN/choice_reason.txt")"
fi
W=$(cat "$RUN/choice.txt")

# --------------------------------- the reported evaluation (flip averaging)
if [ ! -s "$RUN/final_val/evaluate.json" ]; then
  say "final evaluation of $(basename "$W") with the flip averaging"
  $PY tools/evaluate.py --config-file "$CFG" --weights "$W" --out "$RUN/final_val" --flip "${EXTRA[@]}" \
      > "$RUN/final_val.log" 2>&1 || { say "final evaluation FAILED (see $RUN/final_val.log)"; exit 1; }
fi
# the operating points of the full input, frozen for the zeroing diagnostics
read -r TF TM < <(python -c "import json, sys; o = json.load(open(sys.argv[1]))['protocol']['orig']; \
print(o['f1op/score_thresh'], o['maeop/score_thresh'])" "$RUN/final_val/evaluate.json")
TM=${TM%$'\r'}

# ------------------------------------------- zeroing (mode fusion; unflipped)
if [ "$MODE" = fusion ]; then
  zs=(both)
  [ "$INSAR" = True ] && zs+=(gf3 insar)
  for z in "${zs[@]}"; do
    out=$RUN/zero_val
    [ "$z" = both ] || out=$RUN/zero_${z}_val
    [ -s "$out/evaluate.json" ] && continue
    say "zeroing $z at the frozen thresholds $TF (best F1) and $TM (best MAE)"
    $PY tools/evaluate.py --config-file "$CFG" --weights "$W" --out "$out" --zero "$z" \
        --score-thresh "$TF" --score-thresh-mae "$TM" "${EXTRA[@]}" > "$out.log" 2>&1 \
      || { say "zeroing $z FAILED (see $out.log)"; exit 1; }
  done
fi
# the periodic checkpoints are not needed any more (KEEP=1 keeps them)
[ "${KEEP:-0}" = 1 ] || rm -f "$RUN"/model_00*.pth
say "done: $(basename "$W"); $RUN/final_val/evaluate.json"
