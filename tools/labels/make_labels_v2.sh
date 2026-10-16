#!/bin/bash
# The label files the paper's runs read, made from the original split's
# annotations_fixed/instances_{train,val}.json (tools/labels/make_orig_split.py):
#
#   prepared/        the evaluation crops of the BDH labels (the whole 2048 tile)
#                    and their ignore regions (prepare_bdh.py, make_ignore_regions.py);
#   annotations_v2/  the reviewed labels (datasets/bdh_labels_v2_changes.json applied
#   prepared_v2/     by apply_label_changes.py): the training file and the
#                    evaluation crops with their ignore regions.
#
#   bash tools/labels/make_labels_v2.sh datasets/bdh
#
# The files are then checked against datasets/bdh_md5.txt.  PY (default python)
# is the interpreter; numpy, shapely.
set -eo pipefail
cd "$(dirname "$0")/../.."
ROOT=${1:?the dataset folder (datasets/bdh)}
CHANGES=${CHANGES:-datasets/bdh_labels_v2_changes.json}
MD5=$(pwd)/datasets/bdh_md5.txt
PY=${PY:-python}
$PY tools/labels/prepare_bdh.py --root "$ROOT" --out "$ROOT/prepared" --splits train val --crop 2048
$PY tools/labels/make_ignore_regions.py --root "$ROOT" --prepared "$ROOT/prepared" --splits train val --crops 2048
W=$ROOT/labels_v2_work
rm -rf "$W"
mkdir -p "$W" "$ROOT/annotations_v2" "$ROOT/prepared_v2"
for s in train val; do
  $PY tools/labels/apply_label_changes.py --changes "$CHANGES" --split $s \
      --crop "$ROOT/prepared/instances_${s}_crop2048.json" --whole "$ROOT/annotations_fixed/instances_$s.json" \
      --out "$W/$s" > "$W/$s.log"
  tail -n 1 "$W/$s.log"
  cp "$W/$s/instances_$s.json" "$ROOT/annotations_v2/"
  cp "$W/$s/instances_${s}_crop2048.json" "$W/$s/instances_${s}_crop2048_ignore.json" "$ROOT/prepared_v2/"
done
# the validation buildings whose height or offset is a model's (annotation ids of the
# evaluation crops), for tools/paired_compare.py --exclude-gt / --only-gt
$PY -c "import csv, sys; print('\n'.join(r['v2_ann_id'] for r in csv.DictReader(open(sys.argv[1], encoding='utf-8')) \
if r['v2_ann_id']))" "$W/val/model_sourced.csv" > "$ROOT/prepared_v2/instances_val_crop2048_model_sourced_ids.txt"
echo "model-sourced validation buildings: $(wc -l < "$ROOT/prepared_v2/instances_val_crop2048_model_sourced_ids.txt")"
cd "$ROOT"
grep -v s1_insar_asc "$MD5" | md5sum -c -
