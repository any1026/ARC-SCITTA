#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhaoruijin/MOURUI/sictta_reproduction_20260827
WORK=$ROOT/work/SicTTA_table2
RUNTIME=$ROOT/runtime_site
PY=/home/zhaoruijin/MOURUI/venv/bin/python
OUT=$ROOT/outputs/phase1_anatomy_v1_20260829
LOG=$ROOT/logs

case "$ROOT" in
  /home/zhaoruijin/MOURUI/*) ;;
  *) echo "Unsafe ROOT: $ROOT" >&2; exit 2 ;;
esac

export PYTHONPATH=$RUNTIME:$WORK
export OMP_NUM_THREADS=4
mkdir -p "$OUT" "$LOG"
cd "$WORK"

echo "PHASE1_RESUME_START $(date -Is)"
echo "[1/4] Syntax and input validation"
"$PY" -m py_compile sictta_anatomy_v1.py table2_anatomy_eval.py
for seed in 1 2 3; do
  test -s "$ROOT/outputs/source_train_seed${seed}/best.pth"
done
test -s "$ROOT/outputs/anatomy_stats_source_train.json"

run_eval() {
  local method=$1
  local seed=$2
  local order=$3
  local gpu=$4
  local ckpt="$ROOT/outputs/source_train_seed${seed}/best.pth"
  local dst="$OUT/seed${seed}_${method}_${order}"
  local logfile="$LOG/phase1_resume_seed${seed}_${method}_${order}_20260829.log"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u table2_anatomy_eval.py \
    --method "$method" --order "$order" --ckpt "$ckpt" \
    --output "$dst" --gpu 0 > "$logfile" 2>&1
}

echo "[2/4] Anatomy-gated SicTTA v1, seeds 1-3, both orders"
for seed in 1 2 3; do
  run_eval anatomy "$seed" CD 0 & p0=$!
  run_eval anatomy "$seed" DC 1 & p1=$!
  wait "$p0" "$p1"
done

echo "[3/4] Official SicTTA reverse-order controls, seeds 1-3"
run_eval official 1 DC 0 & p0=$!
run_eval official 2 DC 1 & p1=$!
wait "$p0" "$p1"
run_eval official 3 DC 0

echo "[4/4] Result manifest and integrity hashes"
find "$OUT" -type f -name '*.json' -print0 | sort -z | xargs -0 sha256sum > "$OUT/SHA256SUMS"
find "$OUT" -type f -name '*summary.json' -print | sort > "$OUT/summary_files.txt"
echo "PHASE1_RESUME_COMPLETE $(date -Is)"
