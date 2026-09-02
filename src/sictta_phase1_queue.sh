#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhaoruijin/MOURUI/sictta_reproduction_20260827
WORK=$ROOT/work/SicTTA_table2
RUNTIME=$ROOT/runtime_site
PY=/home/zhaoruijin/MOURUI/venv/bin/python
export PYTHONPATH=$RUNTIME:$WORK
export OMP_NUM_THREADS=4
case "$ROOT" in /home/zhaoruijin/MOURUI/*) ;; *) exit 2 ;; esac
mkdir -p "$ROOT/logs" "$ROOT/outputs/phase1_20260828" "$ROOT/archives/phase1_20260828"
echo "PHASE1_START $(date -Is)"
echo '[1/6] Immutable snapshot'
cp -an "$ROOT/repos/SicTTA/sotas/sictta.py" "$ROOT/archives/phase1_20260828/" || true
cp -an "$ROOT/outputs/source_train_seed1/best.pth" "$ROOT/archives/phase1_20260828/" || true
cp -an "$ROOT/outputs/table2_source/source_summary.json" "$ROOT/archives/phase1_20260828/" || true
cp -an "$ROOT/outputs/table2_cotta_upstream/cotta_summary.json" "$ROOT/archives/phase1_20260828/" || true
cp -an "$ROOT/outputs/table2_sictta_official/sictta_summary.json" "$ROOT/archives/phase1_20260828/" || true
find "$ROOT/archives/phase1_20260828" -type f -print0 | sort -z | xargs -0 sha256sum > "$ROOT/archives/phase1_20260828/SHA256SUMS"
echo '[2/6] Source-only anatomy statistics'
cd "$WORK"
"$PY" build_anatomy_stats.py > "$ROOT/logs/anatomy_stats_20260828.log" 2>&1
"$PY" -m py_compile sictta_anatomy_v1.py table2_anatomy_eval.py
echo '[3/6] Seeds 2 and 3 source training'
train() { local s=$1; local g=$2; CUDA_VISIBLE_DEVICES=$g "$PY" -u table2_train_source.py --epochs 200 --batch-size 12 --seed "$s" --gpu 0 --num-workers 0 --output "$ROOT/outputs/source_train_seed${s}" > "$ROOT/logs/source_train_seed${s}_20260828.log" 2>&1; }
train 2 0 & p2=$!
train 3 1 & p3=$!
wait $p2 $p3
echo '[4/6] Source, CoTTA, SicTTA controls for seeds 2 and 3'
eval_one() { local s=$1; local method=$2; local g=$3; local ck="$ROOT/outputs/source_train_seed${s}/best.pth"; local out="$ROOT/outputs/phase1_20260828/seed${s}_${method}"; if [ "$method" = source ]; then CUDA_VISIBLE_DEVICES=$g "$PY" -u table2_eval_stream.py --method source --ckpt "$ck" --output "$out" --gpu 0; elif [ "$method" = cotta ]; then CUDA_VISIBLE_DEVICES=$g "$PY" -u table2_eval_faithful.py --method cotta --ckpt "$ck" --output "$out" --gpu 0 --lr 1e-3 --restore .001 --ap .1 --symmetric; else CUDA_VISIBLE_DEVICES=$g "$PY" -u table2_eval_faithful.py --method sictta --ckpt "$ck" --output "$out" --gpu 0; fi; }
for method in source cotta sictta; do
  eval_one 2 "$method" 0 > "$ROOT/logs/phase1_seed2_${method}_20260828.log" 2>&1 & q2=$!
  eval_one 3 "$method" 1 > "$ROOT/logs/phase1_seed3_${method}_20260828.log" 2>&1 & q3=$!
  wait $q2 $q3
done
echo '[5/6] Anatomy-gated SicTTA v1, both stream orders, seeds 1-3'
for s in 1 2 3; do
  for order in CD DC; do
    g=$((s % 2)); ck="$ROOT/outputs/source_train_seed${s}/best.pth"; out="$ROOT/outputs/phase1_20260828/seed${s}_anatomy_${order}"; CUDA_VISIBLE_DEVICES=$g "$PY" -u table2_anatomy_eval.py --method anatomy --order "$order" --ckpt "$ck" --output "$out" --gpu 0 > "$ROOT/logs/phase1_seed${s}_anatomy_${order}_20260828.log" 2>&1
  done
done
echo '[6/6] Official SicTTA reverse-order control'
for s in 1 2 3; do
  g=$((s % 2)); ck="$ROOT/outputs/source_train_seed${s}/best.pth"; out="$ROOT/outputs/phase1_20260828/seed${s}_official_DC"; CUDA_VISIBLE_DEVICES=$g "$PY" -u table2_anatomy_eval.py --method official --order DC --ckpt "$ck" --output "$out" --gpu 0 > "$ROOT/logs/phase1_seed${s}_official_DC_20260828.log" 2>&1
done
echo "PHASE1_COMPLETE $(date -Is)"
