#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhaoruijin/MOURUI/sictta_reproduction_20260827
WORK=$ROOT/work/SicTTA_table2
RUNTIME=$ROOT/runtime_site
OUT=$ROOT/outputs/source_train_seed1
CKPT=$OUT/best.pth
PY=/home/zhaoruijin/MOURUI/venv/bin/python
export PYTHONPATH=$RUNTIME:$WORK
export CUDA_VISIBLE_DEVICES=0

case "$ROOT" in /home/zhaoruijin/MOURUI/*) ;; *) exit 2 ;; esac
echo "QUEUE_START $(date -Is)"
while true; do
  if [ -s "$OUT/metrics.jsonl" ]; then
    last=$($PY - "$OUT/metrics.jsonl" <<'PY'
import json,sys
last=-1
for line in open(sys.argv[1]):
    try: last=max(last,int(json.loads(line)['epoch']))
    except Exception: pass
print(last)
PY
)
    echo "TRAINING_EPOCH=$last"
    if [ "$last" -ge 200 ] && [ -s "$CKPT" ]; then break; fi
  fi
  sleep 60
done
echo "TRAINING_READY $(date -Is) checkpoint=$(stat -c %s "$CKPT")"
for method in source cotta sictta; do
  echo "RUN_START method=$method $(date -Is)"
  mkdir -p "$ROOT/outputs/table2_$method"
  "$PY" -u "$WORK/table2_eval_stream.py" --method "$method" --ckpt "$CKPT" --output "$ROOT/outputs/table2_$method" --gpu 0
  echo "RUN_DONE method=$method $(date -Is)"
done
echo "QUEUE_COMPLETE $(date -Is)"
