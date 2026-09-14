#!/usr/bin/env bash
set -Eeuo pipefail

# Frozen Stage9 paper-table queue. Every mutable/output path is below MOURUI.
ROOT=/home/zhaoruijin/MOURUI/sictta_stage9_paper400_20260914
SRC=$ROOT/src
BASE=/home/zhaoruijin/MOURUI/sictta_reproduction_20260827
V1=/home/zhaoruijin/MOURUI/sictta_v1_recheck_20260902
PY=/home/zhaoruijin/miniconda3/envs/cotta/bin/python
CFG=$SRC/../configs/stage9_anatomy_frozen.json
OUT=$ROOT/outputs/paper400
AGG=$ROOT/outputs/paper400_aggregate
LOG=$ROOT/logs/paper400_queue.log

case "$ROOT" in /home/zhaoruijin/MOURUI/*) ;; *) echo "unsafe ROOT"; exit 2 ;; esac
mkdir -p "$ROOT/logs" "$SRC" "$OUT"
exec > >(tee -a "$LOG") 2>&1
trap 'rc=$?; if [[ $rc -ne 0 ]]; then echo "PAPER_QUEUE_FAILED $(date -Is) rc=$rc"; fi; exit $rc' EXIT

export PYTHONPATH="$SRC:$BASE/repos/SicTTA:$BASE/work/SicTTA_table2:$V1/src"
export OMP_NUM_THREADS=4

echo "PAPER_QUEUE_START $(date -Is)"
test -x "$PY"
test -f "$CFG"
"$PY" - "$CFG" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
assert c["schema"] == "SCITTA-pro-stage9-frozen-v1"
assert c["seeds"] == [1, 2, 3] and c["orders"] == ["CD", "DC"]
assert c["target_images_per_domain"] == 400
assert c["ccd_history_length"] == c["fifo_capacity"] == 40
assert c["top_k"] == 5 and c["variant"] == "anatomy"
assert c["feature_mean"] == [0] * 8 and c["feature_std"] == [1] * 8
assert c["weight"] == [0, 0, 0, 0, 0, 4, 0, 0] and c["bias"] == 0
assert c["target_labels_used_for_adaptation"] is False
print("FROZEN_CONFIG_AUDIT_OK")
PY
test -f "$BASE/manifests/source_train.csv"
test -f "$BASE/outputs/anatomy_stats_source_train.json"
for seed in 1 2 3; do
  test -f "$BASE/outputs/source_train_seed${seed}/best.pth"
  sha256sum "$BASE/outputs/source_train_seed${seed}/best.pth"
done
for domain in c d; do
  test -f "$BASE/manifests/target_domain_${domain}_stream.csv"
  lines=$(wc -l < "$BASE/manifests/target_domain_${domain}_stream.csv")
  [[ "$lines" -eq 401 ]] || { echo "BAD_TARGET_ROWS domain=$domain lines=$lines"; exit 11; }
done

"$PY" -m py_compile \
  "$SRC/stage3_transactional.py" "$SRC/v2_input_correction.py" \
  "$SRC/stage8_sccf.py" "$SRC/stage9_fusion_sweep.py" \
  "$SRC/stage9_retrieval_sweep.py" "$SRC/stage9_paper_aggregate.py"
"$PY" "$SRC/stage3_transactional.py" --unit-test
"$PY" "$SRC/v2_policy_matrix.py" --unit-test
"$PY" "$SRC/stage8_sccf.py" --unit-test
"$PY" "$SRC/stage9_fusion_sweep.py" --unit-test
"$PY" "$SRC/stage9_retrieval_sweep.py" --unit-test

run_guarded() {
  local dst="$1"; shift
  if [[ -e "$dst" && -n "$(find "$dst" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "REFUSE_NONEMPTY_OUTPUT $dst"; return 41
  fi
  mkdir -p "$dst"
  "$@"
}

for method in source_raw official_raw source_adain official_adain v2_adain stage9_anatomy; do
  for seed in 1 2 3; do
    for order in CD DC; do
      dst="$OUT/$method/seed${seed}_${order}"
      ckpt="$BASE/outputs/source_train_seed${seed}/best.pth"
      echo "PAPER_RUN_START method=$method seed=$seed order=$order ckpt=$ckpt $(date -Is)"
      case "$method" in
        source_raw)
          run_guarded "$dst" "$PY" -u "$SRC/stage3_transactional.py" \
            --method source_only --ckpt "$ckpt" --shape-stats "$BASE/outputs/anatomy_stats_source_train.json" \
            --source-manifest "$BASE/manifests/source_train.csv" --output "$dst" \
            --order "$order" --seed "$seed" --gpu 0 --limit 400 --report-every 100 ;;
        official_raw)
          run_guarded "$dst" "$PY" -u "$SRC/stage3_transactional.py" \
            --method official_fixed --ckpt "$ckpt" --shape-stats "$BASE/outputs/anatomy_stats_source_train.json" \
            --source-manifest "$BASE/manifests/source_train.csv" --output "$dst" \
            --order "$order" --seed "$seed" --gpu 0 --limit 400 --report-every 100 ;;
        source_adain|official_adain|v2_adain)
          mode="${method%_adain}"
          [[ "$mode" == "source" ]] && mode=source_only
          [[ "$mode" == "official" ]] && mode=official_fixed
          [[ "$mode" == "v2" ]] && mode=anatomy
          run_guarded "$dst" "$PY" -u "$SRC/v2_input_correction.py" \
            --mode "$mode" --ckpt "$ckpt" --shape-stats "$BASE/outputs/anatomy_stats_source_train.json" \
            --source-manifest "$BASE/manifests/source_train.csv" --output "$dst" \
            --order "$order" --seed "$seed" --gpu 0 --limit 400 --report-every 100 --save-previews 4 ;;
        stage9_anatomy)
          run_guarded "$dst" "$PY" -u "$SRC/stage9_fusion_sweep.py" \
            --variant anatomy --ckpt "$ckpt" --shape-stats "$BASE/outputs/anatomy_stats_source_train.json" \
            --source-manifest "$BASE/manifests/source_train.csv" --calibration "$CFG" \
            --output "$dst" --order "$order" --seed "$seed" --gpu 0 --limit 400 --report-every 100 ;;
      esac
      echo "PAPER_RUN_DONE method=$method seed=$seed order=$order $(date -Is)"
    done
  done
done

echo "PAPER_RUNS_COMPLETE $(date -Is)"
run_guarded "$AGG" "$PY" "$SRC/stage9_paper_aggregate.py" --root "$OUT" --out "$AGG"
sha256sum "$CFG" "$SRC"/*.py "$SRC/stage9_paper_queue.sh" > "$ROOT/sha256_sources.txt"
echo "PAPER_QUEUE_COMPLETE $(date -Is) rc=0"
