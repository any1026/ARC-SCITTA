# Operation Log: Stage9 Paper Table Queue

Date: 2026-09-14

This log records the frozen configuration and the commands used to launch the server queue. All remote paths are below `/home/zhaoruijin/MOURUI`; existing data, checkpoints, logs, and results are preserved.

## Pre-launch audit

- Python compilation: `stage3_transactional.py`, `v2_input_correction.py`, `stage8_sccf.py`, `stage9_fusion_sweep.py`, `stage9_paper_aggregate.py`.
- Unit tests: Stage3 full snapshot, Stage8 SCCF, Stage9 fusion, and Stage9 retrieval tests where available.
- Checkpoint pairing: `source_train_seed1/2/3/best.pth` paired with the same seed number.
- Metric: capped 2D ASSD only; strict ASSD/HD95 are not claimed.

## Queue

- Work root: `/home/zhaoruijin/MOURUI/sictta_stage9_paper400_20260914`
- Screen: `sictta_stage9_paper400_20260914`
- Log: `/home/zhaoruijin/MOURUI/sictta_stage9_paper400_20260914/logs/paper400_queue.log`
- Output: `/home/zhaoruijin/MOURUI/sictta_stage9_paper400_20260914/outputs/paper400`
- Aggregate: `/home/zhaoruijin/MOURUI/sictta_stage9_paper400_20260914/outputs/paper400_aggregate`

The queue runs six method labels over seeds 1/2/3 and CD/DC, with 400 cases per domain. It refuses non-empty run directories and writes a completion marker only after the aggregator passes.

## Completion evidence to record

The next audit must report `PAPER_QUEUE_COMPLETE rc=0`, `STAGE9_PAPER_AGGREGATE_COMPLETE`, 72 domain-level runs, 400 rows per domain CSV, finite metrics, and the generated summary/table paths. A running screen or source preprocessing marker alone is not completion evidence.
