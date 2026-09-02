# SCITTA-pro

Medical image continual test-time adaptation based on the official SicTTA implementation.

## Current reproducible result

This repository records the first reproducible research version: **Anatomy-SicTTA v1**. It keeps the official SicTTA CCD/SFT admission logic, 40-sample FIFO, Top-K=5, SABE, SFF and recovery semantics, and adds a label-free anatomy plausibility gate before a target sample enters the memory pool.

The gate is estimated from source-training masks using cup/disc area ratio, normalized center distance, and largest-connected-component fractions. Target labels are used only by the evaluator after adaptation; they are never passed to the adaptation function.

### Phase-one results

| Protocol | Seed 1 | Seed 2 | Seed 3 | Mean |
|---|---:|---:|---:|---:|
| Anatomy-SicTTA v1, C→D | 82.50% | 81.02% | 74.84% | **79.45%** |
| Anatomy-SicTTA v1, D→C | 81.68% | 74.87% | 70.76% | **75.77%** |
| Official SicTTA, D→C control | 80.66% | 72.03% | 62.71% | **71.80%** |

These are project re-runs, not copied paper numbers. The current result is preliminary: seed and stream-order variance remain substantial, and strict component ablations are intentionally postponed until the final method version.

## Reproduction scope

- Dataset: author-provided Fundus-doFE package (A/B source; C/D target streams).
- Source split: 208 train / 52 validation, stratified across A/B, no overlap.
- Target protocol: continuous C→D and D→C streams, 400 cases per domain.
- Source model: 2D U-Net, 320×320, 200 epochs, seeds 1–3.
- Main metrics: OD/OC Dice, ASSD and Macro Dice.

## Layout

- `src/`: complete derived implementation and evaluation utilities.
- `results/`: small, human-readable summary JSON files and checksums; raw images, datasets and checkpoints are not committed.
- `docs/`: protocol, operation logs and result interpretation.
- `configs/`: reproducibility metadata.

## Data and checkpoints

The Fundus-doFE data and source checkpoints remain on the project server and are deliberately excluded from this Git repository. See `docs/reproduction_protocol.md` for the server paths and the exact provenance information.

## Research status

The next implementation target is a safer SicTTA extension with explicit drift detection, source/adapted candidate comparison, case-level rollback and calibrated risk diagnostics. Ablations will be run after the final candidate is frozen.
