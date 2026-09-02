# Anatomy-SicTTA v1 reproduction protocol

## Provenance

- Official upstream SicTTA repository is preserved on the server at `/home/zhaoruijin/MOURUI/sictta_reproduction_20260827/repos/SicTTA`.
- The derived implementation is copied from the complete upstream `sotas/sictta.py`; it is not a shortened reimplementation.
- Server work root: `/home/zhaoruijin/MOURUI/sictta_reproduction_20260827`.
- Source checkpoints: `outputs/source_train_seed{1,2,3}/best.pth`.
- Phase-one result directory: `outputs/phase1_anatomy_v1_20260829`.
- Phase-one completion marker: `PHASE1_RESUME_COMPLETE 2026-08-29T10:27:22+08:00`.

## Data protocol

Fundus-doFE domains A/B/C/D are supplied by the official author package. A/B are split into 208 source-training and 52 source-validation cases using a stratified 8:2 split. C and D each contain 400 target-stream cases. Domain1/all.csv is excluded because it mixes source and target paths and can introduce leakage.

Adaptation is continuous within each stream. The adaptation function receives images only; target masks are used only after prediction for evaluation.

## v1 change

The anatomy gate is applied after official CCD/SFT admission. It checks source-derived ranges for cup/disc area ratio, normalized cup-disc center distance, and connected-component fractions. Official FIFO length 40, Top-K=5, SABE, SFF and recovery behavior remain unchanged.

## Reproduction command

The full server command is retained in `src/sictta_phase1_resume_20260829.sh`. It expects the server directory layout above and writes all outputs under `/home/zhaoruijin/MOURUI/sictta_reproduction_20260827`.

## Interpretation limits

The paper Table 2 reports 80.24% average Dice for SicTTA; that value is not substituted for this project's run. The supplied Fundus checkpoint was truncated, so source models were trained from scratch. ASSD implementation and pixel-scale conventions require a later harmonization pass.
