# Stage9 Anatomy Fusion Freeze and Audit

Date: 2026-09-14

## Frozen research object

The confirmation run freezes the Stage9 anatomy-aware continuous fusion configuration in `configs/stage9_anatomy_frozen.json`. The target protocol is Fundus-doFE, with A/B as source, C/D as target, 320x320 images, 400 images per target domain, continuous C->D and D->C streams, and seeds 1/2/3.

The candidate path is per-image RGB AdaIN to source-train moments followed by the audited Anatomy-SicTTA candidate. CCD history and FIFO capacity are both 40 and prototype retrieval is top-k=5. A frozen source checkpoint prediction is retained as an anchor. The final probability is:

```text
p_final = w * p_candidate + (1 - w) * p_source
w = sigmoid(4 * candidate_anatomy_score)
```

Candidate admission, fusion, and memory updates do not read target labels. Labels are read only by the final evaluator.

## Audit gates

Before launching the queue, the server script must compile every Python file, run unit tests, verify the three seed-specific source checkpoints, the source manifest, and source-derived anatomy statistics. It must refuse to overwrite any non-empty output directory.

The post-run aggregator requires 6 methods x 3 seeds x 2 orders x 2 domains = 72 domain-level summaries. Every domain must report `n=400`, finite OD/OC Dice and capped 2D ASSD values, and exactly 400 unique rows in its per-case CSV. Summary values are recomputed from the per-case rows and must agree. It emits a domain-level long table, mean/sample-SD table, paired adaptation gain, negative-transfer rate, worst-quartile and tail metrics, and SHA256 records.

## Method mapping

| Table label | Evaluator | Correction/adaptation |
|---|---|---|
| source_raw | `stage3_transactional.py --method source_only` | frozen source model |
| official_raw | corrected CCD/FIFO adapter with anatomy disabled | official SicTTA control |
| source_adain | `v2_input_correction.py --mode source_only` | source model + RGB AdaIN |
| official_adain | `v2_input_correction.py --mode official_fixed` | official CCD/FIFO + RGB AdaIN |
| v2_adain | `v2_input_correction.py --mode anatomy` | Anatomy-SicTTA + RGB AdaIN |
| stage9_anatomy | `stage9_fusion_sweep.py --variant anatomy` | frozen candidate/source fusion |

## Metric disclosure and limitations

The table calls the geometric metric **capped 2D ASSD**. The existing implementation truncates pixel distances at 10 and does not provide strict physical-unit ASSD or HD95. No paper Table 2 value is substituted for these project reruns. The anatomy variant was selected during a previous 20-image smoke sweep, so the 400-image run is a frozen confirmation, not an independent test of model selection; an independent selection/test split remains required before submission.

## Prior implementation issue

The earlier 100-image Stage9 expansion reused seed-1 checkpoint paths for all seeds. Those exploratory outputs are excluded from this queue. This queue resolves the issue by pairing seed `s` with `outputs/source_train_seed{s}/best.pth` and recording the pairing in the operation log.

The inherited adapter also declared `num_classes=4` while the Fundus model has three output classes. The prototype pool does not use this field in retrieval or updates, so prior predictions were unaffected; it is corrected to 3 here to keep metadata consistent with the checkpoint.
