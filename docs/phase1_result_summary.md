# Phase 1 result summary

| Method / stream | Seed 1 | Seed 2 | Seed 3 | Mean | Population SD |
|---|---:|---:|---:|---:|---:|
| Anatomy-SicTTA v1 C→D | 82.50% | 81.02% | 74.84% | 79.45% | 3.32 pp |
| Anatomy-SicTTA v1 D→C | 81.68% | 74.87% | 70.76% | 75.77% | 4.50 pp |
| Official SicTTA D→C | 80.66% | 72.03% | 62.71% | 71.80% | 7.33 pp |

The three-seed D→C improvement is approximately +3.97 percentage points. This is an initial mechanism result, not a final claim: acceptance is conservative and varies strongly by seed/order.

Representative admission diagnostics:

| Experiment | CCD pass | Anatomy pass | Pool accept |
|---|---:|---:|---:|
| Seed 1 C→D, C domain | 33 | 33 | 33 |
| Seed 1 C→D, D domain | 67 | 58 | 58 |
| Seed 2 D→C, D domain | 105 | 5 | 5 |
| Seed 3 D→C, D domain | 66 | 10 | 10 |

## Decision

Keep CoTTA as a strong comparator and official SicTTA as the main baseline. Freeze v1 as a reproducible checkpoint, then develop drift detection, candidate comparison and rollback. Delay ablations until the final candidate is frozen, as requested.
