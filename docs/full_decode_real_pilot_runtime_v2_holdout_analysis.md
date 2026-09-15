# Qwen3 real-pilot runtime-v2 signature holdout

## Experiment question

The first runtime-v1 result used all 340 observed load signatures to construct
the calibration curves. This experiment asks a stricter question: if complete
load signatures are held out, can the runtime policy still choose the same
hot-prefix as the exact `sieve-cycle-v1` Oracle?

The split is signature-level rather than row-level. All layer-batches with the
same sorted active-expert token-count signature stay in the same fold. Five
folds therefore evaluate all 384 layer-batches while preventing a repeated
signature from appearing in both calibration and evaluation data. The exact
cycle-v1 contention table is used only after the runtime decision to calculate
ground-truth objective and regret; it is not queried by the runtime policy.

## Setup

- Model: Qwen3-30B-A3B, BF16, 48 layers, 128 experts, Top-8;
- Hardware model: one GPU and eight Local HBM-PIM stacks;
- Trace: the existing real pilot, Batch 8 and eight Decode steps;
- Coverage: 340 unique signatures and 384 layer-batches;
- Folds: 5 signature-level folds, 272 training signatures and 68 holdout
  signatures per fold;
- Calibration budgets: 5, 9, 15 and 25 non-zero points, split between GPU READ
  and PIM pipeline curves while preserving nested maximin anchor selection;
- New Ramulator runs: 0; exact timing table entries are reused as evaluation
  ground truth.

## Results

| Non-zero points | Prefix matches | Match rate | Mean regret (us) | P95 regret (us) | Max regret (us) |
|---:|---:|---:|---:|---:|---:|
| 5 | 376/384 | 97.9167% | 0.00085186 | 0 | 0.09737862 |
| 9 | 377/384 | 98.1771% | 0.00066941 | 0 | 0.09737862 |
| 15 | 377/384 | 98.1771% | 0.00066941 | 0 | 0.09737862 |
| 25 | 380/384 | 98.9583% | 0.00005092 | 0 | 0.00488832 |

The 25-point nested calibration is the best tested budget. Its four mismatches
are all one expert-prefix step away from the Oracle, and its maximum regret is
the same as the original in-sample runtime-v1 evaluation. The 5/9/15-point
budgets are still close but produce larger isolated mistakes in one or more
folds. More points are not automatically better if the anchor placement is
changed; this is why the budget sweep uses a nested selection rule.

One GPU count is outside the training range in each relevant fold, so the
candidate search can encounter an extrapolated candidate. No selected runtime
placement uses extrapolation. The PIM token-count ranges remain covered.

## Online decision-cost measurement

The runtime replay currently models a `20 us` scheduler event per layer-batch.
The Python implementation was benchmarked separately on all 384 layer-batches,
with ten warmup rounds and 100 measured rounds (38,400 decisions). The measured
decision CPU time was:

| Statistic | Time (us) |
|---|---:|
| Median | 370.164 |
| P95 | 471.678 |
| P99 | 503.041 |
| Maximum | 554.313 |

This is approximately 18.5x the configured 20 us model at the median. If this
unoptimized Python measurement were substituted directly, the scheduler portion
of 384 layer-batches would be about 142.14 ms instead of 7.68 ms. That arithmetic
is a sensitivity bound, not a revised performance result: it includes interpreter
and allocation overhead and is not a measurement of a production C++ scheduler.
The baseline latency numbers above intentionally retain the configured 20 us
parameter so that policy comparisons remain comparable. A deployment-oriented
runtime-v2 must implement and benchmark the scheduler in the target runtime, then
rerun the end-to-end timing with that measured cost.

Prediction errors across all holdout observations are recorded in
`results/full_decode_real_pilot_runtime_v2_holdout/prediction_errors.csv`. The
per-fold metrics are in `fold_metrics.csv`, and each fold/budget calibration is
saved under `calibrations/` with its own SHA-256 digest.

## Interpretation

This result is stronger than a random layer split because repeated load shapes
are kept together, but it is still an intra-pilot test. The 340 signatures came
from the same eight prompts, Batch, context regime and model. Therefore the
result supports only this claim:

> A sparse runtime model can retain near-Oracle hot-prefix decisions for
> signatures held out from calibration within this pilot.

It does not establish cross-prompt, cross-Batch, cross-context, cross-model or
cross-hardware generalization. The benchmark above measures this repository's
Python decision path but not a production scheduler, so the configured scheduler
overhead remains a model parameter. New prompts and
independent hardware microbenchmarks are required before making a deployment
claim.

## Files

- `results/full_decode_real_pilot_runtime_v2_holdout/summary.json`;
- `results/full_decode_real_pilot_runtime_v2_holdout/fold_metrics.csv`;
- `results/full_decode_real_pilot_runtime_v2_holdout/layers.csv`;
- `results/full_decode_real_pilot_runtime_v2_holdout/prediction_errors.csv`;
- `outputs/qwen3_real_pilot_runtime_v2_holdout_figures/runtime_v2_holdout_budget_comparison.png`.

The figure summarizes prefix mismatches and maximum placement regret. It is a
same-pilot holdout diagnostic, not a claim of broad workload generalization.
