# Full-decode stage design

## Execution order

The event graph is strictly ordered as:

```text
step 0 / layer 0 -> ... -> step 0 / layer 47
-> step 1 / layer 0 -> ...
```

Each layer reuses the established Attention, Router, dispatch, GPU/PIM expert
and Combine DAG. Event names are scoped as `stepN.layerM.event`, while hardware
resource names remain global. The next layer's `norm1` depends on the previous
layer's `residual2`, so no layer or decode-step overlap is assumed.

## Trace invariants

- Every selected step/layer pair must exist.
- Request identity and ordering are identical across layers in a step.
- Context length is identical for a request across layers in a step.
- Context length advances by one for each consecutive decode step.
- Real traces require a manifest whose SHA-256 and model shape match the run.

These constraints intentionally model fixed-batch synchronous decode. Continuous
batching and early request completion are out of scope.

## Timing levels

`analytic-v0` validates the complete graph quickly. `ramulator-table-v0` uses a
multi-entry cycle-v0 table for every exact `(batch, context)` and PIM token
count. `ramulator-contention-v1` additionally requires a schema-v3 table
materialized from the exact workload cache.

For the provided 48-layer, 2-step synthetic trace, 14,208 placement consumers
deduplicate to 146 Ramulator shapes. The planner includes every hot-prefix
mixed run plus GPU-only and PIM-only shapes. This is larger than the original
single-layer monotonic bisection, but makes every per-layer policy choice
available without interpolation.

## Result contract

Multi-layer runs write:

- `summary.json`: total latency, throughput, step statistics, category totals,
  critical path, placement totals and peak memory;
- `steps.csv`: start/end/latency for each decode step;
- `layers.csv`: timing and placement counts for each layer execution;
- `events.csv`: namespaced event schedule;
- `placement.csv`: exact expert target for each step/layer;
- `contention.csv`: per-layer cycle-v1 isolation, contention and row-buffer metrics;
- `run_manifest.json`: full input snapshots and hashes.

For multi-policy cycle-v1 runs, the result root additionally writes an
aggregated `contention.csv` with per-policy totals and per-layer means.

Synthetic and analytic results are functional validation, not paper-level
performance claims. A formal real-trace cycle-v1 result additionally requires
captured Qwen3 routing data, a complete exact workload cache and GPU timing
calibration.
