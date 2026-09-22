# KV CXL Decode Stage 5

Stage 5 is the first controlled connection between the Stage-4 request-level CXL model and the
full Decode event graph.  It uses four pressure points:

| case | Batch | Context |
|---|---:|---:|
| B8/C32k | 8 | 32768 |
| B16/C16k | 16 | 16384 |
| B16/C32k | 16 | 32768 |
| B32/C16k | 32 | 16384 |

The policies are `gpu-only` and analytic `sieve`.  The router input is the manifest-validated
real B8/C4k step-0/layer-0 batch, tiled by an integer batch factor and assigned the requested
context length.  This is a controlled routing template; it is not a new A800 capture and it does
not claim that routing is unchanged for a real longer prompt.

For every case/policy, the runner derives one exact request shape.  Resident KV READ, spilled CXL
KV READ, GPU Expert READ and PIM waves are injected together into the isolated Stage-4 CXL
Ramulator build.  The frontend completion milestones become event durations for all 48 layers.
The graph uses `overlap-local-hbm-v1`: Attention compute starts after RoPE and `o_proj` waits for
Attention, Local KV and CXL KV.  Layer-to-layer dependencies remain the existing Decode graph.

The initial run uses `request_scale=4096`.  This divides all request counts with a ceiling so the
shape remains exact and finishes quickly; GPU arithmetic and fixed model stages remain analytic.
The PIM frontend currently exposes one final PIM completion milestone, so the adapter keeps the
intermediate PIM write/compute events at zero duration and records the final milestone at the PIM
read event.  This avoids inventing stage-level interpolation.

The nominal CXL profile is an explicit model assumption:

| aggregate bandwidth | access latency | CXL channels | extra capacity |
|---:|---:|---:|---:|
| 64 GB/s | 0.25 us | 4 | 64 GB |

The active capacity domain is physical A800 80 GB plus hypothetical CXL 64 GB.  The 96 GB Local
HBM simulation capacity is reported separately as a comparison state; the two capacities are
never pooled.  The controller remains a FIFO memory-only service model.  There is no physical CXL
protocol, paging, eviction, recovery, CXL-PIM compute, multi-GPU behavior or measured A800 timing.

Outputs are in `results/kv_cxl_decode_stage5_v1/`:

* `comparison.json` and `comparison.csv` contain one row per case/policy;
* `layer_results.csv` contains 48 rows per case/policy;
* `exact_cache/` stores the request-level result and its input provenance;
* `stage_manifest.json` records input/output hashes, binary provenance, exact-run count and zero interpolation;
* `stage5_report.md` summarizes total latency, attention share, Expert critical path, CXL critical path and aggregate CXL queue wait.

`cxl_queue_wait_us_all_requests` is the sum of queue waits for every CXL request across the
replayed layers.  It is not a single request's critical-path latency.  The end-to-end result is
classified as Ramulator request-level CXL memory-only timing connected to an analytic Decode event
graph, not as an A800 performance result.

The checked-in Stage-5 matrix does not include a `request_scale=1` combined graph replay.  An
attempted B8/C32k full-count run was stopped before its first cache entry because the current
frontend's per-cycle injection loop makes the roughly 19 million-request combined shape
prohibitively slow in this environment.  The next gate is to optimize or otherwise bound that
frontend cost, then run one full-count shape with the same provenance checks.  Only if CXL
completion milestones change the full Decode critical path should the project add a request-level
CXL-PIM computation model.
