# KV CXL phase-faithful Stage 6

Stage 6 evaluates a phase-faithful request-level CXL memory assumption inside the
current 48-layer Decode event graph. The graph remains

```text
Attention (Local KV + CXL KV) -> Router -> Expert (GPU/PIM) -> Combine
```

The Attention and Expert phases run as separate exact Ramulator workloads. Their
completion milestones are then connected to the analytic layer graph. This preserves
the existing dependency order; it does not claim that KV and Expert traffic overlap
in one Decode layer.

## Formal matrix

The input is the manifest-validated B8/C4k Router template tiled to B8/C32k. The
experiment compares `gpu-only` and `sieve` placements and uses two request scales:

| request scale | meaning | exact phase shapes |
|---:|---|---:|
| 4096 | tractable request-count scaling | 2 |
| 1 | full-count request stream | 2 |

Identical Attention and Expert shapes are shared by both policies. The formal output
contains four exact Ramulator runs, four policy/scale rows, and 192 per-layer rows.
There is no interpolation. The CXL profile is the existing nominal assumption: 64
GB/s, 0.25 us access latency, four channels, and a hypothetical 64 GB capacity.

The full artifacts are in
`results/kv_cxl_phase_stage6_b8_c32k_formal_v1/`. Its execution binding records
Ramulator revision `b30320bc9385b708e86b67ebb9f48858cc66d798`; the Stage 6 validator
reported `passed`, `exact_runs=4`, and `interpolation=0`.

## Results

| scale | policy | total latency (us) | throughput (request tok/s) | Attention share | CXL critical path (us) | CXL per-layer completion (us) | CXL link busy |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4096 | `gpu-only` | 971.384614 | 8235.667 | 0.050224 | 0.000000 | 0.842400 | 0.704537 |
| 4096 | `sieve` | 24469.576486 | 326.937 | 0.923064 | 0.000000 | 0.842400 | 0.704537 |
| 1 | `gpu-only` | 118668.253222 | 67.415 | 0.982822 | 116629.763328 | 2429.786736 | 0.999898 |
| 1 | `sieve` | 119628.253222 | 66.874 | 0.974935 | 116629.763328 | 2429.786736 | 0.999898 |

At the full count, the CXL Attention milestone dominates the critical path under the
nominal profile. The scale-4096 run shows the same request model at lower load and is
useful for tractable comparisons; it should not be read as a full-count latency.
The two policies share the same Attention shape in this controlled B8/C32k setup, so
their CXL phase is identical; their total difference comes from the analytic Expert
placement model.

## Evidence boundary

These are **Ramulator request-level memory results plus an analytic Decode event graph**.
They are not A800 timing measurements and do not establish the performance of a
physical CXL device. Routing comes from the existing A800-derived B8/C4k trace template
and is tiled to C32k; no new C32k Router capture was made here. GPU arithmetic,
Attention compute, Router, and scheduler components remain analytic.

The experiment does not implement paging transfers, eviction, recovery, a physical
CXL protocol, CXL-PIM compute, multi-GPU execution, or Shared CXL-PIM. The capacity
classification remains separate from physical A800 capacity: this case is marked
`spill` under the A800 80 GB domain and `resident` under the simulated 96 GB domain,
with no spill destination or transfer modeled. The CXL link and queue numbers are
conditional on the fixed nominal profile and request injection model.

The next independent check is A800 Attention/Decode timing and memory sampling for
the long-context cases. Only after those measurements should the request-level model
be calibrated or extended to paging and validated KV/Expert competition.
