# KV CXL phase-faithful Stage 6

Stage 6 separates Attention KV traffic from the later Expert phase to match the current Decode dependencies.
Identical shapes are executed once and shared by `gpu-only` and `sieve` consumers.
CXL remains a hypothetical memory-only FIFO profile; these are Ramulator and analytic event-graph results, not A800 measurements.

| scale | policy | total us | throughput tok/s | attention share | CXL critical us | CXL per-layer us | link busy |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4096 | gpu-only | 971.384614 | 8235.667 | 0.050224 | 0.000000 | 0.842400 | 0.704537 |
| 4096 | sieve | 24469.576486 | 326.937 | 0.923064 | 0.000000 | 0.842400 | 0.704537 |

The full-count result is an exact request-level run for the generated B8/C32k shape with zero interpolation.
It does not implement paging transfers, eviction, recovery, a physical CXL protocol, or CXL-PIM computation.
Aggregate queue wait is reported separately and is never treated as one request's critical-path latency.

Unique exact Ramulator runs: `2`.
