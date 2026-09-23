# KV CXL-PIM Decode Stage 11 analysis

## Objective

Stage 11 connects the three exact Stage-10 CXL-PIM Attention pipelines to the
48-layer MoE Decode event graph.  It answers whether the Stage-9/10 hardware
boundary remains visible after Router, Expert, Combine, and the measured
non-Attention Decode time are included.

This stage executes no new Ramulator workload.  It reuses six SHA-256-bound
exact cache artifacts: the Stage-6 full-count Attention and Expert workloads,
the Stage-7 all-Local-HBM counterfactual, and the three Stage-10 unified
CXL-PIM pipelines.  All rows have `interpolation=0`.

## Event graph and evidence

Each of the 48 sequential layers uses this dependency structure:

```text
norm -> qkv -> rope -> local GPU Attention -----+
                    \-> CXL-PIM pipeline -------+-> join -> o_proj
                                                    -> Router
                                                    -> Expert
                                                    -> Combine
```

The ideal-overlap bound lets the two Attention branches start after RoPE.  The
fully serialized bound makes the CXL-PIM branch wait for the local branch.
Ideal overlap is a scheduling bound, not an implemented runtime policy.

The evidence classes remain separate:

- the B8/C32k local Attention branch is extrapolated from independent A800
  B8/C4k, C8k, and C16k measurements;
- the Router input is the validated B8/C4k template tiled to B8/C32k, not a new
  B8/C32k A800 capture;
- CXL-PIM pipeline and Expert memory milestones are exact Ramulator results;
- Router, Expert compute, Combine, and other individual components remain
  modeled rather than separately measured on the A800.

The A800 measurements provide 126.394 ms of aggregate non-Attention time per
Decode step.  Explicit graph nodes account for 34.704891 us/layer before
calibration; a 2598.504868 us/layer residual makes their aggregate equal the
A800 anchor.  This residual is not attributed to Router, Expert, or Combine.

## End-to-end results

| scenario | scheduling | Decode ms | tok/s | Attention ms | Attention share |
|---|---|---:|---:|---:|---:|
| capacity-unlimited Local-HBM counterfactual | resident | 726.037 | 11.019 | 599.643 | 82.59% |
| 64 GB/s memory-only CXL | paired exact memory delta | 834.554 | 9.586 | 708.160 | 84.85% |
| CXL-PIM `c2_mac49152_p4` | ideal overlap | 573.311 | 13.954 | 446.917 | 77.95% |
| CXL-PIM `c2_mac49152_p4` | fully serialized | 1011.558 | 7.909 | 885.164 | 87.51% |
| CXL-PIM `c4_mac24576_p4` | ideal overlap | 573.311 | 13.954 | 446.917 | 77.95% |
| CXL-PIM `c4_mac24576_p4` | fully serialized | 683.924 | 11.697 | 557.530 | 81.52% |
| CXL-PIM `c8_mac24576_p4` | ideal overlap | 573.311 | 13.954 | 446.917 | 77.95% |
| CXL-PIM `c8_mac24576_p4` | fully serialized | 630.118 | 12.696 | 503.724 | 79.94% |

All ideal-overlap rows have the same time because the A800-extrapolated local
GPU branch remains longer than each CXL-PIM branch.  This equality is a bound;
it does not show that the three hardware designs are equivalent.

The fully serialized rows expose the hardware boundary.  Relative to
memory-only CXL, the 2-channel slow-MAC negative control reduces throughput by
17.498%, while the 4-channel nominal and 8-channel robust designs improve it by
22.024% and 32.444%.  The corresponding Decode latency reductions for the
4-channel and 8-channel designs are 18.049% and 24.496%.

The common modeled component values are 0.099 ms for the Router kernel,
0.147 ms for the complete routing-control path, 1.275 ms for the Expert critical
path, and 0.050 ms for Combine.  The Expert value reproduces the Stage-6
full-count result at 26.557051 us/layer.  These small modeled components do not
replace the A800 aggregate non-Attention measurement.

## Capacity interpretation

B8/C32k contains 25,769,803,776 bytes of KV cache and an estimated peak memory
footprint of 86,834,745,344 bytes.  The all-local row is therefore an infeasible
counterfactual in the A800 80 GB capacity domain.  The memory-only and CXL-PIM
rows are classified as `spill` with the separate modeled 64 GB CXL capacity.

The same 86.835 GB footprint is `resident` in the separate 96 GB simulated-local
capacity domain.  The 80 GB and 96 GB domains are never added or used as each
other's spill destination.

## Conclusion and next gate

The 48-layer event graph preserves the Stage-10 design conclusion.  Near-data
Attention can recover the modeled raw-KV link penalty only when the CXL-PIM
media throughput is high enough: the 4-channel nominal point passes even under
the serialized bound, the 8-channel point has more margin, and the 2-channel
slow-MAC point fails.

The next experiment should replace the two scheduling bounds with an explicit
chunked Attention schedule.  It should define query/KV tiles, partial-result
readiness, double-buffer limits, and GPU merge work, then simulate overlap as a
dependency graph rather than an assumed fraction.  This gate should stay on one
GPU and the current three designs.  Shared CXL-PIM and multi-GPU arbitration
remain outside the implemented scope.

This stage does not implement paging, eviction, recovery, a physical CXL
protocol, energy, Shared CXL-PIM, multi-GPU execution, or a deployable two-byte
partial format.  It is a mixed A800-extrapolation and Ramulator sensitivity
result, not a physical CXL-PIM measurement.
