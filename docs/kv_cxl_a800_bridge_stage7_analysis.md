# KV CXL A800 bridge Stage 7 analysis

## Objective

Stage 7 connects the independent A800 Attention/Decode measurements to the
phase-faithful request-level CXL result without relabeling either source. It asks a
single question: for the B8/C32k capacity-pressure point, what latency delta does
the current nominal memory-only CXL model add relative to a capacity-unconstrained
Local-HBM counterfactual?

This is a sensitivity bridge. B8/C32k was not executed on the A800 and no physical
CXL device was measured.

## Inputs and method

The A800 anchors are B8/C4k, B8/C8k and B8/C16k from
`results/a800_timing_memory_v1/summary.json`. A linear fit estimates B8/C32k
Attention time. As a forward check, a fit using only C4k and C8k predicts the held
out C16k point.

The request-level comparison contains two exact full-count Ramulator streams with
the same 16,777,216 KV READ transactions per layer:

- resident counterfactual: all 16,777,216 requests use Local HBM;
- nominal spill: 12,327,512 requests use Local HBM and 4,449,704 use the CXL
  controller, matching the Stage 6 A800 80 GB capacity split.

The resident stream is the one new exact run in this stage. The spill stream is
reused from Stage 6 by its recorded cache key and SHA-256. The A800 trend supplies
the baseline Attention/Decode estimate; only the difference between the paired
Ramulator memory completion milestones is added to it.

## Calibration quality

| check | value |
|---|---:|
| C4k+C8k -> C16k Attention absolute percentage error | 1.668% |
| three-anchor fit R² | 0.999920 |
| extrapolated B8/C32k Attention | 599.643 ms |
| mean measured B8 non-Attention Decode component | 126.394 ms |
| range of measured B8 non-Attention component | 1.064 ms |

The forward holdout supports this limited B8 Context trend. It does not establish
generalization to another model, prompt distribution, Batch, hardware, or a longer
Decode run.

## Paired result

| scenario | capacity state | estimated Attention | estimated Decode | throughput | memory completion/layer |
|---|---|---:|---:|---:|---:|
| all Local HBM counterfactual | infeasible on A800 | 599.643 ms | 726.037 ms | 11.019 token/s | 169.016 us |
| nominal 64 GB/s memory-only CXL spill | spill | 708.160 ms | 834.554 ms | 9.586 token/s | 2429.787 us |

The nominal CXL row adds 108.517 ms per Decode step and reduces estimated
throughput by 13.003% relative to the capacity-unconstrained resident
counterfactual. Its 142,390,528-byte spilled payload per layer achieves
58.602 GB/s effective payload service in the current controller model.

Serving that payload by the exact all-Local-HBM completion deadline gives a derived
842.466 GB/s payload-rate threshold. This number excludes protocol overhead and
Attention computation and is not a physical CXL bandwidth measurement. It is useful
only as a design signal: moving raw spilled KV bytes over the nominal memory-only
link solves capacity admission but does not preserve the Local-HBM memory deadline.

## Evidence boundary

The result has three explicitly separate evidence classes:

1. B8/C4k--C16k Attention and Decode values are real A800 measurements.
2. B8/C32k Attention and Decode values are linear extrapolations from those A800
   measurements.
3. Local-HBM and CXL completion milestones are exact Ramulator results for the
   current abstract request streams.

The experiment does not implement paging transfers, eviction, recovery, a physical
CXL protocol, CXL-PIM compute, multi-GPU access, or a real B8/C32k A800 run. The
formal A800 capacity domain remains 80 GB and the simulated Local-HBM capacity
domain remains 96 GB; they are not combined.

All artifacts are stored in `results/kv_cxl_a800_bridge_stage7_v1/`. The validator
reports `passed`, with one new exact run and zero interpolation. It binds the A800
summary, Stage 6 comparison and Attention cache, capacity table, cycle configuration,
Ramulator revision, Python binding, shared library, exact cache, and generated
outputs by SHA-256.

## Next gate

The next experiment should define a CXL-PIM Attention request model before adding
performance numbers. The model must state how queries reach CXL-PIM, which KV rows
are read locally at CXL-PIM, what partial softmax/value state returns to the GPU,
and how multiple partial results are combined. It should then compare memory-only
CXL and CXL-PIM using the same B8/C32k request, capacity split, A800 calibration,
and exact-run provenance. This comparison will test whether near-data Attention can
reduce link traffic enough to recover the memory-only CXL penalty.
