# KV CXL A800 bridge Stage 7

Stage 7 combines evidence only through an explicit delta model: the B8/C32k A800 baseline is a measured-trend extrapolation, while the Local-HBM/CXL memory delta comes from paired exact Ramulator request streams.

| scenario | capacity state | estimated Attention ms | estimated Decode ms | throughput tok/s | memory completion us/layer |
|---|---|---:|---:|---:|---:|
| resident-local-counterfactual | infeasible-on-a800 | 599.643 | 726.037 | 11.019 | 169.016 |
| nominal-memory-only-cxl-spill | spill | 708.160 | 834.554 | 9.586 | 2429.787 |

The forward holdout trained on C4k/C8k predicts C16k Attention with 1.668% absolute error. The all-anchor linear fit has R^2=0.999920.
Under the nominal 64 GB/s memory-only CXL assumption, spill adds 108.517 ms to the extrapolated Decode estimate and changes estimated throughput by -13.003%.
Serving the spilled payload by the exact all-local completion deadline would require 842.466 GB/s before protocol and compute costs; this is a derived payload-rate threshold, not a physical CXL measurement.

The resident B8/C32k row is capacity-infeasible on the A800 domain and was not run on A800. The CXL row is an A800-anchored extrapolation plus a Ramulator delta, not a physical CXL result. No paging, eviction, transfer state machine, CXL protocol, CXL-PIM compute, or multi-GPU execution is modeled.
