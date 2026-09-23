# KV CXL-PIM Decode Stage 11

Stage 11 replays three exact Stage-10 CXL-PIM pipelines in a calibrated 48-layer Decode event graph. It performs no new Ramulator runs.

| scenario | design | schedule | Attention ms | Expert ms | total ms | tok/s | Attention share |
|---|---|---|---:|---:|---:|---:|---:|
| resident-local-counterfactual | none | resident | 599.643 | 1.275 | 726.037 | 11.019 | 0.825912 |
| memory-only-cxl | memory-only-64GBps | paired-memory-delta | 708.160 | 1.275 | 834.554 | 9.586 | 0.848549 |
| cxl-pim-attention | c2_mac49152_p4 | ideal-overlap-bound | 446.917 | 1.275 | 573.311 | 13.954 | 0.779537 |
| cxl-pim-attention | c2_mac49152_p4 | fully-serialized-bound | 885.164 | 1.275 | 1011.558 | 7.909 | 0.875050 |
| cxl-pim-attention | c4_mac24576_p4 | ideal-overlap-bound | 446.917 | 1.275 | 573.311 | 13.954 | 0.779537 |
| cxl-pim-attention | c4_mac24576_p4 | fully-serialized-bound | 557.530 | 1.275 | 683.924 | 11.697 | 0.815193 |
| cxl-pim-attention | c8_mac24576_p4 | ideal-overlap-bound | 446.917 | 1.275 | 573.311 | 13.954 | 0.779537 |
| cxl-pim-attention | c8_mac24576_p4 | fully-serialized-bound | 503.724 | 1.275 | 630.118 | 12.696 | 0.799412 |

The memory-only CXL comparison is 834.554 ms per Decode step. Ideal overlap is a scheduling bound, not an implemented runtime policy.
The event graph uses `2598.504868` us/layer of calibration residual so that the aggregate non-Attention path matches the independent A800 mean. Router, Expert and Combine component values remain model-derived and must not be read as separate A800 measurements.

B8/C32k routing is controlled/tiled from the validated B8/C4k Router template. B8/C32k Attention is extrapolated from A800 B8/C4k-C16k measurements. CXL-PIM and memory milestones are Ramulator results for the stated assumptions.

This experiment does not implement paging, eviction, physical CXL protocol, Shared CXL-PIM, multi-GPU arbitration, energy, or a deployable two-byte partial format.
