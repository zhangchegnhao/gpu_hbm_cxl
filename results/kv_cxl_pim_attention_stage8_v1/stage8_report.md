# KV CXL-PIM Attention Stage 8

Stage 8 is an optimistic, decomposed CXL-PIM Attention model. Query transfer, PIM GWRITE/MAC/READ, and partial-result transfer are five separate exact Ramulator runs connected in dependency order. The local GPU Attention branch and CXL-PIM branch overlap ideally.

| scenario | estimated Decode ms | throughput tok/s | change vs memory-only CXL |
|---|---:|---:|---:|
| resident-local-counterfactual | 726.037 | 11.019 | 14.946% |
| nominal-memory-only-cxl-spill | 834.554 | 9.586 | 0.000% |
| optimistic-cxl-pim-attention | 573.311 | 13.954 | 45.567% |
| serialized-cxl-pim-attention | 683.924 | 11.697 | 22.024% |

CXL-PIM transfers 1,130,496 bytes per layer versus 142,390,528 raw spilled-KV bytes, a 99.206% reduction.
Its exact decomposed branch takes 2304.426 us per layer; the A800-fit local GPU branch remains critical in the ideal-overlap model.
Ideal overlap and serialized branches bound estimated Decode at 573.311--683.924 ms under the remaining model assumptions.

This is not a physical CXL-PIM result. The four-channel topology, PIM throughput, FP32 partial-state format, ideal branch overlap, and zero merge cost are assumptions. Softmax scalar operations, protocol overhead, paging, eviction, energy, multi-GPU sharing, and contention between the five isolated phases are absent.
