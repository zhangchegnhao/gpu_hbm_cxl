# KV CXL-PIM sensitivity Stage 9

Stage 9 scans 18 hardware/data-format designs and nine scheduling/merge assumptions per design. Exact Ramulator components are deduplicated by simulation inputs; Stage-8 baseline components are reused by SHA-256.

Design points: `18`; scheduling rows: `162`; exact components: `23`; Stage-8 reused: `5`; new exact runs: `18`; interpolation: `0`.

Best row: `c2_mac12288_p2`, overlap=1.0, merge=0.0 us/layer, Decode=573.311 ms.
Worst row: `c2_mac49152_p4`, overlap=0.0, merge=20.0 us/layer, Decode=1012.518 ms.
Designs that beat memory-only CXL even at their worst scheduling/merge row: `16/18`.

These are model-sensitivity results, not physical CXL-PIM measurements. The scan still uses isolated phase runs, an A800-fit B8/C32k local branch, a fixed 64 GB/s FIFO link, and zero paging/protocol/energy modeling.
