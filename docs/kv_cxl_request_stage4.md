# KV CXL request-level Stage 4

Stage 4 adds the first request-level **CXL memory-only** model. The topology in one Ramulator
simulation is:

```text
Local HBM controllers: GPU Expert READ + PIM commands + resident KV READ
CXL controllers:       spilled KV READ only
```

The CXL controller is a FIFO service model with an explicit aggregate bandwidth, fixed access
latency, channel count, queue depth and transaction size. It uses the pinned HBM3 DRAM spec only
for transaction width and address levels; it does not claim to be a physical CXL protocol or a
measured device. No CXL-PIM command, paging, eviction, recovery, or multi-GPU behavior is present.

The frontend is `SieveCXLKVRead` and the Python entry point is
`src/sieve_replay/ramulator/cxl_kv_workload.py`. The build entry point is
`scripts/build_cxl_kv_ramulator.sh`; it preserves the frozen KV READ baseline under
`third_party/ramulator2` and installs the new frontend and controller into the separate ignored
`third_party/ramulator2_cxl` tree. The request-level result runner is
`scripts/run_kv_cxl_request_stage4.py`.

The executed matrix uses the A800 80 GB capacity domain with a hypothetical 64 GB CXL budget,
the four Stage-2 pressure points (`B8/C32k`, `B16/C16k`, `B16/C32k`, `B32/C16k`), six modes,
and three explicit sensitivity profiles:

| profile | aggregate bandwidth | access latency | CXL channels |
|---|---:|---:|---:|
| conservative-assumption | 32 GB/s | 0.5 us | 4 |
| nominal-assumption | 64 GB/s | 0.25 us | 4 |
| optimistic-assumption | 128 GB/s | 0.1 us | 4 |

All 72 rows are exact request-level runs with no interpolation. The checked-in result directory is
`results/kv_cxl_request_stage4_v1/`. Because this first pass is a microbenchmark, `request_scale=4096`
is recorded in every cache input and report: all KV and Expert/PIM request counts are reduced by the
same factor to keep the shape run short. This preserves the controlled request topology but is not a
full per-layer Decode traffic result. `request_scale=1` is reserved for selected formal long runs.
Each result metadata and stage manifest also records the pinned Ramulator revision plus the CXL
Python binding and shared-library SHA-256; the baseline library is kept separate and unchanged.

The observed direction is consistent across the controlled runs: CXL-only and combined modes have
nonzero CXL queue wait and link busy ratio; doubling the spilled KV request count from the two
`B*C=262144` cases to the two `B*C=524288` cases increases both CXL completion time and queue wait.
The faster profiles reduce completion and queue wait under the assumed service model. Local HBM
request counts remain unchanged when CXL KV traffic is added, which confirms that the two request
domains are independent in this frontend.

One selected full-count gate has also completed: A800 `B8/C32k`, nominal profile, CXL-KV-only,
`request_scale=1`. It injected and completed `4,449,704` CXL READs exactly, reached
`2,429.786736 us` at the CXL controller, and had a link busy ratio of `0.999898`. The summed
queue wait over all requests was `2,487,538.335360 us`; this is an aggregate queue-wait metric,
not a single request's critical-path latency. The row is stored independently in
`results/kv_cxl_request_stage4_formal_b8_c32k_nominal/` and is still a Ramulator model result.

Two additional full-count gates are stored independently. The B8/C32k nominal combined row
(`cxl-kv-plus-local-kv-plus-expert`) completed the same `4,449,704` CXL READs in
`2,429.786736 us` while also completing `12,327,512` Local KV READs and `1,179,648` GPU
Expert READs. The CXL link busy ratio remained `0.999898`, so the CXL stream determined the
single-layer completion milestone in this controlled combination. The B16/C32k nominal
CXL-only row completed `21,227,390` CXL READs in `11,590.404384 us` with link busy ratio
`0.999978`. Both rows have `exact_runs=1` and `interpolation=0`; they are in
`results/kv_cxl_request_stage4_formal_b8_c32k_combined_nominal/` and
`results/kv_cxl_request_stage4_formal_b16_c32k_cxl_nominal/`.

The matched B32/C16k CXL-only point completed `21,228,328` CXL READs in `11,590.915440 us`
with link busy ratio `0.999979`. Its slightly higher count than B16/C32k comes from the
batch-dependent activation bytes changing the admitted A800 spill bytes. It is stored in
`results/kv_cxl_request_stage4_formal_b32_c16k_cxl_nominal/` with the same exact-run and
no-interpolation checks.

These observations are **Ramulator request-level sensitivity results**, not A800 measurements.
They are enough to justify the next gate: run selected `request_scale=1` shapes, then connect the
validated Local-HBM/CXL request results to the 48-layer Decode event graph. Only after CXL queue
wait and link utilization affect that exact end-to-end graph should CXL-PIM computation be added.
