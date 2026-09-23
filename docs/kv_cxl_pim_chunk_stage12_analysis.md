# KV CXL-PIM chunk pipeline Stage 12 analysis

## Objective

Stage 12 replaces the Stage-11 ideal-overlap and fully-serialized bounds with
an explicit chunk schedule for B8/C32k.  It models partial-result readiness,
one- and two-slot buffering, and GPU merge work while preserving the calibrated
48-layer MoE Decode graph.

The experiment covers the three Stage-10 boundary designs:

- `c2_mac49152_p4`: 2-channel slow-MAC negative control;
- `c4_mac24576_p4`: 4-channel nominal design;
- `c8_mac24576_p4`: 8-channel robust design.

Each design runs three exact configurations: 1 chunk / 1 slot, 8 chunks / 1
slot, and 8 chunks / 2 slots.  The resulting 9/9 full-count Ramulator
workloads completed with failed=0, remaining=0, and interpolation=0.  The
execution-time Python binding, Ramulator library, source, configuration, and
input hashes are stored with every cache record.

## Request and scheduling model

Query link transfer and PIM GWRITE execute once per layer.  The spilled-KV MAC
waves are divided as evenly as possible among the chunks.  Every chunk then
executes PIM MAC, PIM READ, and result-link transfer.  A chunk returns a full
FP32 online-softmax partial state for every Batch, Attention-head, and PCH
combination.  Therefore 8 chunks issue eight times the Stage-10 PIM READ and
result-link traffic; the model does not treat chunking as free.

One buffer slot prevents the next PIM chunk from starting until the previous
result transfer completes.  Two slots permit the next chunk's PIM work to
overlap the previous result transfer.  The link and PIM media controllers are
still separate simulated resources, so this is not a unified physical CXL
controller model.

The Decode event graph launches the local GPU Attention branch and CXL-PIM
branch after RoPE.  Local GPU Attention is a non-preemptive `GPU_COMPUTE`
event.  Each exact partial-ready milestone releases one merge kernel on the
same resource; all merges must complete before output projection.

Merge time is an analytic roofline assumption.  Each partial state uses
`3 * head_dim + 10` FLOPs, and bytes include the partial input plus accumulator
read and write.  The configured simulated GPU peak, configured HBM bandwidth,
and a 1 us launch overhead are used.  This is not an A800 merge measurement.

## Exact pipeline results

| design | chunks/slots | pipeline us/layer | max buffers | buffer-stall cycles | PIM READ requests | result-link requests |
|---|---:|---:|---:|---:|---:|---:|
| c2_mac49152_p4 | 1/1 | 9130.146 | 1 | 0 | 16,640 | 16,640 |
| c2_mac49152_p4 | 8/1 | 9225.391 | 1 | 209,412 | 133,120 | 133,120 |
| c2_mac49152_p4 | 8/2 | 9159.965 | 2 | 0 | 133,120 | 133,120 |
| c4_mac24576_p4 | 1/1 | 2304.425 | 1 | 0 | 33,280 | 33,280 |
| c4_mac24576_p4 | 8/1 | 2463.267 | 1 | 413,252 | 266,240 | 266,240 |
| c4_mac24576_p4 | 8/2 | 2334.244 | 2 | 0 | 266,240 | 266,240 |
| c8_mac24576_p4 | 1/1 | 1183.473 | 1 | 0 | 66,560 | 66,560 |
| c8_mac24576_p4 | 8/1 | 1469.512 | 1 | 820,932 | 532,480 | 532,480 |
| c8_mac24576_p4 | 8/2 | 1213.292 | 2 | 0 | 532,480 | 532,480 |

All three 1-chunk results reproduce the corresponding Stage-10 query,
GWRITE, MAC, READ, result-link, and final completion milestones in 10/10 cycle
fields.  This proves that the new frontend preserves the previous pipeline when
chunking is disabled.

Relative to 8 chunks / 1 slot, two slots reduce the exact pipeline time by
0.709%, 5.238%, and 17.436% for the 2-, 4-, and 8-channel designs.  Buffer
stall cycles fall to zero.  The benefit grows with channel count because result
transfer occupies a larger share after the PIM MAC stage becomes shorter.

Relative to the one-chunk pipeline, 8 chunks add 1.043%, 6.893%, and 24.169%
with one slot.  Two slots limit those overheads to 0.327%, 1.294%, and 2.520%.
This is the cost of repeatedly materializing a complete partial state.

## Decode results

| design | chunks/slots | merge us/layer | Decode ms | tok/s | throughput vs memory-only CXL |
|---|---:|---:|---:|---:|---:|
| c2_mac49152_p4 | 1/1 | 1.200 | 573.369 | 13.953 | +45.553% |
| c2_mac49152_p4 | 8/1 | 9.597 | 573.772 | 13.943 | +45.451% |
| c2_mac49152_p4 | 8/2 | 9.597 | 573.772 | 13.943 | +45.451% |
| c4_mac24576_p4 | 1/1 | 1.399 | 573.378 | 13.952 | +45.550% |
| c4_mac24576_p4 | 8/1 | 11.195 | 573.849 | 13.941 | +45.431% |
| c4_mac24576_p4 | 8/2 | 11.195 | 573.849 | 13.941 | +45.431% |
| c8_mac24576_p4 | 1/1 | 1.799 | 573.398 | 13.952 | +45.545% |
| c8_mac24576_p4 | 8/1 | 14.390 | 574.002 | 13.937 | +45.392% |
| c8_mac24576_p4 | 8/2 | 14.390 | 574.002 | 13.937 | +45.392% |

All exact CXL-PIM timelines finish before the monolithic local GPU Attention
event.  Consequently one-slot and two-slot cases have identical Decode time
for a fixed design and chunk count: double buffering improves the hidden
CXL-PIM branch but does not shorten the current critical path.  The 8-chunk
Decode penalty relative to one chunk is the seven additional merge launches:
0.403, 0.470, and 0.604 ms per Decode for the three designs.

The explicit schedule lies just above the Stage-11 ideal-overlap bound by
0.058--0.691 ms, depending on design and chunk count.  It lies well below the
Stage-11 fully-serialized bound because the CXL-PIM branch starts with local
Attention.  In particular, the 2-channel negative control no longer fails as
it did under full serialization.  This result depends on concurrent branch
launch and cannot be generalized to a runtime that serializes the branches.

The common memory-only CXL reference is 834.554 ms/Decode.  Under the current
launch and merge assumptions all nine chunked CXL-PIM cases outperform it.
This is a mixed extrapolation/simulation result, not physical CXL-PIM evidence.

## Capacity and evidence boundaries

B8/C32k uses an estimated 25,769,803,776 bytes of KV cache and an 86,834,745,344
byte peak footprint.  It is `spill` in the separate A800 80 GB domain with the
modeled 64 GB CXL capacity, while it is `resident` in the separate 96 GB
simulated-local domain.  The two local-capacity domains are not added.

B8/C32k Router behavior remains the controlled B8/C4k template assigned a C32k
context.  Local Attention remains extrapolated from independent A800 B8/C4k,
C8k, and C16k measurements.  Exact Ramulator evidence applies only to the
stated query, PIM, and result-link request model.  GPU merge, Router, Expert
compute, Combine, and the non-Attention calibration residual remain modeled.

The experiment does not implement paging, eviction, recovery, physical CXL
protocol behavior, energy, Shared CXL-PIM, or multi-GPU arbitration.

## Next gate

The next experiment should test the scheduling assumption that now controls
the conclusion.  It should split local GPU Attention into explicit tiles,
scan CXL-PIM launch skew, and compare immediate merge, batched merge, and
priority choices on `GPU_COMPUTE`.  A small chunk-count sweep should also test
whether 2 or 4 chunks retain most buffering benefit without multiplying full
partial-state traffic eightfold.  Shared CXL-PIM and multi-GPU should remain
out of scope until this single-GPU schedule is stable.
