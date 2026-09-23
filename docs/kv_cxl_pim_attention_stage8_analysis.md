# KV CXL-PIM Attention Stage 8 analysis

## Objective

Stage 8 tests whether near-data Attention can remove the raw-KV link traffic that
made the Stage 7 memory-only CXL path slower. It uses the same B8/C32k capacity
split and A800 calibration as Stage 7. The result is an isolated, optimistic
CXL-PIM model, not a physical CXL-PIM measurement or a Shared CXL-PIM
implementation.

## Dataflow and assumptions

The modeled Attention path is:

```text
GPU query transfer
-> CXL-PIM query GWRITE
-> CXL-PIM QK and AV MAC
-> CXL-PIM partial-state READ
-> GPU partial-result transfer
-> GPU merge
```

The first model assumes four CXL-PIM channels, two pseudo-channels per channel,
and 24 banks per pseudo-channel. It reuses the current HBM3 timing preset,
24.576 ns PIM MAC interval and 1.024 ns PIM I/O interval. These parameters are
model assumptions inherited from the Local HBM-PIM timing configuration; no
physical CXL-PIM device was calibrated.

For each layer, the B8/C32k shape is:

| quantity | value |
|---|---:|
| spilled KV bytes | 142,390,528 B |
| spilled context positions across the batch | 69,527 |
| query bytes | 65,536 B |
| query link transactions | 2,048 |
| partial-result bytes | 1,064,960 B |
| result link transactions | 33,280 |
| PIM GWRITE waves | 2,048 |
| PIM MAC waves | 92,703 |
| PIM READ waves | 4,160 |

The partial state contains one FP32 maximum, one FP32 exponential sum and one
FP32 128-element value vector for every request, Attention head and CXL-PIM
pseudo-channel. QK and AV contribute 569,565,184 modeled MAC operations. As in
the existing Sieve PIM model, one wave provides
`PCH × bank × transaction_bytes` modeled operations; this is a simulator
assumption rather than a measured compute rate.

## Exact phase results

Five isolated Ramulator phases completed with no interpolation:

| phase | exact latency per layer |
|---|---:|
| query link | 1.367 us |
| query GWRITE | 2.098 us |
| Attention MAC | 2,278.270 us |
| partial READ | 4.273 us |
| result link | 18.419 us |
| total serial CXL-PIM branch | 2,304.426 us |

The link runs use the nominal 64 GB/s, 0.25 us, four-channel FIFO profile. The
query write is represented by the existing FIFO READ proxy with the same byte
service; direction-specific protocol behavior is absent. The PIM operations use
the isolated Sieve PIM controller and therefore do not contend with the link or
with each other inside one Ramulator run.

## End-to-end sensitivity

The A800 fit assigns the resident 73.478% of the KV context to the GPU branch,
giving an equivalent Context of 24,077.172 tokens and 446.917 ms of local GPU
Attention. The 48-layer CXL-PIM branch totals 110.612 ms.

| scenario | estimated Decode | throughput | change vs memory-only CXL |
|---|---:|---:|---:|
| capacity-unconstrained all Local HBM | 726.037 ms | 11.019 token/s | +14.946% |
| nominal memory-only CXL | 834.554 ms | 9.586 token/s | baseline |
| CXL-PIM, ideal branch overlap | 573.311 ms | 13.954 token/s | +45.567% |
| CXL-PIM, serialized branches | 683.924 ms | 11.697 token/s | +22.024% |

The two CXL-PIM rows form a scheduling interval using the same exact phase
results. Ideal overlap takes the maximum of the GPU and CXL-PIM Attention
branches; the serialized bound adds them. Both retain zero partial-merge cost.

Raw link bytes fall from 142,390,528 B per layer for memory-only CXL to
1,130,496 B for CXL-PIM query and partial results, a 99.206% reduction. In both
scheduling bounds the A800-fit local GPU branch is longer than the CXL-PIM
branch. The result therefore provides a conditional motivation for near-data
Attention, not proof of a realized speedup.

## Evidence boundary

The five phase timings are exact Ramulator results for their individual request
streams. The B8/C32k GPU timing is still an A800-based extrapolation. The combined
Decode values additionally depend on the assumed work split and scheduling rule.

The experiment omits softmax scalar-operation latency, partial-state merge cost,
direction-specific CXL protocol behavior, a unified CXL-PIM controller, phase
contention, paging, eviction, recovery, energy, multi-GPU sharing and physical
CXL measurements. It does not implement Shared CXL-PIM.

Artifacts are under `results/kv_cxl_pim_attention_stage8_v1/`. The validator
binds the Stage 7 inputs, model, cycle configuration, CXL and PIM source context,
Ramulator revision, binding, shared library, five exact caches and generated
outputs. Its status is `passed`, with five exact phase results and zero
interpolation.

## Next gate

The next experiment should quantify dependence on the assumptions that currently
drive the result. The primary matrix should vary CXL-PIM channel/PCH count, PIM
MAC interval, partial precision and merge latency, and report both ideal-overlap
and serialized scheduling. A unified request-level controller should be built only
after this sensitivity identifies a stable and useful region; multi-GPU and
Shared CXL-PIM remain later stages.
