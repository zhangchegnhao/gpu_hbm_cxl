# A800 independent Attention/Decode timing v1

This experiment measures the real Qwen3-30B-A3B BF16 forward path on one
`NVIDIA A800-SXM4-80GB`. It is independent of Router capture: no Router hook and
no `.cpu().tolist()` synchronization are used. CUDA events measure prefill and
Decode forward time; attention-module CUDA events measure the sum of the 48
attention module intervals. CUDA allocator counters and the returned
`past_key_values` tensors provide the memory and KV-cache measurements.

The model revision is
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`. Each configuration used its own
prompt file, eight fixed Decode steps, and one warmup run. The prompt contains
`Context-1` prefill tokens; the first Decode token reaches the reported Context.
The five manifests, prompt snapshots, timing files, memory files, and aggregate
summary are under `results/a800_timing_memory_v1/`.

| case | Batch | Context | Prefill ms | Decode median ms | Attention share | KV after prefill | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B8/C4k | 8 | 4096 | 3892.591 | 221.865 | 43.19% | 3.220 GB | 66.310 GB | 66.870 GB |
| B8/C8k | 8 | 8192 | 7256.900 | 288.637 | 56.88% | 6.442 GB | 71.547 GB | 72.662 GB |
| B8/C16k | 8 | 16384 | 15722.408 | 436.649 | 71.26% | 12.884 GB | 82.021 GB | 84.224 GB |
| B16/C4k | 16 | 4096 | 6701.391 | 288.702 | 55.14% | 6.441 GB | 71.545 GB | 72.643 GB |
| B16/C8k | 16 | 8192 | 13605.554 | 423.847 | 68.83% | 12.883 GB | 82.017 GB | 84.205 GB |

The measured trend supports the next KV investigation: increasing Context raises
both KV bytes and Attention time, and Attention becomes a larger fraction of Decode
time. B8/C8k and B16/C4k have nearly the same `Batch x Context` product and nearly
the same Decode median, while B8/C16k and B16/C8k show the same product-level
scaling with a modest batch/context difference.

The largest completed configurations remain below the A800's reported 80 GiB device
capacity, but leave little allocator headroom. No real A800 OOM point was attempted
in this matrix. The results do not measure CXL, Local HBM-PIM, paging, eviction,
spill transfer, or Ramulator timing; they are A800 measurements of the original
GPU execution path and must be reported separately from the request-level CXL
simulation.

Aggregate summary hashes:

```text
summary.json  9391b3bdb73017b6801b0fa52d9716ec1e4346f24555b6150c4da6f1beb0d96d
summary.csv   ea677f49c8bcd1ed8af61870a6634ee5fa73ad74e5282628c7122d9320cee7d7
```
