# KV CXL Decode Stage 5

本阶段把 CXL memory-only KV READ 的请求级完成 milestone 接入 48 层、1 step 的 Decode event graph。
路由来自 manifest 校验的真实 B8/C4k trace，并按 Batch/Context 做受控 tiling；CXL 带宽、延迟和容量是明确假设。
结果是 Ramulator request-level + 事件图模拟，不是 A800 实机计时，也没有实现 CXL-PIM 计算。
请求级 CXL queue wait 是所有请求等待时间之和，单独列出，不作为端到端关键路径。

| case | policy | capacity | total us | throughput tok/s | attention share | expert critical us | CXL critical us | CXL queue wait (all requests) us |
|---|---|---|---:|---:|---:|---:|---:|---:|
| b8_c32k | gpu-only | spill | 971.803942 | 8232.113 | 0.050202 | 159.266086 | 0.000000 | 14344.806528 |
| b8_c32k | sieve | spill | 24469.995814 | 326.931 | 0.923048 | 159.266086 | 0.000000 | 14344.806528 |
| b16_c16k | gpu-only | spill | 991.588556 | 16135.725 | 0.049993 | 172.150988 | 0.000000 | 14344.806528 |
| b16_c16k | sieve | spill | 24527.393996 | 652.332 | 0.922453 | 172.150988 | 0.000000 | 14344.806528 |
| b16_c32k | gpu-only | spill | 1089.798860 | 14681.608 | 0.135606 | 172.150988 | 147.783168 | 124265.576448 |
| b16_c32k | sieve | spill | 47075.972300 | 339.876 | 0.959597 | 172.150988 | 0.000000 | 124265.576448 |
| b32_c16k | gpu-only | spill | 1127.795224 | 28373.945 | 0.131037 | 197.920792 | 147.783168 | 124265.576448 |
| b32_c16k | sieve | spill | 47190.768664 | 678.099 | 0.958890 | 197.920792 | 0.000000 | 124265.576448 |

## Scope and limitations

- The two policies use the same controlled B8/C4k route template tiled to the selected Batch; no new A800 Router capture is created.
- The CXL controller is the Stage-4 FIFO memory-only model with the nominal 64 GB/s, 0.25 us, four-channel profile.
- `request_scale=4096` reduces Expert/PIM/KV request counts for a tractable exact shape run; the graph still contains analytic GPU compute and fixed model stages.
- PIM timing uses the final PIM completion milestone exposed by the frontend; intermediate PIM events are zero-duration bookkeeping, so no stage-level interpolation is claimed.
- A800 80 GB plus CXL 64 GB is kept separate from the 96 GB simulated Local-HBM capacity. The latter is reported only as a comparison state without CXL.
- No paging, eviction, recovery, physical CXL protocol, multi-GPU, or CXL-PIM computation is modeled.
- A combined `request_scale=1` B8/C32k gate was attempted separately but stopped before its first cache entry because the current per-cycle frontend makes the roughly 19 million-request shape too slow; it is not part of this matrix.

Exact request runs: `8`; interpolation: `0`.
