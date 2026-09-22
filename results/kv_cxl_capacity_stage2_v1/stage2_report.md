# KV CXL capacity stage 2

本阶段只做容量准入敏感性分析：将 CXL 表示为显式的额外 memory-only spill budget。
A800 80 GB 和模拟 96 GB 是两个独立的本地容量域，CXL 字节不会被并入 Local HBM，
也不会把两个本地容量相加。结果不是 A800 实测、不是 Ramulator 时延，也不是 CXL-PIM。

## Pressure points

| domain | B | C | peak memory (GB) | CXL budget (GB) | state | feasible | spill (GB) | unallocated (GB) |
|---|---:|---:|---:|---:|---|---|---:|---:|
| a800 | 8 | 32768 | 86.834745 | 0 | oom | false | 0.000000 | 6.834745 |
| a800 | 8 | 32768 | 86.834745 | 16 | spill | true | 6.834745 | 0.000000 |
| a800 | 8 | 32768 | 86.834745 | 32 | spill | true | 6.834745 | 0.000000 |
| a800 | 8 | 32768 | 86.834745 | 64 | spill | true | 6.834745 | 0.000000 |
| simulated | 8 | 32768 | 86.834745 | 0 | resident | true | 0.000000 | 0.000000 |
| simulated | 8 | 32768 | 86.834745 | 16 | resident | true | 0.000000 | 0.000000 |
| simulated | 8 | 32768 | 86.834745 | 32 | resident | true | 0.000000 | 0.000000 |
| simulated | 8 | 32768 | 86.834745 | 64 | resident | true | 0.000000 | 0.000000 |
| a800 | 16 | 16384 | 86.835466 | 0 | oom | false | 0.000000 | 6.835466 |
| a800 | 16 | 16384 | 86.835466 | 16 | spill | true | 6.835466 | 0.000000 |
| a800 | 16 | 16384 | 86.835466 | 32 | spill | true | 6.835466 | 0.000000 |
| a800 | 16 | 16384 | 86.835466 | 64 | spill | true | 6.835466 | 0.000000 |
| simulated | 16 | 16384 | 86.835466 | 0 | resident | true | 0.000000 | 0.000000 |
| simulated | 16 | 16384 | 86.835466 | 16 | resident | true | 0.000000 | 0.000000 |
| simulated | 16 | 16384 | 86.835466 | 32 | resident | true | 0.000000 | 0.000000 |
| simulated | 16 | 16384 | 86.835466 | 64 | resident | true | 0.000000 | 0.000000 |
| a800 | 16 | 32768 | 112.605270 | 0 | oom | false | 0.000000 | 32.605270 |
| a800 | 16 | 32768 | 112.605270 | 16 | oom | false | 16.000000 | 16.605270 |
| a800 | 16 | 32768 | 112.605270 | 32 | oom | false | 32.000000 | 0.605270 |
| a800 | 16 | 32768 | 112.605270 | 64 | spill | true | 32.605270 | 0.000000 |
| simulated | 16 | 32768 | 112.605270 | 0 | oom | false | 0.000000 | 16.605270 |
| simulated | 16 | 32768 | 112.605270 | 16 | oom | false | 16.000000 | 0.605270 |
| simulated | 16 | 32768 | 112.605270 | 32 | spill | true | 16.605270 | 0.000000 |
| simulated | 16 | 32768 | 112.605270 | 64 | spill | true | 16.605270 | 0.000000 |
| a800 | 32 | 16384 | 112.606712 | 0 | oom | false | 0.000000 | 32.606712 |
| a800 | 32 | 16384 | 112.606712 | 16 | oom | false | 16.000000 | 16.606712 |
| a800 | 32 | 16384 | 112.606712 | 32 | oom | false | 32.000000 | 0.606712 |
| a800 | 32 | 16384 | 112.606712 | 64 | spill | true | 32.606712 | 0.000000 |
| simulated | 32 | 16384 | 112.606712 | 0 | oom | false | 0.000000 | 16.606712 |
| simulated | 32 | 16384 | 112.606712 | 16 | oom | false | 16.000000 | 0.606712 |
| simulated | 32 | 16384 | 112.606712 | 32 | spill | true | 16.606712 | 0.000000 |
| simulated | 32 | 16384 | 112.606712 | 64 | spill | true | 16.606712 | 0.000000 |

## Interpretation

`resident` 表示估算峰值全部放入本地容量；`spill` 表示超出本地容量的字节数
可以放入显式 CXL budget；`oom` 表示本地容量和该 budget 仍不足。`spill` 只是
容量准入状态，不代表已经实现传输、分页、驱逐或恢复。

本阶段使用 manifest 绑定的真实 Router 模板，仅为了保持模型和路由输入身份；
没有复制 Router Trace，也没有重新捕获 A800 timing。所有时延相关字段均不适用。

## Provenance

- method: `analytic-kv-cxl-capacity-admission-v1`
- input manifest validated: `True`
- pressure points: `4`
- CXL budgets (GB): `[0, 16, 32, 64]`
- limitations: no CXL link timing, CXL-PIM, paging, eviction, recovery, or A800 memory observation
