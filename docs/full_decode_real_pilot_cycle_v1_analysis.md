# Qwen3真实Router Trace完整Decode cycle-v1实验分析

## 实验范围与数据来源

本实验使用真实Qwen3-30B-A3B Router Trace驱动48层、8个Decode step的完整Decode
回放。模型执行只提供路由决策，所有时延仍由仓库中的解析模型和Ramulator时序
表产生。实验范围是1个GPU和8个Local HBM-PIM Stack；Shared CXL-PIM和多GPU不在
本实验中。

- 模型：Qwen3-30B-A3B，BF16，48层，128个专家，Top-8；
- Trace：`traces/real/qwen3_30b_a3b/pilot_8x8/router.jsonl`，manifest绑定；
- Trace规模：384个layer-batch、3072条Router记录、24,576次专家分配；
- 真实负载：340个unique load signatures，活跃专家数范围20--49，均值36.830729；
- Ramulator workload：1030/1030个shape按当前provenance精确命中，插值为0；
- contention表：schema-v3，14,527条逐放置entry，1,030个unique cache key，
  `dual_row_buffer=true`，固定Ramulator revision
  `b30320bc9385b708e86b67ebb9f48858cc66d798`。

`qwen3_real_pilot_decode_cycle_v1_contention.json`通过正式
`RamulatorContentionTable.load`和输入哈希校验。七个策略均生成384条`layers.csv`
和384条逐层`contention.csv`，每个策略有8条step记录、6,912个事件，`summary.json`
和`run_manifest.json`中的模型、硬件、Trace、cycle配置及时序表SHA-256均与当前
输入一致。所有策略的容量检查均通过（峰值总容量约61.08 GB，小于96 GB）。

## 七策略总时延

总时延、step统计和放置汇总如下。时延单位为ms；专家数是384个layer-batch上的
平均每层执行数，token列是整个pilot的总数。

| 策略 | 总时延 | 平均step | P95 step | 吞吐(request token/s) | GPU专家/层 | PIM专家/层 | GPU tokens | PIM tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sieve-cycle-v1 | 30.551096 | 3.818887 | 4.035810 | 2094.851 | 12.765625 | 24.065104 | 14538 | 10038 |
| sieve-fixed-16-cycle-v1 | 34.003663 | 4.250458 | 4.255286 | 1882.150 | 16.000000 | 20.830729 | 16450 | 8126 |
| pimoe | 35.457817 | 4.432227 | 5.022123 | 1804.962 | 3.315104 | 33.515625 | 6136 | 18440 |
| sieve | 37.673216 | 4.709152 | 5.141694 | 1698.820 | 19.244792 | 17.585938 | 17775 | 6801 |
| allexp | 44.893344 | 5.611668 | 5.614029 | 1425.601 | 0.000000 | 36.830729 | 0 | 24576 |
| gpu-only | 49.714591 | 6.214324 | 7.089151 | 1287.348 | 36.830729 | 0.000000 | 24576 | 0 |
| noexp | 50.254423 | 6.281803 | 7.158710 | 1273.520 | 36.830729 | 0.000000 | 24576 | 0 |

在这个pilot和当前模型参数下，`sieve-cycle-v1`是最快策略，比`pimoe`低13.84%，
比`allexp`低31.94%，比`gpu-only`低38.55%。这些是模拟器总时延比较，不是实机
性能或论文复现结果。

## cycle-v0与cycle-v1

共同的五个策略比较如下：

| 策略 | cycle-v0 (ms) | cycle-v1 (ms) | 变化 (ms) | 变化 (%) |
|---|---:|---:|---:|---:|
| gpu-only | 24.455069 | 49.714591 | +25.259522 | +103.29 |
| noexp | 24.994901 | 50.254423 | +25.259522 | +101.06 |
| allexp | 39.286449 | 44.893344 | +5.606895 | +14.27 |
| pimoe | 31.240782 | 35.457817 | +4.217035 | +13.50 |
| sieve | 24.680213 | 37.673216 | +12.993003 | +52.65 |

cycle-v0使用独立Ramulator PIM时序和解析GPU时序；cycle-v1将GPU Expert READ与
PIM命令放入同一次请求级Ramulator运行，GPU算术仍为解析模型。因此变化同时包含
访存后端替换、请求排队和放置目标变化，不能把整列差值解释成GPU/PIM竞争惩罚。
例如`gpu-only`没有PIM专家竞争，但GPU权重读取后端仍从cycle-v0解析估计变为
cycle-v1请求级结果，所以其增幅主要是时序模型变化。`allexp`和`pimoe`的变化也
同时包含PIM命令级模型校准。

## 动态GPU/PIM分界与固定16消融

`sieve-cycle-v1`对每个layer-batch枚举真实hot-prefix候选，并用

```text
scheduler + max(gpu_contended + analytic_gpu_compute, pim_contended)
```

选择目标。384个layer-batch的GPU hot-prefix分布为：

| GPU prefix | layer-batch数 | unique signature数 |
|---:|---:|---:|
| 7 | 1 | 1 |
| 8 | 1 | 1 |
| 9 | 10 | 10 |
| 10 | 22 | 22 |
| 11 | 33 | 33 |
| 12 | 84 | 82 |
| 13 | 108 | 92 |
| 14 | 81 | 66 |
| 15 | 36 | 27 |
| 16 | 8 | 6 |

因此动态边界范围为7--16个GPU专家，均值12.765625个GPU专家和24.065104个PIM
专家。真实Trace的340个signature中，301个出现1次、34个出现2次、5个出现3次；
每个signature在本实验中映射到唯一的hot-prefix，没有一个signature产生多个
不同边界。活跃专家数从20变化到49，边界随负载形状变化，说明真实路由输入确实
触发了动态候选选择；这不等价于证明在更长序列或其他硬件上仍有相同收益。

固定16消融始终使用16个GPU专家和20.830729个PIM专家。它的总时延为34.003663 ms，
动态策略为30.551096 ms，动态策略减少3.452567 ms（10.15%）。逐层比较中384层有
376层动态结果更快、8层相同；相同的8层对应动态选择16-prefix。固定分界消融因此
不应再沿用合成Trace阶段“动态与固定完全相同”的结论，真实负载signature改变了
结论。

## Isolated/contended delta

下表是`contention_summary`中384层累计的GPU和PIM隔离/竞争差值，单位为us；负的
GPU delta表示混合请求完成里程碑在当前FRFCFS请求重排下略早，不表示竞争提供了
真实加速。

| 策略 | GPU contention delta | PIM contention delta | total memory phase |
|---|---:|---:|---:|
| gpu-only | 0.000000 | 0.000000 | 41943.283200 |
| noexp | 0.000000 | 0.000000 | 41943.283200 |
| allexp | 0.000000 | 0.000000 | 37837.283328 |
| pimoe | +1.869192 | +0.454896 | 28398.417216 |
| sieve | -107.612856 | +1.137240 | 21710.602056 |
| sieve-cycle-v1 | -13.776048 | +1.108224 | 15470.373984 |
| sieve-fixed-16-cycle-v1 | -52.062816 | +1.190904 | 18065.039304 |

在当前双Row Buffer和这些请求shape下，显式PIM竞争delta只有约0.45--1.19 us/384
层；GPU侧负delta来自完成顺序变化。`sieve-cycle-v1`的主要收益来自请求级时序
下重新选择更均衡的GPU/PIM路径，而不是一个可独立归因的大型竞争惩罚。

## `pimoe`与科研结论边界

当前`pimoe`只是静态token阈值占位策略（阈值来自配置），不是正式PIMoE论文复现，
不能据此声称复现论文算法或收益。所有结果仍是阶段性模拟器结果：GPU算术、Router、
Scheduler和GPU Attention使用解析模型；PIM Attention没有进入cycle-v1混合请求
模拟；GPU READ是抽象顺序请求流而非真实GPU访存Trace；未建模刷新、能耗、原生逐
Bank PIM命令、多GPU、NVLink和Shared CXL-PIM；容量检查不代表真实布局和驻留已经
实现。本pilot固定为8个Decode step，不能代表更长序列或连续批处理，也不能把结果
描述成真实B200性能。

## 产物

- `ramulator/timing_tables/generated/qwen3_real_pilot_decode_cycle_v1_contention.json`
  及对应`.evidence.json`：真实Trace绑定的schema-v3竞争表；
- `results/full_decode_real_pilot_cycle_v1/`：七策略回放、逐层事件、放置、竞争和
  输入manifest；
- `results/full_decode_real_pilot_cycle_v0/`：cycle-v0五策略基线；
- `results/router_trace_analysis/real_pilot/`：340个真实load signature及路由统计。
