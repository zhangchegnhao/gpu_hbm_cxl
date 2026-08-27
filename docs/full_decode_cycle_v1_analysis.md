# 48层完整Decode cycle-v1实验分析

## 实验问题

本实验在固定的单GPU、8个Local HBM-PIM Stack和Qwen3-30B-A3B配置下，
比较独立时延表`cycle-v0`与GPU/PIM混合请求级时序模型`cycle-v1`，重点回答：

1. 请求级专家权重访存时延会怎样改变五个已有策略的排序；
2. `sieve-cycle-v1`能否利用精确候选表找到更均衡的GPU/PIM放置；
3. 结果变化来自路径时延校准、显式竞争，还是策略选择。

## 固定实验条件

- 模型：Qwen3-30B-A3B，BF16，48层，128个专家，Top-8；
- Trace：确定性合成Router Trace，Batch 8，Context 1024/1025，2个Decode step；
- 执行：96次层执行严格按`step -> layer`串行，每层完整经过Attention、Router、
  GPU/PIM Expert和Combine；
- 硬件：1个解析建模GPU、8个Local HBM3E-PIM Stack、256个伪通道、96 GB；
- Ramulator：固定revision `b30320bc9385b708e86b67ebb9f48858cc66d798`；
- cycle-v1：逻辑双Row Buffer、无刷新，GPU Expert READ与PIM命令进入同一次
  Ramulator运行；GPU算术和GPU Attention仍为解析模型，PIM Attention仍来自
  cycle-v0独立时延表。

完整Trace产生14,208个放置消费者，去重为146个底层workload shape。所有146项
均通过真实Ramulator运行精确生成，没有插值。物化后的schema-v3表包含4,250个
逐放置条目，对应96个层/step工作负载的全部热点前缀候选。

## cycle-v1六策略结果

| 策略 | 总时延(ms) | 平均step(ms) | 吞吐(token/s) | GPU/PIM专家数/层 | GPU/PIM token总数 |
|---|---:|---:|---:|---:|---:|
| sieve-cycle-v1 | 10.145 | 5.072 | 1577.184 | 16 / 33 | 2976 / 3168 |
| pimoe | 10.884 | 5.442 | 1470.054 | 2 / 47 | 1248 / 4896 |
| allexp | 12.803 | 6.401 | 1249.718 | 0 / 49 | 0 / 6144 |
| sieve | 14.646 | 7.323 | 1092.448 | 32 / 17 | 4512 / 1632 |
| gpu-only | 16.109 | 8.055 | 993.203 | 49 / 0 | 6144 / 0 |
| noexp | 17.627 | 8.813 | 907.721 | 49 / 0 | 6144 / 0 |

相对其他cycle-v1策略，`sieve-cycle-v1`总时延分别降低：

- 相对`pimoe`：6.79%；
- 相对`allexp`：20.76%；
- 相对原`Sieve`解析放置：30.73%；
- 相对`gpu-only`：37.03%；
- 相对`noexp`：42.45%。

两个step时延分别为5071.741 us和5072.924 us。差异主要来自Context从1024增加到
1025；两步的专家负载shape和放置比例相同。

## 为什么竞争感知策略更快

合成Trace的每层有64次专家分配、49个活跃专家。专家ID随层旋转，但token-count
分布保持一致。因此96个层执行都从50个热点前缀候选中选择了相同的16/33数量拆分，
具体专家ID仍随层变化。

被选候选的单层专家路径为：

```text
GPU path = 47.121984 us Ramulator READ + 3.130023 us analytic compute
         = 50.252007 us
PIM path = 50.844768 us
scheduler + max(GPU path, PIM path)
         = 20.000000 + 50.844768
         = 70.844768 us
```

两条路径只相差0.593 us，接近平衡。原`sieve`使用解析时延选择32/17拆分，在
cycle-v1下每层GPU内存路径达到94.537 us，而PIM路径只有26.218 us，GPU侧明显
成为关键路径。`pimoe`的2/47拆分则使PIM路径达到78.546 us。`sieve-cycle-v1`
的收益主要来自根据请求级时延重新选择16/33分界点。

## cycle-v0与cycle-v1差异

| 策略 | cycle-v0(ms) | cycle-v1(ms) | 时延变化 |
|---|---:|---:|---:|
| gpu-only | 7.689 | 16.109 | +109.50% |
| noexp | 9.206 | 17.627 | +91.46% |
| allexp | 11.401 | 12.803 | +12.29% |
| pimoe | 9.765 | 10.884 | +11.46% |
| sieve | 9.194 | 14.646 | +59.29% |

最大的变化不是显式GPU/PIM竞争，而是专家权重访存的建模方式。`gpu-only`没有
PIM专家竞争，但GPU权重读取仍从cycle-v0的理想8 TB/s解析估计57.803 us/层，变为
cycle-v1请求级Ramulator结果145.513 us/层，96层累计正好增加8420.159 us。
因此不能把上表全部差值称为“竞争开销”。

PIM侧也发生时延模型变化。`allexp`的PIM专家路径从cycle-v0累计约8057.6 us
增加到cycle-v1的9459.3 us，使端到端时延上升12.29%。这部分同样包含命令级模型
校准，而不只是竞争。

## 显式竞争统计

| 策略 | GPU隔离/竞争(us/层) | PIM隔离/竞争(us/层) | GPU/PIM竞争delta(us/层) |
|---|---:|---:|---:|
| pimoe | 5.407 / 5.406 | 78.545 / 78.546 | -0.000936 / +0.000936 |
| sieve | 94.814 / 94.537 | 26.215 / 26.218 | -0.277368 / +0.002808 |
| sieve-cycle-v1 | 47.130 / 47.122 | 50.841 / 50.845 | -0.008424 / +0.003744 |

在当前双Row Buffer模型和这些shape下，完成里程碑的显式竞争delta很小。
`sieve-cycle-v1`每层记录1,450,368个`gpu_blocked_by_pim_cycles`和1,664个
`pim_blocked_by_gpu_cycles`，但这些计数是跨控制器累加的阻塞事件，不能直接乘以
312 ps当作端到端时延。GPU侧出现微小负delta是FRFCFS混合流重排后的完成里程碑
变化，不应解释为“竞争带来加速”。

因此当前结果支持的结论是：竞争感知表最重要的作用是提供了不同GPU/PIM拆分下的
请求级路径时延，使调度器找到接近平衡的16/33拆分；不能声称观察到了很大的直接
竞争惩罚。

## 结果边界

本结果仍是模拟器阶段性结果，不是论文级硬件性能结论：

- Router Trace是确定性合成数据，不是真实Qwen3 Router输出；
- 只有两个Decode step，不能代表长序列或连续批处理；
- GPU算术、Router、Scheduler和GPU Attention仍是解析模型；
- PIM Attention没有进入cycle-v1混合请求模拟；
- GPU READ是抽象的顺序请求流，不是真实GPU访存trace；
- 当前无刷新、能耗、原生逐Bank PIM ACT/PRE、多GPU、NVLink和Shared CXL-PIM；
- `pimoe`仍是静态token阈值占位实现；
- 合成Trace的负载shape跨层相同，因此尚未验证放置数量随真实层负载动态变化。

所有策略的容量检查通过：模型权重、峰值KV Cache和激活合计约61.87 GB，小于
96 GB。该检查只证明容量可行，不代表真实数据布局和驻留策略已经实现。

## 结果文件

- `results/full_decode_cycle_v1/comparison.csv`：六策略端到端结果；
- `results/full_decode_cycle_v1/cycle_v0_vs_v1.csv`：五个共同策略的后端差异；
- `results/full_decode_cycle_v1/contention.csv`：六策略竞争总量与每层均值；
- `results/full_decode_cycle_v1/<policy>/contention.csv`：96层逐层竞争记录；
- `results/full_decode_cycle_v1/sieve-cycle-v1/placement_search.csv`：4,800个候选记录；
- `ramulator/timing_tables/generated/qwen3_full_decode_cycle_v1_contention.json`：
  schema-v3正式时延表；
- 对应`*.evidence.json`：缓存键数量、配置和工作负载哈希。
