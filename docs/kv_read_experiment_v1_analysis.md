# KV READ 请求级竞争实验 v1

## 实验范围与证据分类

本实验检查抽象 KV READ 与 Expert 请求同时进入 HBM 后的完成时间、驻留时间和
注入背压。模拟硬件仍为 1 GPU + 8 Local HBM-PIM stacks。它是独立的 Ramulator
请求级微基准，不替换 Attention → Router → Expert 串行端到端回放。

三类证据分别保留：

- 解析敏感性与容量准入：`results/kv_capacity_sweep_v1/` 和
  `results/kv_capacity_states_v1/`。80 GB A800 与 96 GB 模拟容量分别比较；正式
  spill budget 为 0，超容量配置为 `oom/infeasible`，没有分页或数据搬运时延。
- A800 真实捕获：每组独立的 prompts、Router Trace、manifest 及 SHA-256，只提供
  路由和 Context 输入，不提供本报告中的时延；prompt 为受控长度构造。
- Ramulator 模拟：`results/kv_read_experiment_v1/`，基于固定 revision、独立
  frontend、请求模型和配置哈希。旧的 35 组端到端回放及其输入保持冻结。

四组输入为 B8/C4k、B8/C8k、B8/C16k、B16/C8k；每组只取 step=0、layer=0，
不覆盖每组完整的 384 个 layer-batch。三组 B8 的这个代表层均有 8 个 active
experts，每个专家 8 token，Expert ID 不同但负载计数 shape 一样；B16 对应
每专家 16 token。因而这些代表层不能说明全 Trace 的路由多样性。

每组比较三种固定放置：GPU-only；继承旧 Expert-only `sieve-cycle-v1` 的
frozen-oracle；以及按 `(-token_count, expert_id)` 排序，将前 `ceil(active/2)`
个专家放在 GPU 的 fixed-half-prefix。本轮后者为 4 GPU + 4 PIM，不是 fixed-16
消融。Oracle 未根据 KV 竞争重新优化。

每种放置分别运行 expert-only、kv-only、combined，36 个消费者去重为 19 个
workload shape。KV 读取量按每层完整读取 K+V 的假设计算：

```text
kv_read_bytes_per_layer =
    2 × sum(context_lengths) × num_key_value_heads × head_dim × dtype_bytes
kv_read_transactions = ceil(kv_read_bytes_per_layer / 32)
```

B8/C4k、B8/C8k、B8/C16k 分别为 67,108,864、134,217,728、268,435,456 bytes，
即 2,097,152、4,194,304、8,388,608 个 READ。B16/C8k 与 B8/C16k 的 KV 字节数
相同。这里是单层读取量，区别于容量报告中的全模型 KV 存储量。

## 请求模型与指标语义

KV 与 GPU Expert 使用同一 controller READ 队列，source ID 分别为 1 与 0。
每个 pseudo-channel 每 tick 尝试一个 normal READ，成功接纳后轮换两条流；
combined 从起始时刻同时注入，因此显式施加了当前单层串行图中不存在的重叠。
地址为抽象顺序请求，GPU/KV 使用独立 row 范围但共享 banks；没有真实 GPU
访存 trace、cache hit、压缩或分页。PIM 命令保留旧的地址语义，可能与 GPU
row 范围重叠，不能解释为已验证的物理存储布局。

`combined − max(isolated)` 表示相对理想重叠完成时间的差，不是相对串行解码的
额外延迟。`expert-only + kv-only` 只是串行内存服务参考，也不包含完整解码计算。
GPU/PIM Expert completion delta 分别反映相应流完成时间的变化；无请求流的
completion 记为 0，不代表执行了零延迟任务。

请求 residence 从接纳到完成计算；accepted-to-column-issue 扣除固定 DRAM
READ latency，仍包含仲裁、ACT/PRE 准备等，不能称为纯 FIFO 等待。
injection-rejected-attempts 统计失败重试次数，不是丢失请求数，也不是去重后的
受阻请求数。旧 controller 的 `gpu_column_issues` 和
`gpu_blocked_by_pim_cycles` 此时包含所有 normal READ，不能用于区分 KV/Expert；
两条流的分解使用新增 frontend counters。

## 复现与完整性

```bash
RAMULATOR_BUILD_JOBS=4 bash scripts/build_kv_read_ramulator.sh
PYTHONPATH=src python3 scripts/freeze_kv_baseline.py --verify
PYTHONPATH=src python3 scripts/run_kv_read_experiments.py --run --workers 12
PYTHONPATH=src python3 scripts/validate_kv_read_results.py
PYTHONPATH=src python3 scripts/plot_kv_read_experiments.py
```

绘图依赖为 `requirements/kv_analysis.txt`。运行使用独立、可恢复的精确缓存；
缓存与 Ramulator 构建产物不提交。结果文件保留每个 shape 的请求计数、完整
时序、来源哈希和缓存哈希。审计还逐字段比较 zero-KV 与旧 Expert-only 缓存，
验证 350 个冻结文件保持不变。独立审计读取本地缓存，需要先完成精确运行。

任何竞争结论仅适用于上述注入、地址、队列和硬件参数。真实 A800 KV 带宽瓶颈、
端到端重叠调度收益、分页、spill transfer、eviction 和跨 workload 泛化均未验证。
