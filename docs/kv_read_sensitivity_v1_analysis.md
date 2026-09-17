# KV READ 控制器敏感性分析 v1

本轮 5 个控制器变体 × 19 个去重 workload 的 95 个精确结果全部完成，失败为 0、
插值为 0。结果支持“在当前同时注入的抽象请求模型下，KV 与 Expert 存在竞争”，
但竞争量级和随 Context 的变化依赖控制器假设。当前证据不足以把这一模型升级为
真实 A800 带宽瓶颈结论，也不足以把同时注入的结果加入串行端到端回放。

## 实验范围和结果来源

输入沿用 `results/kv_read_experiment_v1/kv_read_plan.json` 的四组真实 Router
捕获：B8/C4k、B8/C8k、B8/C16k、B16/C8k，每组仅取 step=0、layer=0。
三种放置为 GPU-only、继承旧 Expert-only Oracle 的 frozen-oracle，以及
fixed-half-prefix；本轮分别为 8/0、7/1、4/4 个 GPU/PIM 专家。没有依据 KV
结果重新选择放置。每种放置有 expert-only、kv-only、combined 三种模式。

| 变体 | READ buffer | PIM/normal row-buffer 假设 |
|---|---:|---|
| `baseline-rb256-dual` | 256 | dual |
| `rb128-dual` | 128 | dual |
| `rb64-dual` | 64 | dual |
| `rb32-dual` | 32 | dual |
| `rb256-single` | 256 | single |

每个变体的 36 个消费者去重为同一组 19 个请求 shape；因此是 95 个
variant–shape 精确结果、180 个模式消费者，以及 60 组可配对的三模式比较，
不是 95 个独立 workload 或 95 次统计重复。原始文件为：

- `results/kv_read_sensitivity_v1/kv_read_sensitivity_plan.json`：参数快照和来源哈希；
- `results/kv_read_sensitivity_v1/kv_read_sensitivity_exact_results.json`：95 个结果及缓存哈希；
- `results/kv_read_sensitivity_v1/kv_read_sensitivity_comparison.json` 和同名 CSV：
  相同 shape 相对基准控制器的比较。
- `results/kv_read_sensitivity_v1/matched_comparison.json` 和同名 CSV：独立审计
  重建的 60 组三模式比较，含逐流增量、比例、准入后延迟及每请求拒绝尝试次数；
- `results/kv_read_sensitivity_v1/validation.json`：来源、请求计数和配对结果审计。

`plots/` 保存 B8 三种放置下的配对竞争增量、GPU Expert 完成增量和 KV 准入后
驻留曲线（PNG/SVG）；由 `scripts/plot_kv_read_sensitivity.py` 读取审计后的
60 组比较生成，B16/C8k 保留在表中。`plots_manifest.json` 绑定输入及图像哈希。

所有变体继续使用 simultaneous injection，KV 与 GPU Expert 为普通 HBM READ，
在相同 bank 上使用不同 row 范围。single/dual 切换的是控制器中 PIM 与 normal
请求的 row-buffer 假设，**没有改变 KV/Expert 地址布局**；没有实现 staggered、
serial 注入或交错地址布局扫描。PIM 命令和其他参数保持计划中的值。

## 控制器变化与竞争增量须分别解释

现有 comparison 文件中的 `delta_total_completion_us` 定义为：

```text
controller_delta(v, shape) = T(v, shape) - T(baseline, shape)
```

下表对每个变体的 19 个去重 shape 等权统计，既包含 combined，也包含
expert-only 和 kv-only。它描述控制器敏感性，不是 KV 的平均竞争惩罚，也不是
按真实层频率加权的延迟。

| 变体 | 平均 controller delta (μs) | 最小 (μs) | 最大 (μs) | 最大相对增幅 |
|---|---:|---:|---:|---:|
| baseline-rb256-dual | 0.000000 | 0.000000 | 0.000000 | 0.00% |
| rb128-dual | 0.540466 | -0.000624 | 1.263600 | 2.89% |
| rb64-dual | 2.587958 | -0.000312 | 6.474936 | 13.84% |
| rb32-dual | 12.879853 | 0.000000 | 45.458088 | 110.86% |
| rb256-single | 10.148670 | 0.000000 | 62.111400 | 69.85% |

最大绝对变化和最大相对变化未必来自同一 shape。负值分别只有 2 个和 1 个
312 ps tick，来自确定性调度变化，不是测量噪声。single 变体下，全部 7 个
不含 PIM 的 shape 与基准结果一致；含 PIM 的 Expert-only 结果自身也发生变化，
因此不能把 single 相对 dual 的全部差值归因于 KV。

为隔离同一个控制器假设内的并发影响，从原始消费者关系重新匹配三模式，定义：

```text
D(v, case, placement) = T_combined(v) - max(T_expert_only(v), T_kv_only(v))
R(v, case, placement) = T_combined(v) / max(T_expert_only(v), T_kv_only(v))
```

这里的 max 是两个互不干扰流同时开始时的完成时间参照，不是当前串行 Decode
图的延迟参照。60 组配对的 D 全为正，范围为 0.016848–64.617696 μs，R 的
范围为 1.000244–4.044676。下表列出全部 60 个 D（单位 μs）。

| 配置 | 放置 | rb256 dual | rb128 dual | rb64 dual | rb32 dual | rb256 single |
|---|---|---:|---:|---:|---:|---:|
| B8/C4k | gpu-only | 20.399496 | 21.459672 | 26.119392 | 41.953704 | 20.399496 |
| B8/C4k | frozen-oracle | 20.047248 | 20.945808 | 25.122864 | 64.617696 | 30.421248 |
| B8/C4k | fixed-half-prefix | 0.016848 | 0.028080 | 0.036192 | 0.045864 | 21.214128 |
| B8/C8k | gpu-only | 22.991592 | 24.022128 | 29.170752 | 47.107632 | 22.991592 |
| B8/C8k | frozen-oracle | 20.022912 | 20.980752 | 25.414896 | 41.078232 | 22.384752 |
| B8/C8k | fixed-half-prefix | 4.065672 | 4.815408 | 7.081776 | 16.554096 | 31.697328 |
| B8/C16k | gpu-only | 23.035272 | 24.022128 | 29.170752 | 47.107632 | 23.035272 |
| B8/C16k | frozen-oracle | 20.039448 | 20.980752 | 25.414896 | 64.617696 | 22.166352 |
| B8/C16k | fixed-half-prefix | 11.051664 | 11.840400 | 14.218464 | 23.242440 | 20.630688 |
| B16/C8k | gpu-only | 23.035272 | 24.022128 | 29.170752 | 47.107632 | 23.035272 |
| B16/C8k | frozen-oracle | 19.677216 | 20.618208 | 24.963744 | 40.330680 | 24.272976 |
| B16/C8k | fixed-half-prefix | 0.024024 | 0.033384 | 0.300456 | 10.009272 | 52.501176 |

B8/C16k 与 B16/C8k 的 KV 请求数相同；GPU-only 的 Expert 权重 READ shape
也相同，因而复用结果，不能视为独立佐证。三组 B8 的代表层 Expert 负载计数
相同，专家 ID 有变化；这仍不能代表完整 Trace 的负载分布。

## 逐流完成与背压的含义

GPU Expert 完成增量按同一变体内的
`GPU_completion(combined) - GPU_completion(expert-only)` 计算。
GPU-only 放置在 B8/C4k、C8k、C16k 的基准增量分别为
20.399496、32.295432、53.533896 μs；rb32 下为
41.953704、56.476992、77.761008 μs。这一放置下，各变体均呈现更长 KV
请求流推迟 GPU Expert 完成的趋势。

这个趋势不能扩展为所有放置和指标均单调：rb32 的 frozen-oracle GPU Expert
增量依次为 64.763400、51.848472、96.712200 μs；single 的 fixed-half-prefix
D 依次为 21.214128、31.697328、20.630688 μs。即使 D 很小，GPU Expert
也可能明显变慢，只是其完成时间仍被较长的 PIM 流覆盖。例如基准 B8/C4k
fixed-half-prefix 的 D 仅 0.016848 μs，GPU Expert 增量仍为 16.284840 μs。

`accepted-to-column-issue` 包含准入后的仲裁和 ACT/PRE 准备，不是纯 FIFO
等待，也不含准入前被拒绝的等待。B8/C16k fixed-half-prefix 的 KV 均值在
基准、rb128、rb64、rb32 下分别为 0.333391、0.168708、0.086926、
0.047005 μs，而 KV 拒绝注入次数分别为 32,287,744、36,155,776、
37,219,712、42,437,248。更小队列下较低的已准入请求驻留均值不等于更低
整体延迟：压力可表现为队列外的反复准入失败。拒绝次数是尝试次数，不能当作
独立请求数、丢失请求数或 GPU 硬件 stall cycle。

## 可复现性与结论边界

本报告对应的文件 SHA-256 为：

| 输入 | SHA-256 |
|---|---|
| sensitivity plan | `a47f7d70aaf730c9839c92c7770f18fbb70e5c68332a1e26dd8a24b8a3c09499` |
| sensitivity exact results | `31954060210fe9d1f544873c26b5a123c5f5b8d0e4b371f272f9a667bac27b88` |
| base KV plan | `83b25936281e4fd31d2c50d9e1ae23b70b61a12abf521b37de341352522ff9c2` |
| frozen baseline inventory | `e60dc296d9c34384d1ec3c65e967f9eded2ae5b0f4b605c815e7ceb7cf1558ca` |

本轮审计以当前计划所需的 95 个 entry 为准，不能用缓存目录文件数判断完成。
目录中两份计划各留下 95 个缓存；当前计划所选缓存的哈希与结果一致，基准变体
的 19 个结果与旧 KV v1 逐字段相同，350 个冻结文件通过校验。
`scripts/audit_kv_read_sensitivity.py` 还独立检查了计划、源文件、95 个请求
shape 的计数和原始 comparison JSON/CSV；状态为
`passed_with_provenance_limitation`。计划记录的 runner/request-model 源文件
哈希可以校验；但历史运行未在 sensitivity cache context 中保存实际加载的
binding/library 二进制哈希，因此不能声称已完整证明执行二进制来源，也不能把
事后计算的二进制哈希补称为当时的记录。冻结清单只校验，不刷新。

三类证据仍须区分：解析容量准入只给字节估计和状态；A800 捕获只给真实 Router
及 prompt/Context 输入；这里的完成时间和背压全部来自 Ramulator。模型假设每层
完整读取 K+V，未模拟 GPU cache hit、真实 memory trace、分页、spill、eviction
或恢复。A800 80 GB 与模拟 96 GB 不合并；本轮不新增实测显存或 OOM 结论。

下一步应先取得独立 A800 Attention/Decode 计时和显存采样，以检查真实设备上的
Context 趋势。即使这些趋势与模拟一致，也不能单独证明同一请求队列竞争机制。
在尚无真实重叠执行依据、地址布局和注入方式验证之前，保留本实验为条件性的
请求级微基准，继续冻结正式 `Attention → Router → Expert` 回放；不直接建立
KV contention-v2 端到端性能结论。
