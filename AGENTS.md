# AGENTS.md

## 项目目标

本项目用于构建面向大语言模型推理的存算一体（Processing-in-Memory，PIM）科研
实验环境，当前主要研究Mixture-of-Experts（MoE）模型端到端Decode阶段。

长期研究设想是三级异构架构：

```text
GPU
+ Local HBM-PIM
+ Shared CXL-PIM
```

- GPU负责常规计算和适合GPU执行的任务；
- Local HBM-PIM与单个GPU紧耦合，提供高带宽、低延迟的近数据计算；
- Shared CXL-PIM是多个GPU可访问的共享容量和近数据计算资源。

核心问题是如何在端到端MoE Decode过程中，根据计算、访存和通信特点，在GPU、
Local HBM-PIM和Shared CXL-PIM之间划分和调度工作负载，降低推理延迟并提高系统
效率。

## 当前实现范围

项目已不再处于空白环境阶段。当前仓库实现了一个Trace驱动的单GPU MoE Decode
回放器，范围严格限定为：

```text
1 GPU + 8 Local HBM-PIM Stacks
```

Shared CXL-PIM、多GPU和跨GPU互连尚未实现，不得默认存在。

当前模型为Qwen3-30B-A3B，BF16、48层、128个专家、Top-8。每个层执行完整经过：

```text
Attention -> Router -> GPU/PIM Expert -> Combine
```

当前提供三类时延后端：

- `analytic-v0`：GPU和PIM均使用解析估计；
- `ramulator-table-v0`：PIM使用独立Ramulator时序，GPU仍为解析模型；
- `ramulator-contention-v1`：GPU Expert READ与PIM命令进入同一次请求级Ramulator
  运行，GPU算术和部分其他组件仍为解析模型。

当前新增`sieve-runtime-v1`在线策略：使用少量离线校准点的分段线性时序模型，
决策时不读取精确放置候选表；最终选定放置仍由`ramulator-contention-v1`精确表做
模拟器真值评价。`sieve-cycle-v1`在该阶段冻结为离线Oracle。

六个主要策略为`gpu-only`、`noexp`、`allexp`、`pimoe`、`sieve`和
`sieve-cycle-v1`。另有`sieve-fixed-16-cycle-v1`，只用于固定分界消融。
当前`pimoe`是静态token阈值占位策略，不是正式PIMoE论文复现，报告中必须明确。

## 已完成实验

48层合成Router Trace、Batch 8、2个Decode step的cycle-v1实验已经完成：

- 14,208个放置消费者去重为146个Ramulator workload shape；
- 146/146均由固定revision的Ramulator精确运行生成；
- 缺失workload为0，插值为0；
- schema-v3正式表包含4,250条精确放置时序；
- Ramulator revision为`b30320bc9385b708e86b67ebb9f48858cc66d798`；
- 六策略均完成96次层执行的端到端回放。

六策略总时延为：

| 策略 | 总时延 |
|---|---:|
| sieve-cycle-v1 | 10.144665 ms |
| pimoe | 10.883954 ms |
| allexp | 12.802888 ms |
| sieve | 14.646008 ms |
| gpu-only | 16.109494 ms |
| noexp | 17.626561 ms |

`sieve-cycle-v1`在所有层选择16个GPU专家和33个PIM专家。固定16专家消融与动态
搜索得到完全相同的10.144665 ms和逐层结果。这是因为当前合成Trace虽然轮换专家
ID，但96个层批次只有一种负载计数signature，不能据此证明动态调度收益。

cycle-v0与cycle-v1的主要差异来自用请求级Ramulator时序替换理想带宽估计，不能
把全部差值称为GPU/PIM竞争惩罚。当前选中放置的显式竞争delta较小。

主要结果和报告：

- `ramulator/timing_tables/generated/qwen3_full_decode_cycle_v1_contention.json`；
- `results/full_decode_cycle_v1/`；
- `results/full_decode_cycle_v1_fixed_split_ablation/`；
- `docs/full_decode_cycle_v1_analysis.md`；
- `docs/full_decode_cycle_v1_fixed_split_ablation.md`。

真实pilot中间结果：

- `results/router_trace_analysis/real_pilot/`；
- `ramulator/timing_tables/generated/qwen3_real_pilot_decode_cycle_v0.json`及其
  evidence；
- `results/full_decode_real_pilot_cycle_v0/`。

## 当前阶段

MoE专家调度的真实pilot、runtime-v1在线策略验证和runtime-v2
signature-level holdout均已完成。`sieve-cycle-v1`在当前阶段冻结为离线Oracle；
当前研究**单GPU设备上的KV Cache容量与延迟瓶颈实验**，目标仍限定为当前的
`1 GPU + 8 Local HBM-PIM Stacks`模拟硬件，不扩展Shared CXL-PIM或多GPU。

已经完成：

- Router捕获器，模型执行只提供路由决策，不导入实机时延；
- manifest schema-v2，绑定Trace、实际prompt快照、请求ID和SHA-256；
- Router特征分析器，输出负载CV/Gini、热点占比、逐专家负载、step间Jaccard、
  load churn和逐请求路由变化；
- Batch 8、8 Decode step的pilot prompts和cycle-v0/cycle-v1配置；
- Qwen模型固定提交
  `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`；
- 在单张`NVIDIA A800-SXM4-80GB`云实例上完成BF16模型加载和真实pilot捕获；
- 真实pilot产生384个层批次、3072条Router记录，并通过manifest完整性校验；
- 真实pilot路由分析完成，得到340个unique load signatures；
- 真实cycle-v0表已按8个非均匀`context_lengths` tuple生成schema-v2，并完成384个
  layer-batch的5策略回放；
- 真实cycle-v1已精确补齐1030/1030个workload shape，插值为0，并生成包含14,527条
  entry的schema-v3 contention table及evidence；
- 真实cycle-v1七策略（含`sieve-fixed-16-cycle-v1`固定分界消融）均完成384个
  layer-batch回放，输入哈希和timing entry完整性校验通过；
- 真实pilot cycle-v1分析报告为`docs/full_decode_real_pilot_cycle_v1_analysis.md`。
- runtime-v1阶段已完成：`sieve-runtime-v1`、cycle-v1 Oracle、legacy `sieve`和
  `sieve-fixed-16-cycle-v1`均完成384个layer-batch回放；runtime校准使用15个非零
  anchor点，运行时决策不读取精确候选表且本阶段新增Ramulator运行数为0；
  结果与评价见`docs/full_decode_real_pilot_runtime_v1_analysis.md`。
- runtime-v2 signature-level holdout已完成：按340个unique load signature进行5-fold
  分组留出，比较5/9/15/25个非零校准点；每个预算覆盖384个留出layer-batch，所有
  选定放置均未外推。25点预算达到380/384 prefix一致，平均regret
  `0.00005092 us`、最大`0.00488832 us`；9/15点为377/384，5点为376/384。候选
  空间中每个fold有1个GPU计数超出训练范围，但没有被runtime选中。结果见
  `results/full_decode_real_pilot_runtime_v2_holdout/`和
  `docs/full_decode_real_pilot_runtime_v2_holdout_analysis.md`。
- 已单独测量Python版runtime决策开销（10轮warmup、100轮、38,400次决策）：中位数
  `370.164 us`、P95 `471.678 us`、P99 `503.041 us`，约为配置20 us scheduler
  假设的18.5倍。该测量包含解释器和对象分配开销，不等同于生产C++ scheduler，
  因此没有直接改写30.551 ms基线；部署型runtime-v2必须在目标运行时测量并重新
  回放端到端时延。敏感性结果见
  `results/full_decode_real_pilot_runtime_v2_holdout/runtime_scheduler_benchmark.json`。
- 云端捕获环境固定为PyTorch `2.7.1+cu126`、Transformers `4.53.2`、Python
  `3.12.11`，驱动报告CUDA `13.2`。

KV Cache 长 Context 阶段已完成首轮五组独立 A800 Router 捕获、精确 workload 填充、
cycle-v0 isolated Attention 表、cycle-v1 contention 表和七策略回放。五组配置为
B8/C4k、B8/C8k、B8/C16k、B16/C4k、B16/C8k；每组 8 Decode step × 48 层，分别
包含 3072、3072、3072、6144、6144 条 Router 记录。342 个 union cycle-v1 shape
全部精确完成，failed=0、remaining=0、interpolation=0。五张 isolated 表的
Attention/Expert entry 数分别为 8/22、8/49、8/21、8/40、8/76；contention 表
entry 数分别为 3754、3888、3717、3855、3934。正式回放与汇总位于
`results/full_decode_real_kv_cycle_v1/` 和 `docs/kv_long_context_cycle_v1_analysis.md`。

上述冻结回放不包含 KV READ request-level 竞争、paging、spill 或 eviction。旧 pilot 的
Batch=8 短 Context 峰值 KV Cache 约 18.68 MB，不能作为 KV 瓶颈证据；长 Context 回放
使用真实 A800 路由但 GPU Attention/GPU arithmetic 仍为解析模型，PIM Attention 使用
isolated cycle-v0 分段时序，Expert 使用 cycle-v1 contention。A800 物理容量为 80 GB，
模拟容量为 96 GB，报告严格分开两者；本轮五组真实配置均未超过估算容量。

本轮长 Context 基线已依次完成：

1. 以旧pilot为基线，补齐Attention、Expert、KV大小、峰值内存和吞吐量分解；
2. 在本地做受控的Batch×Context解析扫描，保持专家路由不变，首轮矩阵为B8的
   `512/2k/4k/8k/16k/32k`、B16的`4k/8k/16k/32k`和B32的`4k/8k/16k`；
3. 根据扫描结果，在A800上捕获新的真实长Context Router Trace，优先B8/C4k、
   B8/C8k、B8/C16k、B16/C4k和B16/C8k；
4. 每个新配置独立保存`prompts.jsonl`、`router.jsonl`和`manifest.json`，绑定
   SHA-256，不能复制旧Trace后修改Context；
5. 对新Trace重新规划并精确补齐Attention及Expert workload timing，禁止插值；
6. 回放并输出KV大小、Attention时延占比、峰值内存、容量状态和总时延分解；

以上端到端回放与解析扫描只提供进一步建模的动机，未证明真实 KV 带宽瓶颈。
`results/kv_read_experiment_v1/frozen_baseline.json` 固定了 350 个基线文件的 SHA-256；
后续实验只能校验，不得刷新该清单以接受基线变化。冻结文档中的“当前/下一步”描述
对应该文档生成时的阶段，最新状态以本文件与新增报告为准。

容量准入模型已实现 `resident / spill / oom` 字节分类，输出为
`results/kv_capacity_states_v1/`，说明见 `docs/kv_capacity_states_v1.md`。正式实验分别
使用 A800 80 GB 与模拟 96 GB 两个容量域，各自 spill budget 均为 0，不把二者相加或
互相作为 spill 目的地。B8/C32k 与 B16/C16k 的峰值约 86.835 GB，在模拟容量内
`resident`，在 A800 容量下 `oom/infeasible`；B16/C32k 与 B32/C16k 约 112.606 GB，
两域均 `oom/infeasible`。这只是解析准入结果，不是实测 OOM，也没有实现分页、传输、
驱逐或恢复时延。

独立 KV READ 微基准使用新增 `SieveKVRead` frontend，KV 与 GPU Expert 普通 READ
共用 HBM 请求队列，分别记录注入、完成、驻留、最大驻留和拒绝注入次数。它不修改
旧 `SieveMixed` 接口或正式 cycle-v1 表。实验范围为 B8/C4k、B8/C8k、B8/C16k、
B16/C8k 各一个 step=0/layer=0，三种固定放置（GPU-only、旧 Expert-only Oracle、
fixed-half-prefix）与 expert-only/kv-only/combined 三种模式，36 个消费者去重为
19 个 workload shape。combined 是强制同时注入的受控请求级实验，端到端串行图
未变；不得将这些数值描述为端到端回放或真实 A800 带宽结论。具体请求模型见
`docs/kv_read_request_model_v1.md`，实验分析见 `docs/kv_read_experiment_v1_analysis.md`。

KV READ v1 已完成 19/19 精确 shape，failed=0、remaining=0、interpolation=0；
5 个 zero-KV shape 与旧 Expert-only 结果逐字段一致。控制器敏感性扫描随后完成
5 个变体 × 19 个 shape：READ buffer 为 256/128/64/32 的 dual-row，以及
READ buffer=256 的 single-row。95/95 完成、失败和插值均为 0。结果位于
`results/kv_read_sensitivity_v1/`，报告为 `docs/kv_read_sensitivity_v1_analysis.md`。

独立审计 `scripts/audit_kv_read_sensitivity.py` 校验当前计划、源码、95 个缓存哈希、
请求计数、原始 JSON/CSV，以及基准变体全部 19 个结果与 KV READ v1 的逐字段一致性。
`validation.json` 状态为 `passed_with_provenance_limitation`：历史敏感性运行没有
保存 binding/library 二进制哈希，不得将事后检查当作执行时证明。缓存目录包含两个
计划版本各 95 个文件，只按当前计划选取结果，不能把文件总数当作完成量或统计重复。

`matched_comparison.json/csv` 按同一变体的 expert-only/kv-only/combined 重建
60 组比较；全部 combined−max(isolated)>0，但增量及 Context 单调性依赖控制器
和放置。19 个去重 shape 的平均控制器差值不是 KV 平均竞争惩罚。single/dual 只改变
PIM/normal row-buffer 假设，并未改变 KV/Expert 地址映射；serial/staggered 注入和
地址布局扫描尚未实现。不能把结果直接纳入串行端到端回放或宣称真实 A800 带宽瓶颈。
独立 A800 Attention/Decode 计时与显存采样已经完成；原 Router hook 的
`.cpu().tolist()` 会引入同步，因此实测使用了不带 Router hook 的独立 pass。B8/C4k、
B8/C8k、B8/C16k、B16/C4k、B16/C8k 的结果位于
`results/a800_timing_memory_v1/`，说明见 `docs/a800_timing_memory_capture_v1.md`。

KV CXL Stage 4--6 已实现 memory-only CXL 请求模型，并将 Local/CXL KV Attention
阶段与后续 Expert 阶段按 Decode 依赖顺序接入 48 层事件图。Stage 6 的 B8/C32k
正式点使用 `request_scale=1`，包含 4 个精确 Ramulator 运行、192 个逐层结果，插值为
0；名义 CXL 假设为 64 GB/s、0.25 us、4 个 channel 和 64 GB 容量。这仍是固定参数
的 CXL FIFO 内存模型，不是物理 CXL 设备、paging 或 CXL-PIM。结果位于
`results/kv_cxl_phase_stage6_b8_c32k_formal_v1/`，分析见
`docs/kv_cxl_phase_stage6_analysis.md`。

Stage 7 已完成 A800 实测锚定的 B8/C32k 配对分析。B8/C4k+C8k 前向预测 C16k
Attention 的绝对百分比误差为 1.668%，三点线性拟合 R² 为 0.999920；C32k 数字仍是
外推，不是 A800 实测。新增一次 16,777,216 条 Local-HBM KV READ 的 full-count
精确 Ramulator 运行，并与 Stage 6 的 CXL spill 结果按哈希配对，插值为 0。名义
64 GB/s memory-only CXL 相对容量不受限的 Local-HBM 反事实增加 108.517 ms/Decode，
估算吞吐下降 13.003%；这说明它可以满足解析容量准入，但不能作为无代价扩容。
842.466 GB/s 只是由 spilled payload 和全 Local-HBM 完成期限导出的 payload-rate
门槛，不是物理 CXL 带宽需求或实测结果。结果位于
`results/kv_cxl_a800_bridge_stage7_v1/`，分析见
`docs/kv_cxl_a800_bridge_stage7_analysis.md`。下一阶段应先定义 CXL-PIM Attention 的
query/partial-result 数据流和请求计数，再与 memory-only CXL 做同一输入、同一容量
拆分的精确比较；不得直接把 Stage 7 外推数字当作 CXL-PIM 性能。

Stage 8 已完成首个分解式 CXL-PIM Attention 下界模型。它假设 4 个 CXL-PIM channel、
每 channel 2 个 PCH、每 PCH 24 个 bank，沿用当前 HBM3 时序与 PIM command interval；
这些都是模型假设，不是物理设备参数。B8/C32k 每层发送 65,536 B query，spilled KV
保留在 CXL-PIM 侧，并返回每 PCH、每请求、每 Attention head 一份 FP32
`(max, sum, value vector)` partial state，共 1,064,960 B。query link、PIM GWRITE、
PIM MAC、PIM READ 和 result link 为 5 个独立精确 Ramulator 运行，全部完成且插值为
0；五阶段在结果中按依赖顺序串联，但没有进入同一个 controller 竞争模拟。

相对 memory-only CXL 每层传输 142,390,528 B raw spilled KV，Stage 8 的 query 与
partial result 合计为 1,130,496 B，解析链路流量减少 99.206%。CXL-PIM 分支每层
2,304.426 us，其中 PIM MAC 为 2,278.270 us。与 A800 拟合的本地 GPU Attention
分支组合后，理想并行与完全串行两种调度界分别得到 573.311 ms 和 683.924 ms 的
估算 Decode 时延；相对 Stage 7 memory-only CXL 的估算吞吐改善分别为 45.567% 和
22.024%。这些数字依赖理想 PIM 吞吐、FP32 partial 格式、零 merge cost 等假设，
不是物理 CXL-PIM 性能，也不表示 Shared CXL-PIM 或多 GPU 已实现。结果位于
`results/kv_cxl_pim_attention_stage8_v1/`，分析见
`docs/kv_cxl_pim_attention_stage8_analysis.md`。下一阶段应扫描 CXL-PIM channel/PCH、
PIM MAC interval、partial 精度、merge cost 和 overlap，并在敏感性边界稳定后再考虑
统一 CXL-PIM controller 或 Shared CXL-PIM。

Stage 9 已完成 CXL-PIM 参数敏感性扫描。矩阵包含 channel=`2/4/8`、每 channel 2 个
PCH、MAC interval=`12288/24576/49152 ps`、partial scalar=`2/4 bytes`，形成 18 个
硬件/格式设计；每个设计再扫描 overlap=`0/0.5/1` 和 merge=`0/5/20 us/layer`，共
162 行。组件按有效模拟输入去重为 23 个，其中 5 个按身份和 SHA-256 复用 Stage 8，
18 个为新增精确 Ramulator 运行；failed=0、interpolation=0，独立验证通过。

162 行中 156 行优于 64 GB/s memory-only CXL，138 行优于容量不受限 Local-HBM
反事实；按每个设计的最差调度行判断，16/18 个设计稳健优于 memory-only CXL，12/18
稳健优于 Local-HBM 反事实。失效区集中在 2-channel 与最慢 49152 ps MAC：
`c2_mac49152_p4` 在 overlap=0、merge=20 us/layer 时为 1012.518 ms、7.901 token/s，
比 memory-only CXL 更差。Stage 8 名义点 `c4_mac24576_p4` 的最差扫描结果为
684.884 ms，仍优于两个对照。2-byte partial 只带来约 0.306 ms 的全矩阵平均时延
改善，且尚未验证数值误差，不能据此宣称低精度可用。

结果位于 `results/kv_cxl_pim_sensitivity_stage9_v1/`，分析见
`docs/kv_cxl_pim_sensitivity_stage9_analysis.md`。这仍是 A800 C32k 外推、隔离
Ramulator 组件和解析调度组合，不是物理 CXL-PIM 或统一 controller 结果。下一阶段应
选择名义点、2-channel 失效边界和 8-channel 稳健点做统一 CXL-PIM 请求级验证，并
独立验证 2-byte partial 的 Attention 数值误差；在此之前不扩展 Shared CXL-PIM。

Stage 10 已完成三个 Stage-9 边界点的统一请求级流水与 partial-state 数值敏感性。
新增 `SieveCXLPIMPipeline` 前端，在同一次 Ramulator simulation 中按完成依赖执行
query link、PIM GWRITE、PIM MAC、PIM READ 和 result link；link controller 与
CXL-PIM media controller 仍是独立资源，三个 PIM 阶段共享持续 controller 状态，
不得称为单一物理队列竞争。`c2_mac49152_p4`、`c4_mac24576_p4` 和
`c8_mac24576_p4` 三次 full-count 运行均精确完成，failed=0、interpolation=0。

三个设计的统一流水相对 Stage 9 隔离求和差值分别为 `-0.001872`、`-0.001872` 和
`-0.001560 us/layer`，只是周期取整量级，说明当前严格串行、分资源模型下隔离求和
没有遗漏明显阶段开销，不能解释为统一流水加速。完全串行 Decode 分别为 1011.558、
683.924 和 630.118 ms；2-channel 慢 MAC 负对照仍差于 memory-only CXL，名义点和
8-channel 点仍优于 memory-only CXL 与 Local-HBM 反事实。

partial 精度实验使用 head_dim=128、Context=`4k/16k/32k`、PCH=`4/8/16`、logit
std=`1/4`、8 个 seed 和 FP32/BF16/FP16，共 432 条确定性合成 trial。BF16 最大绝对
误差为 `0.021268`、最大相对 L2 为 `0.266684`、最低 cosine 为 `0.975323`；FP16
对应为 `0.003107`、`0.024038` 和 `0.999717`。这不是 A800 activation 或任务精度
结果；`partial_scalar_bytes=2` 只能作为流量参数，BF16/FP16 必须分开报告，正式
端到端点继续使用 FP32 partial。

结果位于 `results/kv_cxl_pim_pipeline_stage10_v1/`，分析见
`docs/kv_cxl_pim_pipeline_stage10_analysis.md`。下一阶段应把名义点和 8-channel 点的
统一 timing 接入 48 层 Decode 事件图，以 2-channel 慢 MAC 为负对照；真实 partial
精度验证未完成前不得用合成结果宣称 2-byte 格式可部署，也不扩展 Shared CXL-PIM。

Stage 11 已将三个 Stage-10 FP32 partial 设计接入 A800 校准的 48 层 MoE Decode
事件图，并保留 Router、Expert 和 Combine 节点。实验复用 Stage 6 的 full-count
Expert/Attention、Stage 7 的全 Local-HBM 对照和 Stage 10 的三条统一 pipeline，
共绑定 6 个精确缓存；新增 Ramulator 运行为 0，8 个场景行、384 个逐层结果均完整，
interpolation=0，独立验证通过。

事件图同时报告理想重叠和完全串行两个边界。理想重叠下三个设计均为 573.311 ms，
因为 A800 外推的本地 Attention 分支仍在关键路径；这只是调度上界，不是已实现策略。
完全串行下 `c2_mac49152_p4`、`c4_mac24576_p4` 和 `c8_mac24576_p4` 分别为
1011.558、683.924 和 630.118 ms。相对 834.554 ms 的 64 GB/s memory-only CXL，
2-channel 慢 MAC 负对照吞吐下降 17.498%，4-channel 名义点和 8-channel 稳健点吞吐
分别提高 22.024% 和 32.444%。

Stage 11 用 A800 B8 的 126.394 ms/Decode 非 Attention 均值校准事件图总量；其中
2598.504868 us/layer 是未归因的校准残差，不能把逐项 Router、Expert、Combine 数值
称为 A800 分项实测。B8/C32k Router 仍是 B8/C4k 模板的受控 tiling，Attention 仍是
C4k--C16k 的 A800 趋势外推。结果位于
`results/kv_cxl_pim_decode_stage11_v1/`，分析见
`docs/kv_cxl_pim_decode_stage11_analysis.md`。下一阶段应实现显式 chunk/tile 依赖、
partial readiness、双缓冲限制和 GPU merge 工作，以事件级调度替代理想 overlap 比例；
在该门槛完成前不扩展 Shared CXL-PIM 或多 GPU。

Stage 12 已完成显式 chunk/tile CXL-PIM Attention 流水。三个 Stage-10 边界设计分别
运行 1 chunk/1 slot、8 chunks/1 slot 和 8 chunks/2 slots，共 9/9 个 full-count
Ramulator workload；failed=0、remaining=0、interpolation=0，执行时 binding、library、
源码和输入哈希均已保存。三组单 chunk 结果在 query、GWRITE、MAC、READ、result-link
及最终完成的 10 个周期字段上逐项复现 Stage 10。

8-chunk 模型让每个 chunk 返回一份完整 FP32 partial state，因此 PIM READ 与结果链路
请求总量为单 chunk 的 8 倍。双缓冲相对单缓冲将 2/4/8-channel 三个设计的精确分支
时延分别降低 0.709%、5.238% 和 17.436%，buffer stall 均降为 0；但当前 48 层回放中
三条 CXL-PIM 分支都在不可抢占的本地 GPU Attention 结束前完成，所以相同 chunk 数的
单/双缓冲端到端时延相同。8 chunks 的额外 Decode 开销来自 7 次额外解析 merge kernel，
三个设计分别增加约 0.403、0.470 和 0.604 ms/Decode。

Stage 12 的 9 个显式调度点为 573.369--574.002 ms/Decode，相对 834.554 ms 的
memory-only CXL 吞吐提高 45.392%--45.553%。这依赖 RoPE 后两个 Attention 分支同时启动、
本地 GPU Attention 非抢占且 merge 后置的调度假设；2-channel 慢 MAC 负对照因此不再像
完全串行边界那样失败。GPU merge 使用当前模拟 GPU peak/HBM 参数的 roofline 加每 chunk
1 us kernel overhead，不是 A800 实测。结果位于
`results/kv_cxl_pim_chunk_stage12_v1/`，分析见
`docs/kv_cxl_pim_chunk_stage12_analysis.md`。下一阶段应将本地 GPU Attention 也拆成显式
tile，扫描 CXL-PIM 启动偏移、merge 批处理/优先级和 2/4/8 chunk，在调度结论稳定前不扩展
Shared CXL-PIM 或多 GPU。

物理A800显存为80 GB，当前模拟容量为96 GB，二者必须分开报告。Batch=16、
Context=32k和Batch=32、Context=16k可用于模拟容量压力，但不能直接称为A800实测
结果。若未实现spill，只能将超容量配置标记为`infeasible`，不能声称已经完成KV
paging性能建模。

真实pilot目录必须包含：

```text
prompts.jsonl
router.jsonl
manifest.json
```

云端pilot归档文件为`qwen3_router_pilot_8x8.tar.gz`，已生成的归档SHA-256为
`910d531a9e188f27cd2f3b3fac43180523d5e28b171b2ddd28aa399a332f4208`。当前工作树已
导入该pilot目录；manifest记录的`router.jsonl` SHA-256为
`a22fdc8afb9b93af6ea133ac254bac98830a5d07e4225edf2b37c348ea80bb34`，
`prompts.jsonl` SHA-256为
`50b874b743669f7686b35b58cf3227c3942a1c2a2ea9ee14bcdda0a606a46866`。模型权重不需要
复制或提交，真实Trace文件已纳入当前仓库版本控制。

真实Trace导入后的执行顺序为：

1. 通过manifest、Trace完整性和prompt身份校验；
2. 运行Router特征分析；
3. 构建`third_party/ramulator2/python`下的Sieve Ramulator Python binding；
4. 枚举并去重真实Trace所需的全部精确cycle-v1 workload；
5. 使用可恢复缓存补齐Ramulator运行，禁止插值；云服务器上的首次pilot填充因该
   环境未构建Ramulator Python binding而失败`1030`个shape，这属于环境前置条件
   错误，不是workload时序失败；本地已构建binding并完成1030个shape的精确缓存；
6. 已在`failed=0`、`remaining=0`且插值为0后，物化与真实Trace哈希绑定的schema-v3表；
7. 已回放七个策略和固定16专家消融；
8. 已完成动态放置、固定分界、cycle-v0/v1和isolated/contended结果比较。

详细协议见`docs/real_router_trace_stage.md`。

## 科研结论边界

现有结果是模拟器阶段性结果，不是论文级硬件性能结论。后续工作必须保留以下边界：

- 当前合成Trace性能结果只用于验证和对照；
- 真实Qwen3 pilot已产生cycle-v0和cycle-v1的模拟性能数字；cycle-v1数字仅在
  1030/1030精确workload、无插值和输入哈希校验通过的范围内有效；
- 当前真实pilot固定为8个Decode step，不能代表更长序列或连续批处理；合成验证仍为2个step；
- GPU算术、Router、Scheduler和GPU Attention仍是解析模型；
- PIM Attention尚未进入cycle-v1混合请求模拟；
- GPU READ是抽象顺序请求流，不是真实GPU访存Trace；
- 当前未建模刷新、能耗、原生逐Bank PIM命令、多GPU、NVLink和Shared CXL-PIM；
- 容量可行只表示估算字节数低于96 GB，不代表真实布局和驻留已经实现；
- 不得将当前数字描述为真实B200性能或Sieve/PIMoE论文的完整复现。
- runtime-v1的`98.9583%` prefix一致率、`0.019553 us`端到端Oracle gap和预测误差
  只适用于当前同一真实pilot、模型、硬件和观测计数范围；15个校准点来自已有精确
  cycle-v1表，是独立硬件微基准的代理，不是跨workload验证。尚未进行新prompt、
  Batch、Context或模型的留出测试，禁止宣称泛化。
- 在线策略在真实部署中仍需承担实际调度开销；本阶段没有把离线校准时间计入请求
  延迟，也没有声称校准点可被提前知道。当前模拟器仍用精确cycle-v1表计算选定放置
  的真值时延；这只是评价机制，不是runtime决策依赖。
- 当前30.551 ms runtime基线使用配置的20 us scheduler模型参数；Python决策开销基准
  仅作为敏感性证据，不能直接替代目标运行时scheduler时间。若将370.164 us机械
  替换到384个layer-batch，scheduler部分约为142.14 ms，但这不是生产系统性能结论。
- signature-level holdout仍来自同一个真实pilot，只能说明对未参与训练的已有负载
  形状具有初步留出能力；它不能替代新prompt、不同Batch/Context或独立硬件微基准
  的验证。在这些留出实验完成前，不得宣称runtime-v1具有跨workload泛化能力。

## 后续修改规则

- 每次只围绕用户明确任务修改，不提前实现Shared CXL-PIM或未确定的硬件参数；
- 优先复用现有配置、Trace、timing provider、policy和report接口；
- 科研参数必须注明来源或明确标记为模型假设；
- 正式cycle-v1表只允许精确Ramulator结果，禁止插值和哈希不匹配的缓存复用；
- 真实Trace必须有manifest，合成验证不得放入`traces/real/`；
- 生成结果必须保存输入哈希、配置快照、时延分类和限制说明；
- runtime校准必须保存anchor、方法、来源哈希和适用范围；评价脚本必须校验策略、
  layer-batch数量、timing backend和输入哈希；
- `.cache`、Ramulator构建树、模型权重和认证信息不得提交；
- 不修改或撤销用户已有的无关改动；
- 修改后按风险补充测试，完整验证命令为：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts tests
git diff --check
```
