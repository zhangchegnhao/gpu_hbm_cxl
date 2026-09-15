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
下一阶段转向**单GPU设备上的KV Cache容量与延迟瓶颈实验**，目标仍限定为当前的
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

KV Cache瓶颈阶段尚未开始正式回放，目前处于实验设计阶段。当前旧pilot的Batch为8、
Context很短，峰值KV Cache约18.68 MB，不能作为KV瓶颈证据。现有实现只提供KV容量
估算和解析Attention时延，尚未建模KV READ请求、KV与Expert的HBM请求竞争、paging、
spill或eviction。

下一阶段执行顺序固定为：

1. 以旧pilot为基线，补齐Attention、Expert、KV大小、峰值内存和吞吐量分解；
2. 在本地做受控的Batch×Context解析扫描，保持专家路由不变，首轮矩阵为B8的
   `512/2k/4k/8k/16k/32k`、B16的`4k/8k/16k/32k`和B32的`4k/8k/16k`；
3. 根据扫描结果，在A800上捕获新的真实长Context Router Trace，优先B8/C4k、
   B8/C8k、B8/C16k、B16/C4k和B16/C8k；
4. 每个新配置独立保存`prompts.jsonl`、`router.jsonl`和`manifest.json`，绑定
   SHA-256，不能复制旧Trace后修改Context；
5. 对新Trace重新规划并精确补齐Attention及Expert workload timing，禁止插值；
6. 回放并输出KV大小、Attention时延占比、峰值内存、容量状态和总时延分解；
7. 只有在证据表明KV确实形成瓶颈后，才扩展KV请求级Ramulator竞争或容量溢出模型。

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
