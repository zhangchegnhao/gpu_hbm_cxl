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

## 当前阶段

当前阶段是**真实Qwen3 Router Trace驱动的动态cycle-v1实验**。真实pilot的路由捕获
已经在外部云实例完成，但真实Trace的Ramulator时序和六策略性能回放尚未完成。

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
- 云端捕获环境固定为PyTorch `2.7.1+cu126`、Transformers `4.53.2`、Python
  `3.12.11`，驱动报告CUDA `13.2`。

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
   错误，不是workload时序失败；当前本地工作树已有Ramulator环境，但真实Trace
   workload填充尚未在本地重试；
6. 仅在`failed=0`、`remaining=0`且插值为0后，物化与真实Trace哈希绑定的schema-v3表；
7. 回放六个主要策略和固定16专家消融；
8. 比较动态放置、固定分界、cycle-v0/v1和isolated/contended结果。

详细协议见`docs/real_router_trace_stage.md`。

## 科研结论边界

现有结果是模拟器阶段性结果，不是论文级硬件性能结论。后续工作必须保留以下边界：

- 当前已完成性能结果使用确定性合成Router Trace；
- 真实Qwen3 pilot目前只提供路由Trace和路由统计，尚未产生真实Trace驱动的性能数字；
- 只有2个Decode step，不能代表长序列或连续批处理；
- GPU算术、Router、Scheduler和GPU Attention仍是解析模型；
- PIM Attention尚未进入cycle-v1混合请求模拟；
- GPU READ是抽象顺序请求流，不是真实GPU访存Trace；
- 当前未建模刷新、能耗、原生逐Bank PIM命令、多GPU、NVLink和Shared CXL-PIM；
- 容量可行只表示估算字节数低于96 GB，不代表真实布局和驻留已经实现；
- 不得将当前数字描述为真实B200性能或Sieve/PIMoE论文的完整复现。

## 后续修改规则

- 每次只围绕用户明确任务修改，不提前实现Shared CXL-PIM或未确定的硬件参数；
- 优先复用现有配置、Trace、timing provider、policy和report接口；
- 科研参数必须注明来源或明确标记为模型假设；
- 正式cycle-v1表只允许精确Ramulator结果，禁止插值和哈希不匹配的缓存复用；
- 真实Trace必须有manifest，合成验证不得放入`traces/real/`；
- 生成结果必须保存输入哈希、配置快照、时延分类和限制说明；
- `.cache`、Ramulator构建树、模型权重和认证信息不得提交；
- 不修改或撤销用户已有的无关改动；
- 修改后按风险补充测试，完整验证命令为：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts tests
git diff --check
```
