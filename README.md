# Sieve-compatible MoE Decode Trace Replay

本仓库用于构建单 GPU、Qwen3-30B-A3B、Local HBM-PIM 架构下的端到端
Decode 模拟基线。回放器支持单层回归，以及按 `step -> layer` 串联的多层、多
Decode step 工作负载；每层均经过 Attention、Router、GPU/PIM Expert FFN 和
Combine，并输出事件时间线与关键路径。

当前提供三个后端：`analytic-v0` 用于快速功能检查；
`ramulator-table-v0` 使用固定版 Ramulator 2.1 执行独立 PIM 命令微基准，再将
严格时延表回放到端到端 DAG；`ramulator-contention-v1` 将普通GPU专家权重READ
与PIM波次放进同一次Ramulator运行，加入共享命令仲裁和逻辑双Row Buffer。
GPU计算仍是解析模型，因此三者都不是论文级性能结论。

## 当前范围

- 单 GPU，8 个本地 HBM3E-PIM stacks；
- Qwen3-30B-A3B 单层回归和 48 层、多 Decode step 回放；
- `gpu-only`、`noexp`、`allexp`、`pimoe`、`sieve` 五个基线，以及
  `sieve-cycle-v1` 竞争时延感知策略；
- 确定性合成 Router Trace，以及带严格 manifest 的真实 Router Trace 接口；
- 事件依赖、资源互斥、GPU/PIM 并行和关键路径计算；
- 256 PCH 并行的 `PIM_GWRITE`、`PIM_MAC`、`PIM_READ` 周期微基准；
- 普通GPU专家权重READ与PIM命令的同控制器竞争；
- 双Row Buffer的PIM侧行状态、ACT/PRE延迟和单Row Buffer消融开关；
- Ramulator 时延表生成、输入哈希和原始周期证据；
- JSON/CSV 结构化结果。

暂不包含多 GPU、NVLink 和 Shared CXL-PIM。真实 Router Trace 采集代码已经提供，
但仓库不包含 Qwen3 权重或正式数据集 Trace。`cycle-v1` 仍不包含刷新、能耗、
原生逐bank PIM ACT/PRE、GPU算术周期模拟和真实GPU访存 trace，不能称为完整
Sieve 复现。

其中 `pimoe` 当前是配置化静态 token 阈值占位实现，并非论文基线的最终复现，
不能用于正式对比。

## 快速运行

项目只依赖 Python 3.10+ 标准库。

```bash
PYTHONPATH=src python3 -m sieve_replay.cli run \
  --experiment configs/experiments/single_layer_smoke.json \
  --policy sieve \
  --output results/single_layer_smoke/sieve
```

运行全部策略并生成 `comparison.csv`：

```bash
PYTHONPATH=src python3 -m sieve_replay.cli run-all \
  --experiment configs/experiments/single_layer_smoke.json \
  --output results/single_layer_smoke
```

运行测试：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## 48 层 Decode replay

重新生成确定性的 48 层、2 step 合成 Router Trace：

```bash
python3 scripts/generate_synthetic_decode_trace.py \
  --output traces/synthetic/decode_48layers_2steps.jsonl
```

运行五策略解析回放：

```bash
PYTHONPATH=src python3 -m sieve_replay.cli run-all \
  --experiment configs/experiments/full_decode_synthetic.json \
  --output results/full_decode_synthetic
```

`summary.json` 另外包含每 step 平均/P50/P95 时延和请求 token 吞吐率；多层输出新增
`layers.csv` 和 `steps.csv`。`events.csv`、`placement.csv` 的每一行都带 step/layer。

生成完整工作负载的 cycle-v0 表并运行五策略：

```bash
PYTHONPATH=src python3 scripts/generate_ramulator_table.py \
  --experiment configs/experiments/full_decode_cycle_v0.json \
  --cycle-config configs/ramulator/sieve_hbm3e_cycle_v0.json \
  --output ramulator/timing_tables/generated/qwen3_full_decode_cycle_v0.json

PYTHONPATH=src python3 -m sieve_replay.cli run-all \
  --experiment configs/experiments/full_decode_cycle_v0.json \
  --output results/full_decode_cycle_v0
```

## 真实 Router Trace

采集依赖与模拟器核心依赖分离，不应把模型运行时间写入模拟结果：

```bash
python3 -m pip install -r requirements/trace_capture.txt

PYTHONPATH=src python3 scripts/capture_qwen3_router_trace.py \
  --model Qwen/Qwen3-30B-A3B \
  --revision <immutable-commit> \
  --prompts traces/real/prompts.example.jsonl \
  --output traces/real/qwen3_30b_a3b/<dataset> \
  --decode-steps 16 \
  --dataset <dataset-name> \
  --dataset-revision <dataset-revision> \
  --dataset-split <split>
```

输出目录包含 `prompts.jsonl`、`router.jsonl` 和 `manifest.json`。schema-v2会绑定
实际使用的prompt快照和Trace SHA-256、模型shape、batch、decode step及固定贪心
生成语义。真实Trace实验配置必须提供`trace_manifest`。
仓库不会自动下载约 60 GB 的 BF16 模型权重。

捕获后先验证完整性并分析路由动态性：

```bash
PYTHONPATH=src python3 scripts/analyze_router_trace.py \
  --experiment configs/experiments/<real-trace-experiment>.json \
  --output results/router_trace_analysis/<capture-name>
```

分析输出包括逐层负载、逐专家负载和相邻Decode step的热点重合与load churn。
当前真实Trace阶段协议、pilot prompts和本机执行限制见
`docs/real_router_trace_stage.md`。
Pilot捕获固定使用Qwen官方提交
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`，不跟随`main`漂移。

## 多层 cycle-v1 workload cache

先生成目录，不运行 Ramulator：

```bash
PYTHONPATH=src python3 scripts/plan_decode_workloads.py \
  --experiment configs/experiments/full_decode_cycle_v1.json \
  --cycle-config configs/ramulator/sieve_hbm3e_cycle_v1.json \
  --cache-dir ramulator/timing_tables/generated/.cache/decode_workloads \
  --output /tmp/full_decode_workloads.json
```

提供正数上限后只补齐对应数量的缺失 shape，可重复执行并从缓存恢复：

```bash
PYTHONPATH=src python3 scripts/plan_decode_workloads.py \
  --experiment configs/experiments/full_decode_cycle_v1.json \
  --cycle-config configs/ramulator/sieve_hbm3e_cycle_v1.json \
  --cache-dir ramulator/timing_tables/generated/.cache/decode_workloads \
  --output /tmp/full_decode_workloads.json \
  --max-new-runs 20
```

需要补齐大量shape时，可以使用独立进程并行运行；父进程为每个完成shape立即写入
同一精确缓存，`--workers`应按宿主机实际CPU和内存能力设置：

```bash
PYTHONPATH=src python3 scripts/fill_decode_workloads_parallel.py \
  --experiment configs/experiments/full_decode_cycle_v1.json \
  --cycle-config configs/ramulator/sieve_hbm3e_cycle_v1.json \
  --cache-dir ramulator/timing_tables/generated/.cache/decode_workloads \
  --catalog-output /tmp/full_decode_workloads.json \
  --workers 4
```

当 `missing_workload_shapes` 为 0 时，物化严格的 schema-v3 contention table：

```bash
PYTHONPATH=src python3 scripts/build_decode_contention_table.py \
  --experiment configs/experiments/full_decode_cycle_v1.json \
  --cycle-config configs/ramulator/sieve_hbm3e_cycle_v1.json \
  --cache-dir ramulator/timing_tables/generated/.cache/decode_workloads \
  --output ramulator/timing_tables/generated/qwen3_full_decode_cycle_v1_contention.json

PYTHONPATH=src python3 -m sieve_replay.cli run-all \
  --experiment configs/experiments/full_decode_cycle_v1.json \
  --output results/full_decode_cycle_v1
```

物化器要求每个混合及隔离 shape 均精确命中；缺少任何键都会失败，不插值，也不会
留下不完整的正式表。当前仓库不跟踪 `.cache`，因此新克隆必须重新补齐或显式传入
已有缓存。

当前提供的48层合成Trace已经完成146/146个精确workload、schema-v3表和六策略
cycle-v1回放。`sieve-cycle-v1`选择每层16个GPU专家和33个PIM专家，总时延为
10.144665 ms；完整的cycle-v0/v1差异、路径平衡和竞争指标解释见
`docs/full_decode_cycle_v1_analysis.md`。这些数字仍受合成Trace和解析GPU模型限制，
不能作为真实B200或论文级性能结论。

固定16专家热点前缀消融与动态搜索得到完全相同的10.144665 ms，证明当前合成
Trace只有一种负载计数shape，不能评价动态放置收益。消融见
`docs/full_decode_cycle_v1_fixed_split_ablation.md`，真实Router Trace阶段协议见
`docs/real_router_trace_stage.md`。

## Ramulator cycle-v0

构建固定依赖和项目 overlay：

```bash
scripts/build_sieve_ramulator.sh
```

生成 Qwen3 当前 Router Trace 所需的 50 个专家 token-count 条目和一个 Attention
条目。生成器只运行三次最长微基准，并从周期里程碑提取所有较小 shape：

```bash
PYTHONPATH=src python3 scripts/generate_ramulator_table.py \
  --experiment configs/experiments/single_layer_smoke.json \
  --cycle-config configs/ramulator/sieve_hbm3e_cycle_v0.json \
  --output ramulator/timing_tables/generated/qwen3_single_layer_cycle_v0.json
```

将时延表接入完整单层 replay：

```bash
PYTHONPATH=src python3 -m sieve_replay.cli run-all \
  --experiment configs/experiments/single_layer_cycle_v0.json \
  --output results/single_layer_cycle_v0
```

每次运行生成：

- `summary.json`：总时延、分项时延、放置结果和输入哈希；
- `events.csv`：每个事件的资源、依赖、起止时间；
- `placement.csv`：每个活跃专家的 token 数和执行位置；
- `run_manifest.json`：本次运行的完整配置快照。

时延表旁的 `*.evidence.json` 另外记录 wave 数、Ramulator cycles、注入请求数和
完成请求数。查询不插值；缺少准确的 batch/context 或 token-count 时直接失败。

## Ramulator cycle-v1

生成五个基线和新搜索策略所需的严格竞争表：

```bash
PYTHONPATH=src python3 scripts/generate_contention_table.py \
  --experiment configs/experiments/single_layer_cycle_v1.json \
  --cycle-config configs/ramulator/sieve_hbm3e_cycle_v1.json \
  --output ramulator/timing_tables/generated/qwen3_single_layer_cycle_v1_contention.json
```

执行六策略端到端回放：

```bash
PYTHONPATH=src python3 -m sieve_replay.cli run-all \
  --experiment configs/experiments/single_layer_cycle_v1.json \
  --output results/single_layer_cycle_v1
```

除原有文件外，输出根目录包含 `contention.csv`，记录隔离/竞争时延delta、阻塞
周期、命令数和PIM Row Buffer统计。cycle-v1的完整语义和假设见
`ramulator/extensions/sieve_hbm_pim/docs/cycle_v1_design.md`。

`sieve-cycle-v1` 保留原 `sieve` 作为旧版对照，按当前Router负载将专家从热到冷
排序，在51个热点前缀组成的空间中使用单调交叉二分选择候选；当前实际测量10个
前缀。每个被搜索的候选都执行严格
GPU/PIM混合Ramulator运行；最终候选额外执行GPU和PIM隔离运行。搜索直接最小化：

```text
scheduler + max(GPU mixed READ + analytic GPU compute, PIM mixed path)
```

生成表记录候选空间、已测前缀、最终前缀、已测点单调性验证和全部输入哈希。端到端
`summary.json` 的 `placement_search` 字段记录每个候选的路径时延和选择依据。当前
策略目录中的 `placement_search.csv` 提供相同候选的平面表。当前hotspot trace
选择16个GPU专家和34个PIM专家，单层时延为107.181024 us。该结果
只是在热点前缀约束下搜索，并不是全部 `2^50` 放置的穷举全局最优。

## Runtime-v1在线策略

`sieve-runtime-v1`将`sieve-cycle-v1`冻结为离线Oracle，仅使用少量分段线性校准
点预测GPU READ和PIM pipeline时延；在线决策不读取精确放置候选表，也不运行
Ramulator。先生成校准文件，再执行四策略回放和Oracle评价：

```bash
PYTHONPATH=src python3 scripts/build_runtime_calibration.py \
  --experiment configs/experiments/full_decode_real_pilot_cycle_v1.json \
  --output ramulator/timing_tables/generated/qwen3_real_pilot_runtime_calibration_v1.json \
  --evidence-output ramulator/timing_tables/generated/qwen3_real_pilot_runtime_calibration_v1.evidence.json

PYTHONPATH=src python3 -m sieve_replay.cli run-all \
  --experiment configs/experiments/full_decode_real_pilot_runtime_v1.json \
  --output results/full_decode_real_pilot_runtime_v1

PYTHONPATH=src python3 scripts/evaluate_runtime_scheduler.py \
  --experiment configs/experiments/full_decode_real_pilot_runtime_v1.json \
  --results results/full_decode_real_pilot_runtime_v1 \
  --output results/full_decode_real_pilot_runtime_v1/runtime_evaluation
```

该阶段的同pilot结果和适用边界见
`docs/full_decode_real_pilot_runtime_v1_analysis.md`；图表由
`scripts/plot_runtime_v1_results.py`生成。当前结果不代表跨workload泛化。

### Runtime-v2 signature holdout

对340个unique load signature做5-fold分组留出，校准只使用训练signature，验证
部分仍用精确cycle-v1表做事后Oracle评价。运行：

```bash
PYTHONPATH=src python3 scripts/evaluate_runtime_holdout.py \
  --experiment configs/experiments/full_decode_real_pilot_cycle_v1.json \
  --output results/full_decode_real_pilot_runtime_v2_holdout \
  --folds 5 --budgets 5,9,15,25
```

25个非零校准点达到`380/384` prefix一致率，最大regret`0.00488832 us`；该结果
仍只适用于同一pilot的signature-level留出，不能替代新prompt、Batch、Context或
硬件的泛化验证。详见`docs/full_decode_real_pilot_runtime_v2_holdout_analysis.md`。

另外测得Python实现的runtime决策中位数为`370.164 us`（P95 `471.678 us`），高于
仿真使用的20 us scheduler参数。该数字包含解释器开销，仅用于敏感性分析，未直接
替换基线总时延；部署型实现需在目标运行时重新测量。

## 目录

```text
configs/
  experiments/             # 单层和完整Decode的解析版/cycle-v0/cycle-v1入口
  hardware/                # 端到端硬件参数
  models/                  # Qwen3 官方模型 shape
  ramulator/               # Ramulator 拓扑和命令速率
ramulator/
  extensions/sieve_hbm_pim/# 上游补丁、PIM frontend/controller
  timing_tables/           # Schema、生成说明和生成产物
scripts/                   # 获取、构建和时延表生成
src/sieve_replay/
  capture/                 # Qwen3真实Router Trace采集
  policy/                  # 五个基线和cycle-v1感知放置策略
  ramulator/               # wave 映射和微基准驱动
  simulation/              # 单层 DAG 与离散事件调度
  timing/                  # analytic/isolated/contention timing provider
tests/
```

## 参数来源与限制

Qwen3 配置来自 Qwen 官方 ModelScope 仓库的 `config.json` 快照；架构交叉核对
Qwen3 官方技术报告。报告给出 128 个专家、每 token 激活 8 个。Sieve 论文正文将
实验中的 Qwen3 描述为 128 个专家中激活 4 个，两者不一致。本项目默认保持官方
模型语义 `8/128`，不会静默覆盖为 4；若复现 Sieve 论文特定设置，应建立单独的
实验配置并明确标注模型变体。

Sieve 硬件参数取自论文 Table 1。`cycle-v0` 使用 8 stacks、32 PCH/stack、
24 banks/PCH、96 GB、8 TB/s 和 1 op/byte。一个全 PCH MAC wave 表示
`256 x 24 x 32 = 196608` operations；命令间隔由 8 TOPS 推导，而不是实测值。

`analytic-v0` 中的效率系数和固定开销属于当前模型假设。即使使用
`ramulator-table-v0`，GPU kernel、Router、Scheduler 等时延仍来自这些解析参数，
不能作为实机 B200 测量值引用。

Ramulator 2.1 固定 revision 为
`b30320bc9385b708e86b67ebb9f48858cc66d798`。扩展的已实现行为与缺失机制见
`ramulator/extensions/sieve_hbm_pim/README.md`。
