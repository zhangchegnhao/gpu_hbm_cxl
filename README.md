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

输出目录包含 `router.jsonl` 和 `manifest.json`。真实 Trace 实验配置必须提供
`trace_manifest`；回放时会校验 Trace SHA-256、模型 shape、batch 和 decode step。
仓库不会自动下载约 60 GB 的 BF16 模型权重。

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
