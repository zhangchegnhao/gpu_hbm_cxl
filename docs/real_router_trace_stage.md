# 真实Router Trace阶段

## 阶段目标

本阶段用Qwen3-30B-A3B真实模型执行产生的Decode Router决策替换确定性合成
Trace，回答竞争感知放置能否随层和Decode step的负载变化动态调整GPU/PIM分界。
模型执行只提供路由决策，不提供任何硬件时延；所有性能数字仍由模拟器产生。

## 当前执行状态

仓库内已经完成：

- manifest schema-v2：捕获目录同时保存实际prompt快照、SHA-256和生成语义；
- 旧schema-v1兼容读取；
- 完整性检查：模型shape、batch、step、context、Trace和prompt哈希；
- Router特征分析：负载CV/Gini、热点占比、逐专家负载、step间热点重合、
  load churn和逐请求路由变化；
- 8条固定pilot prompts；
- 合成Trace上的分析管线验证。

当前主机没有可用GPU、没有Qwen3权重、没有Trace捕获依赖，系统内存约15 GiB。
Qwen3-30B-A3B的BF16权重约60 GB，因此本机不能完成真实捕获。这里不使用随机路由、
小模型路由或人工数据冒充真实Qwen3 Trace。

## Pilot捕获协议

第一轮固定为Batch 8、8个Decode step、48层、Top-8，使用
`traces/real/prompts.pilot.jsonl`。在具有足够显存或CPU内存的外部环境运行：

模型固定为Hugging Face `Qwen/Qwen3-30B-A3B`提交
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`。2026-08-27读取该提交的官方
`config.json`，确认BF16、48层、hidden size 2048、MoE intermediate size 768、
128专家、Top-8和`norm_topk_prob=true`均与项目模型配置一致。

```bash
python3 -m pip install -r requirements/trace_capture.txt

PYTHONPATH=src python3 scripts/capture_qwen3_router_trace.py \
  --model Qwen/Qwen3-30B-A3B \
  --revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  --prompts traces/real/prompts.pilot.jsonl \
  --output traces/real/qwen3_30b_a3b/pilot_8x8 \
  --decode-steps 8 \
  --dataset project-pilot-prompts \
  --dataset-revision v1 \
  --dataset-split pilot \
  --seed 0
```

必须完整带回捕获目录中的三个文件：

```text
prompts.jsonl
router.jsonl
manifest.json
```

捕获器使用greedy argmax并保持固定batch，即使产生EOS也继续执行到固定step数。
这避免不同请求提前退出后改变batch形状。正式实验应在pilot通过后扩展到至少32个
Decode step和多个固定prompt组。

## 导入后的实验流程

真实Trace进入仓库后，仓库中的
`configs/experiments/full_decode_real_pilot_cycle_v1.json`会直接引用该捕获目录，
然后执行：

```bash
PYTHONPATH=src python3 scripts/analyze_router_trace.py \
  --experiment configs/experiments/full_decode_real_pilot_cycle_v1.json \
  --output results/router_trace_analysis/real_pilot

PYTHONPATH=src python3 scripts/plan_decode_workloads.py \
  --experiment configs/experiments/full_decode_real_pilot_cycle_v1.json \
  --cycle-config configs/ramulator/sieve_hbm3e_cycle_v1.json \
  --cache-dir ramulator/timing_tables/generated/.cache/qwen3_real_pilot_cycle_v1 \
  --output /tmp/qwen3_real_pilot_workloads.json

PYTHONPATH=src python3 scripts/fill_decode_workloads_parallel.py \
  --experiment configs/experiments/full_decode_real_pilot_cycle_v1.json \
  --cycle-config configs/ramulator/sieve_hbm3e_cycle_v1.json \
  --cache-dir ramulator/timing_tables/generated/.cache/qwen3_real_pilot_cycle_v1 \
  --catalog-output /tmp/qwen3_real_pilot_workloads.json \
  --workers 4
```

缓存达到`missing_workload_shapes=0`后才能物化schema-v3表并进行六个主要策略加
固定16专家消融的回放。禁止插值，也不能复用哈希不匹配的合成Trace正式表。

## 合成管线验证结果

`results/router_trace_analysis/synthetic_validation`只用于验证分析器，不是真实路由
实验。当前合成Trace的96个层批次具有：

- 49个活跃专家/层，64次专家分配/层；
- 唯一负载计数signature为1，主signature覆盖100%层批次；
- 负载Gini为0.700684，最热8个专家承载35.9375%的分配；
- 相邻step平均活跃专家Jaccard为0.020833，平均逐请求路由Jaccard为0.016667；
- 虽然专家ID变化很大，但全部48个step转换的负载计数signature均相同。

这解释了上一阶段为何所有层都选择相同的16/33数量拆分：Trace改变了专家身份，
但没有改变决定GPU/PIM路径平衡的负载计数形状。

固定16专家与动态搜索的正式消融也得到完全相同的10.144665 ms，详见
`docs/full_decode_cycle_v1_fixed_split_ablation.md`。该结果进一步确认真实Trace是
继续评价动态策略的必要输入。

## Pilot验收条件

- manifest为schema-v2且Trace、prompt哈希全部通过；
- 48层乘8 step共384个层批次，记录数为3072；
- workload cache缺失为0、插值为0；
- 六个主要策略和固定16专家消融全部完成；
- 报告动态GPU/PIM分界、固定16/33消融以及isolated/contended差异；
- 若放置仍不变化，必须由真实负载signature和候选路径数据解释，而不能预设结论。
