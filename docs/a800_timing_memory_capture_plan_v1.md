# A800 独立 timing/memory 捕获计划 v1

当前仓库只提供 plan-only 入口：

```bash
PYTHONPATH=src python3 scripts/plan_a800_capture.py
```

默认输出 `results/a800_capture_plan_v1/plan.json`。该命令只核验已经导入的五组
真实 Router 输入并写出协议，`execution_implemented=false`，不会加载模型权重、申请
CUDA 或修改任何旧 Trace。五组输入为 B8/C4k、B8/C8k、B8/C16k、B16/C4k 和 B16/C8k；
每组的 prompts、router、manifest 路径和 SHA-256 都独立记录。

## 采样协议

未来实现采样器时，每个配置必须写入独立目录：

```text
prompts.jsonl
router.jsonl
timing.jsonl
memory.jsonl
manifest.json
```

Router 必须先用现有 capture 入口重新捕获，保持相同 prompt 快照、Qwen3 固定 revision、
BF16、greedy argmax 和 8 个 Decode step。不能复制旧 Router JSONL 再修改 Context。随后
用相同 prompt 重新做 timing pass，模型放在唯一可见的 `cuda:0`，显式单卡，不允许
`device_map=auto` 或 CPU offload。

Timing pass 要求三个可区分的测量范围：

1. 无 instrumentation 的 Decode step 基线 CUDA events；
2. 仅在 `self_attn` 模块边界插入 CUDA events 的 Attention pass；
3. 显存统计 pass，每次 repeat 清零 peak counters。

每个 repeat 都重新做相同 Context 的 prefill 以重建 KV cache；warmup 不纳入结果。每个
采样行必须带 repeat、step、context lengths 和测量范围。Attention 模块事件包含 Python
和模块 launch 间隙，不能称为 kernel-only 时间。Timing pass 不得安装 Router hook，避免
原 Router hook 中 `.cpu().tolist()` 的同步污染计时。接受结果前需要比较 timing pass
产生的 greedy token 与 Router pass 的 token 轨迹；不一致时整组标记失败并丢弃 timing。

Manifest 必须绑定五个输出文件的大小和 SHA-256，并记录设备名称、驱动/CUDA、PyTorch、
Transformers、模型 revision、dtype、batch、Context、Decode steps、repeat/warmup、
`router_timing_isolated=true` 及上述限制。A800 物理容量固定按 80 GB 报告，与模拟 96 GB
容量分开。

## 开始 B8/C4k 试测前的核验

在 A800 上开始第一个配置前，先确认 `plan.json` 五组输入哈希通过；唯一可见 GPU 为
`NVIDIA A800-SXM4-80GB`，显存至少 80 GB，驱动和 CUDA 与环境记录一致；模型 revision
可无网络从本地缓存解析；输出目录为空且具有足够磁盘空间。当前尚缺真正执行器：它需要
实现无 hook 基线、Attention 事件 instrumentation、greedy token 对齐、每 repeat
prefill 重建、显存 peak 采样、manifest 写入和 fail-closed 校验。完成这些代码并通过
小模型单元测试后，才可在云端运行 B8/C4k；本仓库本轮没有构造或执行假的捕获命令。

该计划只描述 A800 真实硬件采样，不改变既有解析敏感性分析、Router 输入或 Ramulator
请求级 microbenchmark 的结论边界。
