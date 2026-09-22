# KV CXL timing stage 3

阶段 3 把阶段 2 的容量准入结果接入完整的解析 Decode 事件图。Local HBM KV READ 使用 `HBM_MEM_PATH`，CXL spill KV READ 使用独立的 `CXL_MEM_PATH`。`serial` 模式要求两条读流完成后才能开始 Attention compute；`overlap` 模式让两条读流和 Attention compute 在 RoPE 后并行，并在 `o_proj` 汇合。

实验使用 schema-v2 manifest 校验的真实 pilot 第一层批次作为受控路由模板，分别扩展到 `B8/C32k`、`B16/C16k`、`B16/C32k` 和 `B32/C16k`，每组重复完整 48 层、1 个 Decode step。它不是新的 A800 捕获，也没有复制或修改真实 Router Trace。

由于尚未选定具体 CXL 设备，带宽/延迟 profile 明确标记为模型假设：

| profile | bandwidth | latency |
|---|---:|---:|
| conservative-assumption | 32 GB/s | 0.5 us |
| nominal-assumption | 64 GB/s | 0.25 us |
| optimistic-assumption | 128 GB/s | 0.1 us |

阶段 3 的 `oom` 行不会被回放；`spill` 行只表示静态 KV 字节被放到 CXL 并按 KV 总量比例分摊到每层。当前没有 CXL 链路协议、分页、eviction、恢复、CXL-PIM、Ramulator CXL 模型或 A800 timing。

运行入口为 `scripts/run_kv_cxl_timing_stage3.py`，正式结果写入 `results/kv_cxl_timing_stage3_v1/`。只有当结果显示 CXL READ 进入关键路径并成为主要延迟来源，下一阶段才进入 CXL-PIM KV 操作卸载建模。
