# KV CXL capacity stage 2

本阶段把前一阶段的 KV 容量压力点扩展为一个 **memory-only CXL 容量准入敏感性分析**。它回答的是：当 Local HBM 容量不足时，显式增加一块 CXL 容量预算，哪些配置从 `oom` 变为 `spill` 或 `resident`。

实验仍然限定为 `1 GPU + 8 Local HBM-PIM stacks`。A800 的 80 GB 物理显存和模拟硬件的 96 GB 容量是两个独立的本地容量域；它们不相加，也不会把 A800 显存当成模拟容量。CXL budget 是额外容量假设，不能当作已部署的 CXL 设备参数。

压力点为 `B8/C32k`、`B16/C16k`、`B16/C32k` 和 `B32/C16k`，扫描的 CXL budget 为 `0/16/32/64 GB`（十进制）。`resident` 表示估算峰值完全放入本地容量，`spill` 表示超出本地容量的字节数可以被显式 budget 接纳，`oom` 表示本地容量加该 budget 仍不足。

运行入口为 `scripts/run_kv_cxl_capacity_stage2.py`。结果目录 `results/kv_cxl_capacity_stage2_v1/` 保存 JSON、CSV 和带输入哈希的阶段报告。Router Trace 只通过 schema-v2 manifest 校验后作为受控路由模板用于字节估算；没有复制或修改真实 Trace。

该阶段不包含 CXL 链路带宽、CXL-PIM 命令、KV READ 请求级竞争、分页、spill 传输、eviction、恢复时延、A800 实测显存或实测 OOM。因此 `spill` 是容量准入状态，不能直接解释为端到端性能收益。下一阶段只有在 CXL 容量准入边界明确后，才为选定的 resident/spill 点加入显式数据搬运模型。
