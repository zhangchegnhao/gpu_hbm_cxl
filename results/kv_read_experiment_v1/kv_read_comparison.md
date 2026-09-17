# KV READ 代表性请求级竞争实验

每组仅使用已独立捕获真实 Router 的 step=0、layer=0。36 个配置消费者共享去重后的精确 Ramulator workload；本报告不覆盖完整 384 个 layer-batch。

Expert-only、KV-only 与 combined 使用同一请求模型和固定放置；combined 强制同时注入 KV 与 Expert 请求。当前端到端图仍按 Attention → Router → Expert 串行，因此下面是受控竞争微基准，不能直接作为端到端时延或真实 A800 带宽结论。

| 配置 | 放置 | GPU/PIM experts | Expert-only (us) | KV-only (us) | Combined (us) | Combined − max(isolated) (us) |
|---|---|---:|---:|---:|---:|---:|
| b16_c8k | gpu-only | 8/0 | 23.288616 | 84.240936 | 107.276208 | 23.035272 |
| b16_c8k | frozen-oracle | 7/1 | 24.661104 | 84.240936 | 103.918152 | 19.677216 |
| b16_c8k | fixed-half-prefix | 4/4 | 98.535528 | 84.240936 | 98.559552 | 0.024024 |
| b8_c16k | gpu-only | 8/0 | 23.288616 | 84.240936 | 107.276208 | 23.035272 |
| b8_c16k | frozen-oracle | 7/1 | 20.298408 | 84.240936 | 104.280384 | 20.039448 |
| b8_c16k | fixed-half-prefix | 4/4 | 49.285080 | 84.240936 | 95.292600 | 11.051664 |
| b8_c4k | gpu-only | 8/0 | 23.288616 | 20.662824 | 43.688112 | 20.399496 |
| b8_c4k | frozen-oracle | 7/1 | 20.298408 | 20.662824 | 40.710072 | 20.047248 |
| b8_c4k | fixed-half-prefix | 4/4 | 49.285080 | 20.662824 | 49.301928 | 0.016848 |
| b8_c8k | gpu-only | 8/0 | 23.288616 | 41.860104 | 64.851696 | 22.991592 |
| b8_c8k | frozen-oracle | 7/1 | 20.298408 | 41.860104 | 61.883016 | 20.022912 |
| b8_c8k | fixed-half-prefix | 4/4 | 49.285080 | 41.860104 | 53.350752 | 4.065672 |

请求注入/完成计数、控制器 READ 完成数、请求驻留时间、接收到列命令发出前的周期以及注入被拒次数均保存在 JSON/CSV。接收到列命令前的周期包含仲裁及 ACT/PRE 准备，不能解释为纯 FIFO 排队。

固定半数放置按 (-token_count, expert_id) 排序，前 ceil(active/2) 个专家分给 GPU；本轮代表性层均为 4 GPU + 4 PIM。它是独立的典型混合放置，并非旧 fixed16 消融。Oracle 放置继承旧 Expert-only 回放，没有按新 KV 竞争重新优化。

A800 提供真实路由输入，不提供本报告时延；模拟范围仍为 1 GPU + 8 Local HBM-PIM stacks。没有新增 paging、spill、eviction 或 Shared CXL-PIM。
