# KV CXL-PIM Stage 10 统一流水与 Partial 精度分析

## 1. 研究问题

Stage 9 使用五个隔离 Ramulator 组件相加得到 CXL-PIM Attention 分支时延。本阶段回答
两个问题：

1. 将 query link、PIM GWRITE、PIM MAC、PIM READ 和 result link 放进同一次、按完成
   依赖推进的请求级 simulation 后，隔离求和是否漏掉明显的阶段衔接开销；
2. Stage 9 中只按字节数表示的 2-byte partial state，在 BF16 或 FP16 编码下会产生
   多大的数值误差。

## 2. Stage 10A：统一请求级流水

新增 `SieveCXLPIMPipeline` 前端。每个设计只运行一次 Ramulator simulation，并严格
执行：

```text
query link 完成
-> PIM GWRITE 完成
-> PIM MAC 完成
-> PIM READ 完成
-> result link 完成
```

CXL FIFO link controller 和 CXL-PIM media controller 是同一事件时间线中的独立资源。
三个 PIM 阶段使用同一组持续存在的 `SieveHBMPIM` controller。该模型可以验证阶段
依赖、controller 状态延续和请求计数，但不表示链路请求与 PIM 命令进入同一个物理
请求队列。

选择 Stage 9 的三个边界点：

| 设计 | 用途 | channel | MAC interval | partial |
|---|---|---:|---:|---:|
| `c2_mac49152_p4` | 失效边界 | 2 | 49,152 ps | FP32/4 bytes |
| `c4_mac24576_p4` | Stage 8 名义点 | 4 | 24,576 ps | FP32/4 bytes |
| `c8_mac24576_p4` | 稳健点 | 8 | 24,576 ps | FP32/4 bytes |

三次运行均为 B8/C32k full-count workload。3/3 精确完成，failed=0、interpolation=0。
前端与 controller 的 query、GWRITE、MAC、READ 和 result 请求数逐项一致，阶段完成周期
严格单调。Ramulator revision 为
`b30320bc9385b708e86b67ebb9f48858cc66d798`，本阶段 library SHA-256 为
`4babe0a5dfe87044c6e35430b00d165a2e46fa313a98626d5e53831128aa93f4`。

## 3. 流水结果

| 设计 | 隔离求和 | 统一流水 | 差值 | 串行 Decode | 串行吞吐 |
|---|---:|---:|---:|---:|---:|
| `c2_mac49152_p4` | 9130.147728 us/layer | 9130.145856 us/layer | -0.001872 us | 1011.558 ms | 7.909 token/s |
| `c4_mac24576_p4` | 2304.426384 us/layer | 2304.424512 us/layer | -0.001872 us | 683.924 ms | 11.697 token/s |
| `c8_mac24576_p4` | 1183.474968 us/layer | 1183.473408 us/layer | -0.001560 us | 630.118 ms | 12.696 token/s |

统一流水与隔离求和的比例为 `0.99999868--0.99999979`。负差只有约 1.6--1.9 ns/layer，
小于一个数量级上有意义的体系结构开销，应解释为离散周期边界和阶段起点取整差异，
不能描述成统一流水带来的性能收益。

结果说明：在当前严格串行依赖、link 与 PIM media 分资源的模型中，Stage 9 的隔离求和
没有遗漏明显的阶段衔接开销。它不说明存在或不存在多请求、多 GPU 或共享链路竞争，
因为这些流量没有进入本阶段实验。

失效边界仍慢于 834.554 ms memory-only CXL 和 726.037 ms Local-HBM 反事实。名义点和
8-channel 稳健点在完全串行的本地 GPU/CXL-PIM 分支组合下仍优于两个对照。三点的
理想 overlap Decode 都是 573.311 ms，因为固定的本地 GPU Attention 分支仍是关键
路径；该值仍是调度上界，不代表已经实现完全 overlap。

## 4. Stage 10B：Partial State 数值敏感性

数值实验使用 Qwen3 的 `head_dim=128`，扫描：

- Context：4k、16k、32k；
- total PCH：4、8、16；
- logit 标准差：1.0、4.0；
- 8 个固定 seed；
- partial scalar：FP32、BF16、FP16。

共生成 432 条 trial。每条 trial 将 Context 位置条带化分配到 PCH，分别计算稳定在线
softmax partial `(max, exp sum, weighted value)`，量化后再合并。value 使用四个确定性
基函数投影到 128 维。直接双精度在线 softmax 是参考，FP32 partial 是传输精度控制组。

| 格式 | 最大绝对误差 | 最大相对 L2 误差 | 最低 cosine similarity |
|---|---:|---:|---:|
| FP32 | 4.103e-7 | 4.393e-6 | 0.999999999993 |
| BF16 | 2.127e-2 | 2.667e-1 | 0.975323244 |
| FP16 | 3.107e-3 | 2.404e-2 | 0.999716980 |

BF16 的最大相对 L2 case 为 C32k、4 PCH、logit std=1、seed=3；相对误差为 26.668%，
cosine 为 0.978411。最大绝对误差来自 C16k、16 PCH、logit std=4、seed=4，为
0.021268。FP16 在相同合成矩阵中的误差明显小于 BF16，这是因为该数值范围内 FP16
提供更多尾数位；本实验没有覆盖 FP16 溢出风险或真实模型 activation 分布。

因此 Stage 9 的 `partial_scalar_bytes=2` 只能解释为流量大小，不能同时代表 BF16 和
FP16 的数值行为。当前证据不支持直接采用 BF16 partial，也不能仅凭合成 trial 宣称
FP16 满足模型精度要求。

## 5. 结论边界

本阶段支持：

- 三个边界点的五阶段统一流水请求数、依赖顺序和精确时序已验证；
- 在当前分资源且严格串行的模型中，隔离求和与统一运行一致；
- 2-byte partial 的具体编码是必须单独验证的架构参数。

本阶段不支持：

- 将结果称为物理 CXL-PIM、A800 C32k 或 Shared CXL-PIM 实测；
- 声称链路与 PIM media 存在统一物理队列竞争；
- 声称 BF16 或 FP16 partial 已通过真实 Qwen 精度验证；
- 声称 paging、eviction、恢复、CXL 协议、能耗或多 GPU 仲裁已实现。

## 6. 下一阶段

下一阶段应把 `c4_mac24576_p4` 和 `c8_mac24576_p4` 的统一 pipeline timing 接入现有
48 层 Decode 事件图，替换 Stage 6 的 memory-only CXL Attention 分支，并保留
`c2_mac49152_p4` 作为负对照。回放必须继续区分理想 overlap 与可实现串行依赖，使用
相同 B8/C32k 容量拆分和 Expert timing，输出完整 Attention/Router/Expert/Combine
分解。

数值侧需要在真实 Qwen Attention activation 上重复 partial merge，对 BF16/FP16
分别报告输出误差和任务级质量。在获得真实 activation 证据前，端到端性能回放应继续
使用 FP32 partial 的 4-byte 正式点，2-byte 只保留为流量敏感性。
