# KV CXL-PIM Stage 9 参数敏感性分析

## 1. 研究问题

Stage 8 只评价了一个 CXL-PIM 参数点。本阶段固定同一个 B8/C32k 容量拆分和
A800 Attention 拟合输入，扫描 CXL-PIM 拓扑、PIM MAC 间隔、partial state
精度、GPU/CXL-PIM 分支重叠度和逐层 merge 开销，用于判断 Stage 8 的收益是否只
依赖单个乐观参数点。

本阶段仍是分解式模型：query link、PIM GWRITE、PIM MAC、PIM READ 和 result
link 分别进行精确 Ramulator 运行，再按 Decode 依赖解析组合。它不是物理
CXL-PIM 实测，也不是统一 controller 竞争模拟。

## 2. 实验矩阵

硬件和 partial 格式共有 18 个设计点：

- CXL-PIM channel：2、4、8；
- 每 channel 固定 2 个 pseudo-channel，每 PCH 固定 24 个 bank；
- PIM MAC interval：12,288、24,576、49,152 ps；
- partial scalar：2、4 bytes。

每个设计点再组合 9 个调度条件：

- GPU/CXL-PIM Attention 分支 overlap：0、0.5、1；
- merge：0、5、20 us/layer。

因此正式结果包含 162 行。每行按下式组合：

```text
attention_ms =
    local_gpu_attention_ms
  + cxl_pim_branch_ms
  - overlap * min(local_gpu_attention_ms, cxl_pim_branch_ms)
  + merge_us_per_layer * 48 / 1000
```

固定对照为：

| 对照 | Decode 时延 | 吞吐量 |
|---|---:|---:|
| 64 GB/s memory-only CXL | 834.554 ms | 9.586 token/s |
| 容量不受限 Local-HBM 反事实 | 726.037 ms | 11.019 token/s |

两个对照都来自 Stage 8 绑定的 Stage 7 输入。B8/C32k 本地 GPU Attention 分支仍是
A800 B8/C4k--C16k 趋势外推，不是 A800 C32k 实测。

## 3. 精确性与来源

18 个设计点的组件按有效模拟输入去重为 23 个：1 个 query link、4 个 result
link、3 个 GWRITE、9 个 MAC 和 6 个 READ。Stage 8 的 5 个组件按 simulation
identity 和 SHA-256 复用，本阶段新增 18 个精确 Ramulator 运行。

验证结果为：

- 18/18 设计点和 162/162 调度行完成；
- 23/23 组件哈希和请求完成数通过校验；
- Stage 8 理想重叠与完全串行结果均精确复现；
- failed=0，interpolation=0；
- Ramulator revision：`b30320bc9385b708e86b67ebb9f48858cc66d798`；
- binding SHA-256：`d798c022f722177d7c184f2e0f5630c787f6b927632b9868e57b3473348d1547`；
- library SHA-256：`3d2eb8a91aaef784bd15bfe74c42b809b636b2916461ddcd9cf0d9595d1e6a23`。

## 4. 主要结果

162 行中有 156 行优于 memory-only CXL，138 行优于容量不受限 Local-HBM 反事实。
按每个设计的最差调度条件判断，16/18 个设计稳健优于 memory-only CXL，12/18 个
设计稳健优于 Local-HBM 反事实。

| 因素 | 参数值 | 跨其余因素平均 Decode 时延 |
|---|---:|---:|
| channel | 2 | 701.624 ms |
| channel | 4 | 637.996 ms |
| channel | 8 | 606.428 ms |
| MAC interval | 12,288 ps | 606.155 ms |
| MAC interval | 24,576 ps | 638.051 ms |
| MAC interval | 49,152 ps | 701.842 ms |
| overlap | 0 | 723.654 ms |
| overlap | 0.5 | 648.683 ms |
| overlap | 1 | 573.711 ms |

最差点为 `c2_mac49152_p4`、overlap=0、merge=20 us/layer：Decode 为
1012.518 ms，吞吐量为 7.901 token/s。它比 memory-only CXL 的 Decode 时延高
21.324%，吞吐量低 17.576%。这说明低通道数、慢 MAC 和串行调度同时出现时，
CXL-PIM 不再优于直接搬运 KV。

不稳健优于 memory-only CXL 的两个设计均为 2-channel、49,152 ps MAC；不稳健优于
Local-HBM 反事实的六个设计为 2-channel 的 24,576/49,152 ps，以及 4-channel 的
49,152 ps，且都包括两种 partial 格式。8-channel 的六个设计在完整调度矩阵内均优于
两个对照。

Stage 8 名义设计 `c4_mac24576_p4` 在 overlap 为 0、0.5、1 且 merge=0 时，Decode
分别为 683.924、628.618、573.311 ms。即使取该设计的最差组合
overlap=0、merge=20 us/layer，Decode 仍为 684.884 ms，低于两个对照。

2-byte partial 相对 4-byte partial 的全矩阵平均 Decode 时延只降低 0.306 ms。
partial 精度在当前固定 64 GB/s 链路和 MAC 主导模型中不是主要性能因素；但 2-byte
partial 的数值误差尚未验证，因此本结果不能支持降低精度的正确性结论。

完全 overlap 时所有 CXL-PIM 分支都短于固定的本地 GPU Attention 分支，所以硬件参数
差异被关键路径遮蔽，只剩 merge 开销。该结果是调度上界，不是已实现的并行执行。

## 5. 结论边界

本阶段支持的结论是：在当前 B8/C32k、固定 64 GB/s FIFO 链路和分解式模型中，
CXL-PIM 相对 memory-only CXL 的优势覆盖大部分扫描范围，但并非无条件成立；2-channel
与最慢 MAC 的组合在弱 overlap 下会失去收益。channel、MAC 吞吐和实际 overlap 是
后续架构必须约束的参数。

本阶段不能证明：

- 物理 CXL-PIM 能达到这些时延；
- query、PIM 命令和 result transfer 在共享 controller 下仍保持隔离时序；
- 2-byte partial 满足 Attention 数值精度要求；
- B8/C32k 是 A800 实测结果；
- paging、eviction、恢复、CXL 协议、能耗、多 GPU 或 Shared CXL-PIM 已实现。

## 6. 下一阶段

下一阶段应进入统一 CXL-PIM 请求级验证，先选择三个边界点：Stage 8 名义点
`c4_mac24576_p4`、失效边界 `c2_mac49152_p4` 和稳健点 `c8_mac24576_p4`。需要在同一
controller/事件模型中表达 query 写入、PIM GWRITE/MAC/READ、partial result 返回及
阶段依赖，并与本阶段隔离结果配对比较。与此同时，应对 2-byte partial 做 Attention
输出误差验证，并用可实现的事件依赖替代自由 overlap 参数。完成这两项后，才适合把
CXL-PIM 接入完整 MoE Decode 回放并进一步研究 Shared CXL-PIM 和多 GPU 共享。
