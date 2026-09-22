# KV Decode stage 1: explicit local-HBM KV READ

本阶段把 KV READ 接入完整 Attention → Router → Expert → Combine 回放。旧的五组
cycle-v1 基线没有被覆盖；每个新配置都在输出目录保存了完整 JSON snapshot。

KV READ 使用 `analytic-local-hbm-v1`，Expert 使用已有精确 cycle-v1 contention 表。
因此本阶段验证端到端事件依赖和 accounting，不宣称 KV 请求级 Ramulator 或真实
A800 带宽结果。

## 代表性 sieve-cycle-v1 结果

| 配置 | 基线 us | serial us | overlap us | serial delta | overlap delta | KV bytes |
|---|---:|---:|---:|---:|---:|---:|
| b8_c4k | 49435.350 | 49435.350 | 46211.372 | +0.000 | -3223.978 | 25,791,823,872 |
| b8_c8k | 75262.999 | 75262.999 | 68817.796 | +0.000 | -6445.203 | 51,561,627,648 |
| b8_c16k | 126744.500 | 126744.500 | 113856.846 | +0.000 | -12887.654 | 103,101,235,200 |
| b16_c4k | 76697.748 | 76697.748 | 70249.792 | +0.000 | -6447.956 | 51,583,647,744 |
| b16_c8k | 128294.471 | 128294.471 | 115404.064 | +0.000 | -12890.407 | 103,123,255,296 |

串行模式的总时延应与旧基线保持一致，因为它只是把原 Attention 时延拆为 KV READ
和剩余 Attention 计算。Overlap 模式允许两者在 RoPE 后并行，再由 o_proj 等待两者
完成；这是假设性执行语义，用于下一阶段请求级模型的接口验证。

冻结基线 inventory SHA-256: `07873a163dd41e86faaf4f002844cc7c078858906be696c3016db29cc0ee255b`
