# KV CXL-PIM pipeline Stage 10

Stage 10 runs three Stage-9 boundary designs through one dependency-gated Ramulator simulation per design. Link controllers and CXL-PIM media controllers remain separate resources in one event timeline.

| design | isolated us/layer | unified us/layer | delta | serialized Decode ms |
|---|---:|---:|---:|---:|
| c2_mac49152_p4 | 9130.148 | 9130.146 | -0.002 | 1011.558 |
| c4_mac24576_p4 | 2304.426 | 2304.425 | -0.002 | 683.924 |
| c8_mac24576_p4 | 1183.475 | 1183.473 | -0.002 | 630.118 |

Exact unified runs: `3`; interpolation: `0`.

The precision sweep uses deterministic synthetic logits and values, not captured A800 activations. It reports FP32, BF16 and FP16 partial-state merge errors without imposing an unsupported accuracy threshold.

| PCH | format | trials | max abs error | max relative L2 | min cosine |
|---:|---|---:|---:|---:|---:|
| 4 | fp32 | 48 | 3.749254e-07 | 4.392689e-06 | 1.000000000 |
| 4 | bf16 | 48 | 1.801138e-02 | 2.666841e-01 | 0.975323244 |
| 4 | fp16 | 48 | 3.018372e-03 | 2.403783e-02 | 0.999716980 |
| 8 | fp32 | 48 | 3.806477e-07 | 2.019122e-06 | 1.000000000 |
| 8 | bf16 | 48 | 2.118045e-02 | 1.385093e-01 | 0.991342268 |
| 8 | fp16 | 48 | 2.205545e-03 | 1.824904e-02 | 0.999857398 |
| 16 | fp32 | 48 | 4.102652e-07 | 1.287302e-06 | 1.000000000 |
| 16 | bf16 | 48 | 2.126783e-02 | 1.261258e-01 | 0.994532370 |
| 16 | fp16 | 48 | 3.107290e-03 | 1.100208e-02 | 0.999940439 |

This remains a single-GPU analytic/Ramulator study. It does not implement paging, physical CXL protocol, energy, multi-GPU arbitration or Shared CXL-PIM.
