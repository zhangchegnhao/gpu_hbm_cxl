# KV CXL request-level Stage 4

本报告是 Ramulator request-level memory-only CXL 敏感性实验。Local HBM controller
继续承载 Expert READ、PIM 命令和 resident KV READ；CXL controller 只承载静态 spilled KV READ。
三个带宽/时延 profile 是明确的模型假设，不是 A800 或真实 CXL 设备测量。
本轮 request_scale=1；scale=1 使用完整生成请求量，其他值是统一缩放的微基准。

| case | profile | mode | CXL READ (M) | total (us) | CXL queue wait (us) | link busy |
|---|---|---|---:|---:|---:|---:|
| b32_c16k | nominal-assumption | cxl-kv-only | 21.228328 | 11590.915440 | 11868534.128256 | 0.999979 |

所有行都由精确 request-level simulation 生成，interpolation=0。这里的 CXL READ
数量是每个压力点一个 Decode layer 的受控请求量；结果不能直接解释为完整 48 层
端到端时延，下一步才接入 Decode event graph。
