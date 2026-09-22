# KV CXL request-level Stage 4

本报告是 Ramulator request-level memory-only CXL 敏感性实验。Local HBM controller
继续承载 Expert READ、PIM 命令和 resident KV READ；CXL controller 只承载静态 spilled KV READ。
三个带宽/时延 profile 是明确的模型假设，不是 A800 或真实 CXL 设备测量。
本轮 request_scale=4096；scale=1 使用完整生成请求量，其他值是统一缩放的微基准。

| case | profile | mode | CXL READ (M) | total (us) | CXL queue wait (us) | link busy |
|---|---|---|---:|---:|---:|---:|
| b8_c32k | conservative-assumption | expert-only | 0.000000 | 0.059904 | 0.000000 | 0.000000 |
| b8_c32k | conservative-assumption | local-kv-only | 0.000000 | 0.087048 | 0.000000 | 0.000000 |
| b8_c32k | conservative-assumption | cxl-kv-only | 0.001087 | 1.599624 | 574.116192 | 0.689048 |
| b8_c32k | conservative-assumption | local-kv-plus-expert | 0.000000 | 0.084552 | 0.000000 | 0.000000 |
| b8_c32k | conservative-assumption | cxl-kv-plus-local-kv | 0.001087 | 1.599624 | 574.116192 | 0.689048 |
| b8_c32k | conservative-assumption | cxl-kv-plus-local-kv-plus-expert | 0.001087 | 1.599624 | 574.116192 | 0.689048 |
| b8_c32k | nominal-assumption | expert-only | 0.000000 | 0.059904 | 0.000000 | 0.000000 |
| b8_c32k | nominal-assumption | local-kv-only | 0.000000 | 0.087048 | 0.000000 | 0.000000 |
| b8_c32k | nominal-assumption | cxl-kv-only | 0.001087 | 0.842400 | 298.850136 | 0.704537 |
| b8_c32k | nominal-assumption | local-kv-plus-expert | 0.000000 | 0.084552 | 0.000000 | 0.000000 |
| b8_c32k | nominal-assumption | cxl-kv-plus-local-kv | 0.001087 | 0.842400 | 298.850136 | 0.704537 |
| b8_c32k | nominal-assumption | cxl-kv-plus-local-kv-plus-expert | 0.001087 | 0.842400 | 298.850136 | 0.704537 |
| b8_c32k | optimistic-assumption | expert-only | 0.000000 | 0.059904 | 0.000000 | 0.000000 |
| b8_c32k | optimistic-assumption | local-kv-only | 0.000000 | 0.087048 | 0.000000 | 0.000000 |
| b8_c32k | optimistic-assumption | cxl-kv-only | 0.001087 | 0.438672 | 161.114928 | 0.773115 |
| b8_c32k | optimistic-assumption | local-kv-plus-expert | 0.000000 | 0.084552 | 0.000000 | 0.000000 |
| b8_c32k | optimistic-assumption | cxl-kv-plus-local-kv | 0.001087 | 0.438672 | 161.114928 | 0.773115 |
| b8_c32k | optimistic-assumption | cxl-kv-plus-local-kv-plus-expert | 0.001087 | 0.438672 | 161.114928 | 0.773115 |
| b16_c16k | conservative-assumption | expert-only | 0.000000 | 0.061152 | 0.000000 | 0.000000 |
| b16_c16k | conservative-assumption | local-kv-only | 0.000000 | 0.087048 | 0.000000 | 0.000000 |
| b16_c16k | conservative-assumption | cxl-kv-only | 0.001087 | 1.599624 | 574.116192 | 0.689048 |
| b16_c16k | conservative-assumption | local-kv-plus-expert | 0.000000 | 0.084552 | 0.000000 | 0.000000 |
| b16_c16k | conservative-assumption | cxl-kv-plus-local-kv | 0.001087 | 1.599624 | 574.116192 | 0.689048 |
| b16_c16k | conservative-assumption | cxl-kv-plus-local-kv-plus-expert | 0.001087 | 1.599624 | 574.116192 | 0.689048 |
| b16_c16k | nominal-assumption | expert-only | 0.000000 | 0.061152 | 0.000000 | 0.000000 |
| b16_c16k | nominal-assumption | local-kv-only | 0.000000 | 0.087048 | 0.000000 | 0.000000 |
| b16_c16k | nominal-assumption | cxl-kv-only | 0.001087 | 0.842400 | 298.850136 | 0.704537 |
| b16_c16k | nominal-assumption | local-kv-plus-expert | 0.000000 | 0.084552 | 0.000000 | 0.000000 |
| b16_c16k | nominal-assumption | cxl-kv-plus-local-kv | 0.001087 | 0.842400 | 298.850136 | 0.704537 |
| b16_c16k | nominal-assumption | cxl-kv-plus-local-kv-plus-expert | 0.001087 | 0.842400 | 298.850136 | 0.704537 |
| b16_c16k | optimistic-assumption | expert-only | 0.000000 | 0.061152 | 0.000000 | 0.000000 |
| b16_c16k | optimistic-assumption | local-kv-only | 0.000000 | 0.087048 | 0.000000 | 0.000000 |
| b16_c16k | optimistic-assumption | cxl-kv-only | 0.001087 | 0.438672 | 161.114928 | 0.773115 |
| b16_c16k | optimistic-assumption | local-kv-plus-expert | 0.000000 | 0.084552 | 0.000000 | 0.000000 |
| b16_c16k | optimistic-assumption | cxl-kv-plus-local-kv | 0.001087 | 0.438672 | 161.114928 | 0.773115 |
| b16_c16k | optimistic-assumption | cxl-kv-plus-local-kv-plus-expert | 0.001087 | 0.438672 | 161.114928 | 0.773115 |
| b16_c32k | conservative-assumption | expert-only | 0.000000 | 0.061152 | 0.000000 | 0.000000 |
| b16_c32k | conservative-assumption | local-kv-only | 0.000000 | 0.117000 | 0.000000 | 0.000000 |
| b16_c32k | conservative-assumption | cxl-kv-only | 0.005183 | 5.752968 | 4827.140448 | 0.913539 |
| b16_c32k | conservative-assumption | local-kv-plus-expert | 0.000000 | 0.084552 | 0.000000 | 0.000000 |
| b16_c32k | conservative-assumption | cxl-kv-plus-local-kv | 0.005183 | 5.752968 | 4827.140448 | 0.913539 |
| b16_c32k | conservative-assumption | cxl-kv-plus-local-kv-plus-expert | 0.005183 | 5.752968 | 4827.140448 | 0.913539 |
| b16_c32k | nominal-assumption | expert-only | 0.000000 | 0.061152 | 0.000000 | 0.000000 |
| b16_c32k | nominal-assumption | local-kv-only | 0.000000 | 0.117000 | 0.000000 | 0.000000 |
| b16_c32k | nominal-assumption | cxl-kv-only | 0.005183 | 3.078816 | 2588.866176 | 0.919158 |
| b16_c32k | nominal-assumption | local-kv-plus-expert | 0.000000 | 0.084552 | 0.000000 | 0.000000 |
| b16_c32k | nominal-assumption | cxl-kv-plus-local-kv | 0.005183 | 3.078816 | 2588.866176 | 0.919158 |
| b16_c32k | nominal-assumption | cxl-kv-plus-local-kv-plus-expert | 0.005183 | 3.078816 | 2588.866176 | 0.919158 |
| b16_c32k | optimistic-assumption | expert-only | 0.000000 | 0.061152 | 0.000000 | 0.000000 |
| b16_c32k | optimistic-assumption | local-kv-only | 0.000000 | 0.117000 | 0.000000 | 0.000000 |
| b16_c32k | optimistic-assumption | cxl-kv-only | 0.005183 | 1.716624 | 1468.777440 | 0.942021 |
| b16_c32k | optimistic-assumption | local-kv-plus-expert | 0.000000 | 0.084552 | 0.000000 | 0.000000 |
| b16_c32k | optimistic-assumption | cxl-kv-plus-local-kv | 0.005183 | 1.716624 | 1468.777440 | 0.942021 |
| b16_c32k | optimistic-assumption | cxl-kv-plus-local-kv-plus-expert | 0.005183 | 1.716624 | 1468.777440 | 0.942021 |
| b32_c16k | conservative-assumption | expert-only | 0.000000 | 0.088608 | 0.000000 | 0.000000 |
| b32_c16k | conservative-assumption | local-kv-only | 0.000000 | 0.117000 | 0.000000 | 0.000000 |
| b32_c16k | conservative-assumption | cxl-kv-only | 0.005183 | 5.752968 | 4827.140448 | 0.913539 |
| b32_c16k | conservative-assumption | local-kv-plus-expert | 0.000000 | 0.088920 | 0.000000 | 0.000000 |
| b32_c16k | conservative-assumption | cxl-kv-plus-local-kv | 0.005183 | 5.752968 | 4827.140448 | 0.913539 |
| b32_c16k | conservative-assumption | cxl-kv-plus-local-kv-plus-expert | 0.005183 | 5.752968 | 4827.140448 | 0.913539 |
| b32_c16k | nominal-assumption | expert-only | 0.000000 | 0.088608 | 0.000000 | 0.000000 |
| b32_c16k | nominal-assumption | local-kv-only | 0.000000 | 0.117000 | 0.000000 | 0.000000 |
| b32_c16k | nominal-assumption | cxl-kv-only | 0.005183 | 3.078816 | 2588.866176 | 0.919158 |
| b32_c16k | nominal-assumption | local-kv-plus-expert | 0.000000 | 0.088920 | 0.000000 | 0.000000 |
| b32_c16k | nominal-assumption | cxl-kv-plus-local-kv | 0.005183 | 3.078816 | 2588.866176 | 0.919158 |
| b32_c16k | nominal-assumption | cxl-kv-plus-local-kv-plus-expert | 0.005183 | 3.078816 | 2588.866176 | 0.919158 |
| b32_c16k | optimistic-assumption | expert-only | 0.000000 | 0.088608 | 0.000000 | 0.000000 |
| b32_c16k | optimistic-assumption | local-kv-only | 0.000000 | 0.117000 | 0.000000 | 0.000000 |
| b32_c16k | optimistic-assumption | cxl-kv-only | 0.005183 | 1.716624 | 1468.777440 | 0.942021 |
| b32_c16k | optimistic-assumption | local-kv-plus-expert | 0.000000 | 0.088920 | 0.000000 | 0.000000 |
| b32_c16k | optimistic-assumption | cxl-kv-plus-local-kv | 0.005183 | 1.716624 | 1468.777440 | 0.942021 |
| b32_c16k | optimistic-assumption | cxl-kv-plus-local-kv-plus-expert | 0.005183 | 1.716624 | 1468.777440 | 0.942021 |

所有行都由精确 request-level simulation 生成，interpolation=0。这里的 CXL READ
数量是每个压力点一个 Decode layer 的受控请求量；结果不能直接解释为完整 48 层
端到端时延，下一步才接入 Decode event graph。
