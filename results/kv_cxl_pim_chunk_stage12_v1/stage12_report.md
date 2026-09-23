# KV CXL-PIM chunk pipeline Stage 12

Stage 12 replaces the Stage-11 overlap bounds with exact chunk readiness and an explicit GPU merge model for B8/C32k.

| design | chunks/slots | pipeline us/layer | merge us/layer | Decode ms | tok/s | vs memory-only |
|---|---:|---:|---:|---:|---:|---:|
| c2_mac49152_p4 | 1/1 | 9130.146 | 1.200 | 573.369 | 13.953 | +45.553% |
| c2_mac49152_p4 | 8/1 | 9225.391 | 9.597 | 573.772 | 13.943 | +45.451% |
| c2_mac49152_p4 | 8/2 | 9159.965 | 9.597 | 573.772 | 13.943 | +45.451% |
| c4_mac24576_p4 | 1/1 | 2304.425 | 1.399 | 573.378 | 13.952 | +45.550% |
| c4_mac24576_p4 | 8/1 | 2463.267 | 11.195 | 573.849 | 13.941 | +45.431% |
| c4_mac24576_p4 | 8/2 | 2334.244 | 11.195 | 573.849 | 13.941 | +45.431% |
| c8_mac24576_p4 | 1/1 | 1183.473 | 1.799 | 573.398 | 13.952 | +45.545% |
| c8_mac24576_p4 | 8/1 | 1469.512 | 14.390 | 574.002 | 13.937 | +45.392% |
| c8_mac24576_p4 | 8/2 | 1213.292 | 14.390 | 574.002 | 13.937 | +45.392% |

Exact Ramulator workloads: `9`; new executions in this invocation: `9`; interpolation: `0`.
All three one-chunk runs reproduce the Stage-10 pipeline in `10` cycle fields.

Each chunk returns a complete FP32 online-softmax partial state. Therefore the 8-chunk cases transfer and read eight partial states per PCH, increasing result traffic rather than assuming a free tile boundary. One buffer slot waits for result-link completion before launching the next chunk; two slots permit the following PIM chunk to overlap the prior result transfer.

GPU merge is an analytic roofline model using the configured simulated GPU peak and HBM bandwidth plus one kernel launch per chunk. It is not an A800 measurement. The event scheduler launches local Attention and CXL-PIM after RoPE, treats local GPU Attention as non-preemptive, and schedules ready merge kernels afterward on the same GPU_COMPUTE resource.

B8/C32k routing remains a controlled B8/C4k template and local Attention remains an A800 C4k-C16k trend extrapolation. CXL-PIM topology and timing are Ramulator assumptions, not physical CXL-PIM, Shared CXL-PIM, or multi-GPU results.
