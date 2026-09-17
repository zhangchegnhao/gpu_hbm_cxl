# Long-context Router capture v1

Five independent Qwen3-30B-A3B Router captures were completed on one
`NVIDIA A800-SXM4-80GB` cloud instance. The model revision is
`ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, with BF16, 48 layers, 128 experts,
Top-8 routing, Transformers 4.53.2, and PyTorch 2.7.1+cu126.

Each directory contains its own `prompts.jsonl`, `router.jsonl`, and
`manifest.json`. The manifest and file hashes were checked on the capture host
and again after import. The captures use greedy argmax and eight fixed Decode
steps. They contain routing decisions only; no A800 wall-clock timing is
included.

The transferred archive `qwen3_kv_long_context_traces.tar.gz` has SHA-256
`b1f2ea3c62806947bc7f1972fd559162ac851164eaa179b8b49ff5e6d12f7f6b`.
The archive and model weights are not repository artifacts; the five
manifest-bound trace directories are the reproducible inputs.

| configuration | records | Context values | router SHA-256 |
|---|---:|---|---|
| B8/C4k | 3072 | 4096–4103 | `afc5f6bf27a84bdf54294cd79b01bdc54037b6d8905ace7e6ae5c1e60eab750f` |
| B8/C8k | 3072 | 8192–8199 | `24e318ce191d407a49b2b0e049706d7f43224018b8920459169f5ae722cb3bc9` |
| B8/C16k | 3072 | 16384–16391 | `2cee5ba38e5d91d1bab52926974e03ed6338c9f9a76ffb6f02c39584aa089463` |
| B16/C4k | 6144 | 4096–4103 | `8cea512f5d4172ae4b8f6c54ee876f773660d0199eebae1d02a0aa88564f9a88` |
| B16/C8k | 6144 | 8192–8199 | `03ac02f1de90c0486918fe6cd17c379c2ab58c74ca0d517b186a15c4aa61d552` |

The imported traces are under
`traces/real/qwen3_30b_a3b/kv_{b8_c4k,b8_c8k,b8_c16k,b16_c4k,b16_c8k}`.
Router analysis passed for every configuration: each has 384 layer-batches and
is classified as `manifest-bound-real`. Unique load-signature counts are 43,
64, 41, 115, and 103 in the table order.

The long-context capture initially exhausted the A800 during prefill because
the LM head attempted to materialize logits for every context token. The
capture path now passes `logits_to_keep=1`; only the final prefill token is
needed to seed greedy Decode. This changes the memory behavior of capture and
does not change the Router hook or captured routing semantics.

The next timing step is to plan and fill exact Ramulator workloads separately
for each imported Trace. The first planning pass found 65, 218, 66, 118, and
276 unique cycle-v1 shapes respectively; interpolation remains forbidden.
