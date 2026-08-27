# Real router traces

Real Qwen3 router traces are intentionally not checked into the repository.
Capture them with `scripts/capture_qwen3_router_trace.py`; this records routing
decisions only and never imports wall-clock model execution time into the
simulator.

Each trace directory contains:

```text
prompts.jsonl
router.jsonl
manifest.json
```

Schema-v2 manifests bind both the trace and exact prompt snapshot SHA-256 to an
immutable model revision, model shape, dtype, dataset revision/split, prompt
count, decode steps, random seed, generation semantics, framework versions and
device map. Schema-v1 remains readable for older captures. Experiments using a path under
`traces/real/` must set `trace_manifest`; replay rejects missing or stale
manifests.

Prompt input is JSONL with exactly two fields per line:

```json
{"request_id":"request-0","prompt":"..."}
```

`request_id` remains the `token_id` across every layer and decode step. The
capture path currently assumes a fixed batch with no early request removal.

Use `traces/real/prompts.pilot.jsonl` for the first Batch-8 pipeline run. The
full protocol and current capture-environment gate are documented in
`docs/real_router_trace_stage.md`.
