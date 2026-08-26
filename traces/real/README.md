# Real router traces

Real Qwen3 router traces are intentionally not checked into the repository.
Capture them with `scripts/capture_qwen3_router_trace.py`; this records routing
decisions only and never imports wall-clock model execution time into the
simulator.

Each trace directory contains:

```text
router.jsonl
manifest.json
```

The manifest binds the trace SHA-256 to an immutable model revision, model
shape, dtype, dataset revision/split, prompt count, decode steps, random seed,
framework versions and device map. Experiments using a path under
`traces/real/` must set `trace_manifest`; replay rejects missing or stale
manifests.

Prompt input is JSONL with exactly two fields per line:

```json
{"request_id":"request-0","prompt":"..."}
```

`request_id` remains the `token_id` across every layer and decode step. The
capture path currently assumes a fixed batch with no early request removal.
