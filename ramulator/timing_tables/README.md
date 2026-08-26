# Ramulator timing tables

Generated files are stored under `generated/`. Each table records:

- the pinned Ramulator revision;
- an extension content hash;
- the cycle configuration hash;
- the trace-generator content hash;
- the model, hardware, and Router trace hashes for cycle-v1;
- exact batch/context and token-count keys.

The companion `*.evidence.json` records wave counts, simulator cycles, injected
requests, and completed requests. Table lookup is strict: a replay fails if an
exact shape is absent.

Generate the Qwen3 single-layer table with:

```bash
scripts/build_sieve_ramulator.sh
PYTHONPATH=src python3 scripts/generate_ramulator_table.py \
  --experiment configs/experiments/single_layer_smoke.json \
  --cycle-config configs/ramulator/sieve_hbm3e_cycle_v0.json \
  --output ramulator/timing_tables/generated/qwen3_single_layer_cycle_v0.json
```

Generate the cycle-v1 mixed GPU/PIM contention table with:

```bash
PYTHONPATH=src python3 scripts/generate_contention_table.py \
  --experiment configs/experiments/single_layer_cycle_v1.json \
  --cycle-config configs/ramulator/sieve_hbm3e_cycle_v1.json \
  --output ramulator/timing_tables/generated/qwen3_single_layer_cycle_v1_contention.json
```

Cycle-v1 generation caches every completed shape under `generated/.cache/`.
The formal table is keyed by exact GPU/PIM expert sets and token counts. Mixed
simulation caches are invalidated by cycle configuration, extension, or mixed
workload-driver changes. Schema v2 records the measured hot-prefix candidates,
selected prefix, monotonicity check, and nullable isolated fields for unselected
candidates; mixed candidate timings are always strict Ramulator results.
