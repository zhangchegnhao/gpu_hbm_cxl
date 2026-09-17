# KV capacity admission v1

This is a byte-level analytic admission model for four KV capacity stress points.
It reuses the existing full-model memory estimator. The 80 GB decimal A800 and
96 GB decimal simulated hardware are separate device domains; their capacities
are not added or treated as spill destinations. No A800 execution or Ramulator
simulation is performed here.

| Batch | Context | Peak estimate | A800 80 GB | Simulated 96 GB |
|---:|---:|---:|---|---|
| 8 | 32k | 86.835 GB | oom / infeasible | resident |
| 16 | 16k | 86.835 GB | oom / infeasible | resident |
| 16 | 32k | 112.605 GB | oom / infeasible | oom / infeasible |
| 32 | 16k | 112.607 GB | oom / infeasible | oom / infeasible |

The generic classifier supports `resident`, `spill`, and `oom` within **one**
capacity domain. `resident` fits that domain's resident bytes; `spill` additionally
requires an explicitly configured spill budget; `oom` exceeds those explicit
budgets and is marked `infeasible`. Both domains in the formal four-point result
use **zero spill capacity**. Hypothetical spill admission is tested only with unit
test byte budgets and does not introduce a new hardware device or parameter.

This implements resource admission and byte allocation accounting only. It does
not implement paging, data transfer, eviction, recovery or transfer timing. The
memory estimate also excludes allocator reserve, fragmentation, implementation
workspace and prefill activations. `resident` is therefore an estimated byte fit,
and `oom` is an analytical admission rejection, not observed A800 behavior.

The routing template is loaded from the existing real pilot, validated against
its schema-v2 manifest and prompt snapshot, then tiled only for this controlled
byte estimate. It does not create or relabel a real trace. JSON output preserves
input hashes, full configuration snapshots and source template identity.

Machine-readable output:

- `results/kv_capacity_states_v1/capacity_states.json`
- `results/kv_capacity_states_v1/capacity_states.csv`

Generator: `scripts/run_kv_capacity_states.py`. Generic admission function:
`src/sieve_replay/model/capacity.py`.
