# KV READ request model v1

The independent `SieveKVRead` frontend and Python `kv_read_workload` module add
an optional `kv_read_transactions` dimension. The existing `SieveMixed`
frontend, cycle-v1 interfaces, and their cache provenance remain unchanged.
`KVWorkloadShape` and `kv_read_cache_context()` give the new experiment its own
shape and provenance namespace; they cannot reuse an old Expert-only cache.

KV requests use ordinary HBM `Read` requests with `source_id=1`; GPU Expert
reads use `source_id=0`. Both enter the same controller read queue. Disjoint
row namespaces and unique request addresses prevent stream aliasing. Each
pseudo-channel attempts at most one normal READ per frontend tick, alternating
between pending streams after each successful admission. Both streams start
together: this is a controlled overlap stress assumption, not the sequential
Attention → Router → Expert schedule of a single-layer decode replay.

The frontend counts injected/completed requests, completion cycles, rejected
admission attempts, and the sum/maximum of accepted-to-completed residence
for each stream. The wrapper cross-checks the controller's normal READ total
against `expert + KV` and the controller residence sum against the two stream
sums. `accepted_to_column_issue_cycles` subtracts the resolved HBM read latency
from each completed request's residence. This includes activation/precharge
and scheduling delays and is not a pure FIFO queue wait.

The subtraction follows pinned Ramulator `controller_base.cpp`: normal READ
retirement sets `depart = issue_cycle + read_latency`; completion records
`depart - arrive`. HBM3 serializes its resolved `read_latency` into the
controller DRAM configuration, including the half-clock tick conversion.

The change does not modify formal cycle-v1 replay tables or claim a validated
KV bandwidth bottleneck. Nonzero results require the independently built
frontend and exact request-count checks. Paging, spill, and eviction remain
outside this model.

The reused controller's legacy `gpu_column_issues` and
`gpu_blocked_by_pim_cycles` counters describe all normal READs (Expert plus KV)
in this experiment. Only the frontend's source-specific counters separate KV
from GPU Expert reads. The address model stripes both streams across the same
banks with separate row ranges; it assumes neither cache hits nor a GPU memory
transaction trace. The injection schedule and row layout are modeling choices,
so measured interference is conditional on them.

The disjoint row ranges apply to the GPU Expert and KV normal READ streams.
PIM commands retain the baseline `SieveMixed` row addressing, which can overlap
the GPU row range under the configured dual-row-buffer abstraction. This model
does not establish physically disjoint storage for every Expert/PIM operand.

Build using `bash scripts/build_kv_read_ramulator.sh`, then run the small
experiment using `PYTHONPATH=src python3 scripts/run_kv_read_experiments.py --run
--workers 12`. Each of four cases uses only step 0, layer 0, with GPU-only,
frozen Expert-only Oracle, and fixed-half Expert placements. It is not a
re-optimization of the Oracle for KV contention. The complete long-context
replay baseline is verified by `scripts/freeze_kv_baseline.py --verify`.
