# Sieve HBM-PIM cycle-v1 design

## Scope

`cycle-v1` models the expert memory phase of one MoE decode layer. Ordinary
GPU expert-weight reads and PIM commands are injected in the same Ramulator
simulation and are arbitrated by the same 128 HBM channel controllers. The
backend returns separate GPU-weight-ready and PIM-result-ready milestones.

This is a contention-aware local HBM-PIM model, not a complete Sieve paper
reproduction. GPU arithmetic, Router, Scheduler, and non-expert GPU kernels
remain analytic.

## Topology and addresses

The topology is fixed by `configs/ramulator/sieve_hbm3e_cycle_v1.json`:

```text
8 stacks x 16 channels/stack x 2 PCH/channel = 256 PCH
1 SID x 6 bank groups x 4 banks = 24 banks/PCH
```

Both request classes use the explicit vector:

```text
Channel, PseudoChannel, Sid, BankGroup, Bank, Row, Column
```

GPU reads are striped over every PCH and bank. Bank is the fastest-changing
dimension, followed by the 32 cache-line positions in a row and then row.
Each active expert contributes exactly 9,437,184 bytes, or 294,912 32-byte
transactions. This mapping is deterministic but is a project assumption; the
Sieve paper does not publish a physical expert address mapping.

Each PIM request represents one all-bank command for one PCH. All 24 banks in
a PCH use the same logical row for a wave, so a single PIM open-row value per
PCH is sufficient for the lockstep all-bank state.

## Dual row buffers

The normal HBM row state remains in Ramulator's `DRAMDevice`. A separate PIM
open-row state is maintained per PCH. A closed PIM buffer pays `nRCDRD`; a row
change pays `nRP + nRCDRD`. These timings come from the selected HBM3 timing
preset. Row transitions are implicit logical PIM ACT/PRE operations rather
than new native HBM3 commands.

Formal tables require `dual_row_buffer=true`. The false setting is only an
ablation: a PIM row transition closes matching normal row state and blocks
normal issue during the logical transition. It is not a calibrated single-row
buffer device model.

## Shared-resource arbitration

Normal ACT/PRE commands continue on Ramulator's HBM row-command slot. Normal
RD/WR and `PIM_GWRITE`, `PIM_MAC`, and `PIM_READ` compete for the controller's
column-command slot. When both classes are eligible, priority alternates after
each contested issue. This prevents starvation.

Treating `PIM_MAC` as occupying one column-command issue slot is a cycle-v1
model assumption because the paper does not publish the command-bus encoding.
The next MAC for a PCH is still rate-limited to 24,576 ps; PIM I/O uses
1,024 ps. The frontend uses the same picosecond accumulator to avoid filling
queues with requests that cannot yet be serviced. The controller remains the
source of completion timing.

Requests are FIFO within each PCH. The controller may choose between the two
PCH heads, but it cannot prepare a future row while an older request for that
PCH remains queued.

## Timing-table contract

The contention table is keyed exactly by:

```text
GPU expert IDs, PIM expert IDs, GPU token count, PIM token count
```

No interpolation is permitted. Schema v2 contains the legacy placements plus
the exact hot-prefix candidates visited by a monotonic crossing bisection. It
runs a mixed shape for every candidate and GPU-only/PIM-only shapes for the
legacy and selected placements. Unselected search rows use null isolated
timings instead of substituting an analytic estimate. The evidence file records
injected/completed request counts, cycles, wave counts, row transitions, queue
wait, arbitration statistics, candidate objectives, and the selected prefix.

Legacy `sieve` still searches with isolated `cycle-v0` estimates and is retained
unchanged as a baseline. `sieve-cycle-v1` enumerates the exact measured rows and
minimizes `scheduler + max(GPU mixed READ + GPU compute, PIM mixed path)`.
The search covers hot-to-cold expert prefixes, verifies opposing path
monotonicity at every measured point, and measures the two integer neighbors of
the crossing. It is not an exhaustive search over every possible expert set.

## Known limitations

- PIM ACT/PRE are logical delays, not native per-bank Ramulator commands.
- Refresh, energy, thermal limits, and power are absent.
- GPU compute and non-expert GPU memory stages remain analytic.
- PIM attention still comes from the isolated cycle-v0 table.
- GPU reads are a deterministic synthetic stream, not an Accel-Sim trace.
- The Router trace is synthetic and the experiment covers one layer/step.
- The placement optimizer is restricted to deterministic hot-expert prefixes.
- A negative isolated-to-mixed completion delta is possible because PIM
  column-slot gaps can change FR-FCFS ordering while ACT uses a separate bus.
