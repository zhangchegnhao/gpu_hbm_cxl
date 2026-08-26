# Sieve HBM-PIM cycle backends

The extension contains two versioned backends. `cycle-v0` is an isolated PIM
wave microbenchmark. `cycle-v1` adds mixed ordinary GPU reads, shared command
arbitration, and a logical dual-row-buffer PIM state. The cycle-v1 contract and
assumptions are specified in `docs/cycle_v1_design.md`.

## cycle-v0

## Topology

The microbenchmark instantiates 128 Ramulator controllers:

```text
8 stacks x 16 channels/stack x 2 pseudo-channels/channel = 256 PCH
```

The explicit address vector is:

```text
Channel, PseudoChannel, Sid, BankGroup, Bank, Row, Column
```

The organization uses `Sid=1`, `BankGroup=6`, and `Bank=4`, giving the 24
banks/PCH and 96 GB total capacity stated in Sieve Table 1.

## Command rate

One `PIM_MAC` command represents:

```text
24 banks x 32 bytes/bank x 1 op/byte = 768 operations/PCH
```

At 256 PCH and 8 TOPS aggregate throughput, the target interval is 24.576 ns
per PCH. A 32-byte I/O transaction at 8 TB/s aggregate has a 1.024 ns target
interval per PCH. The controller accumulates these intervals in picoseconds
and issues on the next 312 ps Ramulator tick, avoiding systematic integer-tick
rounding error.

## Implemented

- all-PCH parallel command-wave injection;
- independent PCH queues and rate limits;
- token broadcast, MAC, and result-read phases;
- HBM read latency on result reads;
- 1 KB logical row progression in generated address vectors;
- strict completion and request-count checks;
- per-wave cycle milestones.

## Not implemented

- NeuPIMs-style dual row buffers;
- ACT/PRE behavior for the PIM-side row buffer;
- refresh interference during PIM commands;
- concurrent ordinary GPU HBM traffic and PIM traffic in the same stack;
- command/data-bus conflicts between those two paths;
- power and energy.

Consequently, `cycle-v0` is appropriate for validating the trace-replay
pipeline and command-volume model. It is not yet a complete or calibrated
Sieve reproduction.

## cycle-v1 additions

- ordinary GPU HBM reads use native HBM3 ACT/RD/PRE timing;
- GPU reads and PIM requests share the column-command slot;
- alternating arbitration prevents GPU or PIM starvation;
- a separate logical PIM row buffer pays HBM3 `nRCDRD`/`nRP` delays;
- requests remain FIFO within each PCH;
- GPU and PIM completion milestones and contention statistics are reported;
- injected and completed request counts must match exactly.

Cycle-v1 still omits refresh, energy, native per-bank PIM ACT/PRE commands,
GPU arithmetic simulation, and real GPU memory traces.
