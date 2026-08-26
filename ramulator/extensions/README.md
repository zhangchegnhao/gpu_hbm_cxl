# Ramulator extensions

`sieve_hbm_pim/` is the project-owned overlay for pinned Ramulator 2.1 revision
`b30320bc9385b708e86b67ebb9f48858cc66d798`.

The overlay adds:

- explicit `PIM_GWRITE`, `PIM_MAC`, and `PIM_READ` request types;
- a frontend that injects one command wave across all 256 pseudo-channels;
- a controller with independent per-pseudo-channel issue timing;
- memory-system draining after the frontend has injected its last request;
- cycle milestones used to generate many strict timing-table entries in three
  Ramulator runs.

This is `cycle-v0`, not the final Sieve memory model. See
`sieve_hbm_pim/README.md` for implemented behavior and exclusions.
