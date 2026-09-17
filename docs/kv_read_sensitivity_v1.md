# KV READ controller sensitivity v1

`scripts/run_kv_read_sensitivity.py` replays the 19 exact request shapes from
`results/kv_read_experiment_v1` while changing only controller admission
parameters. The default plan has five variants:

| Variant | Read buffer | Row buffers |
|---|---:|---|
| `baseline-rb256-dual` | 256 | dual |
| `rb32-dual` | 32 | dual |
| `rb64-dual` | 64 | dual |
| `rb128-dual` | 128 | dual |
| `rb256-single` | 256 | single |

Plan and run:

```bash
PYTHONPATH=src python3 scripts/run_kv_read_sensitivity.py
PYTHONPATH=src python3 scripts/run_kv_read_sensitivity.py --run --workers 8
```

Outputs are written under `results/kv_read_sensitivity_v1/`. Every cache entry
contains its shape, variant and a context keyed by the sensitivity plan.
The plan records the original exact result, frozen inventory, cycle configuration
and runner/request-model hashes. The independent audit checks these records
against the saved results and current source files. Historical sensitivity runs
did not record binding/library binary hashes; this remains a provenance limit.

The completed scan has 95 results. Use the audit below to inspect those results;
it does not run Ramulator or replace the saved plan. The original runner rewrites
its plan and lacks an exclusive run lock, so do not launch concurrent runs or
rewrite the plan while another run is active. Two old plan namespaces remain in
the ignored cache directory; only the current plan's 95 entries are reported.

```bash
PYTHONPATH=src python3 scripts/audit_kv_read_sensitivity.py
PYTHONPATH=src python3 scripts/plot_kv_read_sensitivity.py
```

Install the plotting dependency from `requirements/kv_analysis.txt` if needed.
Detailed findings and matched isolated controls are in
`docs/kv_read_sensitivity_v1_analysis.md` and
`results/kv_read_sensitivity_v1/matched_comparison.json`.

This is a Ramulator request-level sensitivity study. KV and Expert streams are
abstract sequential HBM READs; the decode graph remains serial and no A800
hardware timing, paging, spill, eviction, or end-to-end bandwidth claim is
made. Staggered and serial injection modes are not implemented in this
revision; queue depth and row-buffer assumptions are the scanned variables.
