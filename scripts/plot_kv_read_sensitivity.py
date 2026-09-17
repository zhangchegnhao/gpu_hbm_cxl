#!/usr/bin/env python3
"""Plot audited matched-mode sensitivity results without running Ramulator."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sieve_replay.report.writer import sha256_file

OUTPUT = ROOT / "results/kv_read_sensitivity_v1"
VARIANTS = ("baseline-rb256-dual", "rb128-dual", "rb64-dual", "rb32-dual", "rb256-single")
LABELS = ("256 / dual (baseline)", "128 / dual", "64 / dual", "32 / dual", "256 / single")
PLACEMENTS = ("gpu-only", "frozen-oracle", "fixed-half-prefix")


def main() -> None:
    audit_path = OUTPUT / "validation.json"
    matched_path = OUTPUT / "matched_comparison.json"
    audit = json.loads(audit_path.read_text())
    matched = json.loads(matched_path.read_text())
    if audit.get("matched_comparison_sha256") != sha256_file(matched_path):
        raise ValueError("matched comparison differs from audited evidence")
    if audit.get("status") != "passed_with_provenance_limitation":
        raise ValueError("sensitivity audit is missing or failed")
    for name in ("plan", "exact_results", "comparison"):
        if audit["input_hashes"][name + "_sha256"] != sha256_file(OUTPUT / f"kv_read_sensitivity_{name}.json"):
            raise ValueError(f"sensitivity {name} changed since audit")
    rows = matched["rows"]
    expected = {(v, c, p) for v in VARIANTS for c in ("b8_c4k", "b8_c8k", "b8_c16k", "b16_c8k") for p in PLACEMENTS}
    if len(rows) != 60 or {(r["variant"], r["case"], r["placement"]) for r in rows} != expected:
        raise ValueError("sensitivity comparison matrix is incomplete")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = OUTPUT / "plots"
    output.mkdir(exist_ok=True)
    figures = {}
    specifications = (
        ("combined_minus_isolated_max_us", "Completion above max(isolated) (us)", "B8: competition within each controller variant", "b8_matched_competition"),
        ("gpu_expert_completion_delta_us", "GPU Expert completion increase (us)", "B8: GPU Expert delay under simultaneous KV injection", "b8_gpu_expert_delay"),
        ("kv_accepted_to_column_issue_mean_us", "KV accepted-to-column issue (us)", "B8: KV residence before column issue", "b8_kv_residence"),
    )
    for field, ylabel, title, name in specifications:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharex=True, sharey=True)
        for ax, placement in zip(axes, PLACEMENTS):
            for variant, label in zip(VARIANTS, LABELS):
                selected = sorted((r for r in rows if r["batch_size"] == 8 and r["variant"] == variant and r["placement"] == placement), key=lambda r: r["context_length"])
                ax.plot([r["context_length"] for r in selected], [r[field] for r in selected], marker="o", label=label)
            ax.set_title(placement, fontsize=11)
            ax.set_xticks((4096, 8192, 16384), ("4k", "8k", "16k"))
            ax.set_xlabel("Context (tokens)")
            ax.grid(alpha=0.25)
        # Set the shared limit after all placements so no later series is clipped.
        axes[0].set_ylim(0, max(r[field] for r in rows if r["batch_size"] == 8) * 1.08)
        axes[0].set_ylabel(ylabel)
        fig.suptitle(title, fontsize=13)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=5, fontsize=9, title="READ buffer size / row-buffer model", title_fontsize=9)
        fig.text(0.5, 0.015, "Ramulator microbenchmark | step 0, layer 0 only | source/cache audited; historical binary hash unrecorded", ha="center", fontsize=8)
        fig.subplots_adjust(bottom=0.29, top=0.83, left=0.07, right=0.99, wspace=0.13)
        for extension in ("png", "svg"):
            path = output / f"{name}.{extension}"
            fig.savefig(path, dpi=160)
            figures[path.name] = sha256_file(path)
        plt.close(fig)
    manifest = {
        "classification": "audited Ramulator matched-control sensitivity plots; not A800 or end-to-end timing",
        "scope": "3 B8 contexts, 3 fixed placements, 5 controller variants; B16/C8k remains in matched comparison",
        "audit_sha256": sha256_file(audit_path), "matched_comparison_sha256": sha256_file(matched_path),
        "plotter_sha256": sha256_file(Path(__file__)), "figures": figures,
        "limitations": matched["limitations"],
    }
    (output / "plots_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"plots": len(specifications), "directory": str(output)}))


if __name__ == "__main__":
    main()
