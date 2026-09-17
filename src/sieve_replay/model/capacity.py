"""Byte-level resource admission; this module performs no data transfers."""

from __future__ import annotations

from typing import Any


def classify_capacity(total_bytes: int, resident_capacity_bytes: int,
                      spill_capacity_bytes: int = 0) -> dict[str, Any]:
    """Classify one device domain using only an explicitly supplied spill budget.

    ``spill`` means the byte allocation fits the two supplied budgets; it does
    not implement paging or identify any physical spill device.  The default
    zero spill budget makes an over-capacity allocation ``oom`` and infeasible.
    ``resident_bytes``/``spill_bytes`` are admitted byte counts, not traffic.
    """
    values = (total_bytes, resident_capacity_bytes, spill_capacity_bytes)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("memory and capacity byte counts must be nonnegative integers")
    resident_bytes = min(total_bytes, resident_capacity_bytes)
    overflow_bytes = max(0, total_bytes - resident_capacity_bytes)
    spill_bytes = min(overflow_bytes, spill_capacity_bytes)
    unallocated_bytes = overflow_bytes - spill_bytes
    state = "oom" if unallocated_bytes else ("spill" if spill_bytes else "resident")
    return {
        "state": state,
        "feasible": state != "oom",
        "admission_status": "infeasible" if state == "oom" else "feasible",
        "resident_capacity_bytes": resident_capacity_bytes,
        "spill_capacity_bytes": spill_capacity_bytes,
        "resident_bytes": resident_bytes,
        "spill_bytes": spill_bytes,
        "unallocated_bytes": unallocated_bytes,
        "resident_headroom_bytes": max(0, resident_capacity_bytes - total_bytes),
        "spill_headroom_bytes": spill_capacity_bytes - spill_bytes,
    }
