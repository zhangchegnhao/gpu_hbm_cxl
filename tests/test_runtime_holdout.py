from __future__ import annotations

import unittest

from scripts.evaluate_runtime_holdout import (
    _budget_split,
    _select_counts,
    _signature_folds,
)


class RuntimeHoldoutTest(unittest.TestCase):
    def test_budget_split_preserves_requested_nonzero_points(self) -> None:
        self.assertEqual(_budget_split(5), (2, 3))
        self.assertEqual(_budget_split(9), (4, 5))
        self.assertEqual(_budget_split(15), (7, 8))
        self.assertEqual(_budget_split(25), (12, 13))

    def test_select_counts_keeps_endpoints_and_is_deterministic(self) -> None:
        available = {1, 2, 3, 4, 8, 16, 32, 49}
        first = _select_counts(available, 4)
        second = _select_counts(available, 7)
        self.assertEqual(second, _select_counts(available, 7))
        self.assertEqual(second[0], 1)
        self.assertEqual(second[-1], 49)
        self.assertEqual(len(second), 7)
        self.assertLessEqual(set(first), set(second))

    def test_signature_folds_keep_all_rows_and_groups_together(self) -> None:
        class Load:
            def __init__(self, token_count: int) -> None:
                self.token_count = token_count

        class Trace:
            def __init__(self, counts: tuple[int, ...]) -> None:
                self.expert_loads = tuple(Load(value) for value in counts)

        traces = tuple(
            Trace(counts)
            for counts in ((1, 2), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10))
        )
        folds = _signature_folds(traces, 3)
        flattened = [index for fold in folds for index in fold]
        self.assertEqual(sorted(flattened), list(range(len(traces))))
        self.assertEqual(len({index for index in folds[0] if index in (0, 1)}), 2)
        self.assertTrue(all(not ({0, 1} & set(fold)) or {0, 1} <= set(fold) for fold in folds))


if __name__ == "__main__":
    unittest.main()
