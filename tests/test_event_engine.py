from __future__ import annotations

import unittest

from sieve_replay.simulation import Event, EventEngine


class EventEngineTest(unittest.TestCase):
    def test_independent_resources_overlap_and_join_waits(self) -> None:
        graph = (
            Event("root", "test", ("GPU",), (), 2.0),
            Event("gpu_branch", "test", ("GPU",), ("root",), 5.0),
            Event("pim_branch", "test", ("PIM",), ("root",), 8.0),
            Event("join", "test", ("GPU",), ("gpu_branch", "pim_branch"), 1.0),
        )
        events = EventEngine().run(graph)
        by_name = {event.event.name: event for event in events}
        self.assertEqual(by_name["gpu_branch"].start_us, 2.0)
        self.assertEqual(by_name["pim_branch"].start_us, 2.0)
        self.assertEqual(by_name["join"].start_us, 10.0)
        self.assertEqual(by_name["join"].end_us, 11.0)
        self.assertEqual(
            EventEngine.critical_path(events),
            ("root", "pim_branch", "join"),
        )

    def test_shared_resource_serializes_events(self) -> None:
        graph = (
            Event("first", "test", ("GPU",), (), 3.0),
            Event("second", "test", ("GPU",), (), 4.0),
        )
        events = EventEngine().run(graph)
        self.assertEqual(events[1].start_us, 3.0)
        self.assertEqual(events[1].blocking_event, "first")


if __name__ == "__main__":
    unittest.main()
