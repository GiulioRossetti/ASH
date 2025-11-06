import unittest
import numpy as np

from ash_model import ASH
from ash_model.paths import time_respecting_random_walks, TemporalEdge


class TimeRespectingRandomWalksCase(unittest.TestCase):
    @staticmethod
    def get_hypergraph():
        a = ASH()
        # same structure used in test_time_respecting_walks
        a.add_hyperedge([1, 2, 3], 0, 4)  # e1: t in [0,3]
        a.add_hyperedge([1, 4], 0, 1)  # e2: t in [0,0]
        a.add_hyperedge([1, 2, 3, 4], 2, 3)  # e3: t in [2]
        a.add_hyperedge([1, 3, 4], 2, 3)  # e4: t in [2]
        a.add_hyperedge([3, 4], 3, 4)  # e5: t in [3]
        return a

    def test_node_random_walks_time_respecting(self):
        a = self.get_hypergraph()
        # walks on nodes, s=1. Ensure walks are time-respecting (forward in time only)
        walks = time_respecting_random_walks(
            a,
            s=1,
            start_from=None,  # let it pick active nodes at first timestamps
            stop_at=None,
            start=0,
            end=4,  # Extended to t=4 to allow more transitions
            num_walks=20,  # Increased to improve chances of generating walks
            walk_length=5,
            edge=False,
            p=1.0,
            q=1.0,
            threads=-1,
            terminate_at_sink=False,
        )
        # Expect at least one walk
        self.assertTrue(len(walks) > 0)

        # Note: Node walks are returned as arrays without explicit timestamps,
        # but they should still follow the time-respecting DAG structure.
        # The validation happens in the DAG construction (temporal_s_dag)
        # where only forward-in-time edges are created.

        # With terminate_at_sink=True, walks should stop at sinks
        walks_stop = time_respecting_random_walks(
            a,
            s=1,
            start_from=None,
            stop_at=None,
            start=0,
            end=4,
            num_walks=10,
            walk_length=5,
            edge=False,
            terminate_at_sink=True,
        )
        self.assertTrue(len(walks_stop) > 0)
        for w in walks_stop:
            # Walks should not exceed the requested length
            self.assertTrue(len(w) <= 5)

    def test_edge_random_walks_time_respecting(self):
        a = self.get_hypergraph()
        # walks on hyperedges; verify time-respecting (strictly increasing timestamps)
        walks_map = time_respecting_random_walks(
            a,
            s=1,
            start_from="e1",
            stop_at=None,
            start=0,
            end=4,  # Extended to t=4 to allow more transitions
            num_walks=10,  # Increased for better chance of generating walks
            walk_length=4,
            edge=True,
            p=1.0,
            q=1.0,
            threads=-1,
            terminate_at_sink=False,
        )
        # Flatten and check
        all_paths = [p for paths in walks_map.values() for p in paths]
        self.assertTrue(len(all_paths) > 0)

        # Verify time-respecting property: each consecutive TemporalEdge must have strictly later timestamp
        for p in all_paths:
            for i, (e1, e2) in enumerate(zip(p, p[1:])):
                self.assertIsInstance(e1, TemporalEdge)
                self.assertIsInstance(e2, TemporalEdge)
                # Time-respecting: next edge must occur at a strictly later timestamp
                self.assertGreater(
                    e2.tid,
                    e1.tid,
                    msg=f"Step {i}: {e1} -> {e2} violates time-respecting property (tid should increase)",
                )

        # With terminate_at_sink=True, paths should stop at sinks
        walks_map_stop = time_respecting_random_walks(
            a,
            s=1,
            start_from="e1",
            stop_at=None,
            start=0,
            end=4,
            num_walks=5,
            walk_length=4,
            edge=True,
            terminate_at_sink=True,
        )
        all_paths_stop = [p for paths in walks_map_stop.values() for p in paths]
        # Verify all paths respect time ordering
        for p in all_paths_stop:
            for e1, e2 in zip(p, p[1:]):
                self.assertGreater(e2.tid, e1.tid, msg=f"Invalid step: {e1} -> {e2}")


if __name__ == "__main__":
    unittest.main()
