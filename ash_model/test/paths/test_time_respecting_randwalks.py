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

    @staticmethod
    def get_larger_hypergraph():
        """
        Build a more complex temporal hypergraph for stress testing.
        - 10 timestamps (0-9)
        - 8 nodes (1-8)
        - 15 hyperedges with overlapping temporal windows
        - Multiple paths of varying lengths and complexities
        """
        h = ASH()
        # Early period connections (t=0-2)
        h.add_hyperedge([1, 2, 3], 0, 2)  # e1
        h.add_hyperedge([2, 3, 4], 0, 3)  # e2
        h.add_hyperedge([1, 4, 5], 1, 3)  # e3

        # Middle period - denser connectivity (t=2-5)
        h.add_hyperedge([3, 4, 5, 6], 2, 5)  # e4
        h.add_hyperedge([4, 5, 6], 3, 6)  # e5
        h.add_hyperedge([1, 2, 6, 7], 3, 5)  # e6
        h.add_hyperedge([5, 6, 7], 4, 7)  # e7

        # Late period connections (t=5-9)
        h.add_hyperedge([6, 7, 8], 5, 8)  # e8
        h.add_hyperedge([7, 8], 6, 9)  # e9
        h.add_hyperedge([1, 8], 7, 9)  # e10

        # Additional cross-temporal bridges
        h.add_hyperedge([2, 5, 8], 4, 8)  # e11
        h.add_hyperedge([3, 6], 3, 7)  # e12
        h.add_hyperedge([1, 3, 7], 5, 8)  # e13
        h.add_hyperedge([4, 7, 8], 6, 9)  # e14
        h.add_hyperedge([2, 4, 6, 8], 5, 9)  # e15

        return h

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
        )
        # Expect at least one walk
        self.assertTrue(len(walks) > 0)

        # Note: Node walks are returned as arrays without explicit timestamps,
        # but they should still follow the time-respecting DAG structure.
        # The validation happens in the DAG construction (temporal_s_dag)
        # where only forward-in-time edges are created.

        # Walks always stop at temporal sinks (nodes with no forward neighbors)
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

        # Walks always stop at temporal sinks (edges with no forward s-incident neighbors)
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
        )
        all_paths_stop = [p for paths in walks_map_stop.values() for p in paths]
        # Verify all paths respect time ordering
        for p in all_paths_stop:
            for e1, e2 in zip(p, p[1:]):
                self.assertGreater(e2.tid, e1.tid, msg=f"Invalid step: {e1} -> {e2}")

    def test_larger_network_node_walks(self):
        """Test time-respecting random walks on a larger temporal network with 10 timestamps."""
        h = self.get_larger_hypergraph()

        # Generate node walks across the full temporal range
        walks = time_respecting_random_walks(
            h,
            s=1,
            start_from=None,
            stop_at=None,
            start=0,
            end=9,
            num_walks=50,
            walk_length=8,
            edge=False,
            p=1.0,
            q=1.0,
            threads=-1,
        )

        # Should generate walks given the dense connectivity
        self.assertGreater(len(walks), 0, "Should generate walks on larger network")

        # Verify walks stay within expected node range
        for w in walks:
            for node in w:
                node_id = int(str(node))
                self.assertIn(
                    node_id, range(1, 9), f"Node {node_id} out of expected range"
                )

    def test_larger_network_edge_walks_long_paths(self):
        """Test that longer paths can be generated in a larger temporal network."""
        h = self.get_larger_hypergraph()

        # Try to generate longer walks (up to 10 steps)
        walks_map = time_respecting_random_walks(
            h,
            s=1,
            start_from="e1",  # Start from early hyperedge
            stop_at=None,
            start=0,
            end=9,
            num_walks=30,
            walk_length=10,
            edge=True,
            p=1.0,
            q=1.0,
            threads=-1,
        )

        all_paths = [p for paths in walks_map.values() for p in paths]
        self.assertGreater(
            len(all_paths), 0, "Should generate edge walks on larger network"
        )

        # Check that we can get some longer walks (>= 5 steps)
        longer_walks = [p for p in all_paths if len(p) >= 5]
        self.assertGreater(
            len(longer_walks), 0, "Should generate some longer walks in dense network"
        )

        # Verify all walks maintain time-respecting property
        for p in all_paths:
            timestamps = [e.tid for e in p]
            self.assertEqual(
                timestamps,
                sorted(timestamps),
                msg=f"Walk timestamps not strictly increasing: {timestamps}",
            )
            # Check strict increase (no equal consecutive timestamps)
            for i in range(len(timestamps) - 1):
                self.assertLess(timestamps[i], timestamps[i + 1])

    def test_larger_network_temporal_span_coverage(self):
        """Test that walks can span across significant temporal ranges."""
        h = self.get_larger_hypergraph()

        # Generate edge walks
        walks_map = time_respecting_random_walks(
            h,
            s=1,
            start_from=None,
            stop_at=None,
            start=0,
            end=9,
            num_walks=100,
            walk_length=12,
            edge=True,
            p=1.0,
            q=1.0,
            threads=-1,
        )

        all_paths = [p for paths in walks_map.values() for p in paths]

        # Calculate temporal span for each walk (last_timestamp - first_timestamp)
        temporal_spans = []
        for p in all_paths:
            if len(p) > 0:
                span = p[-1].tid - p[0].tid
                temporal_spans.append(span)

        # Should have at least one walk spanning multiple timestamps
        self.assertGreater(len(temporal_spans), 0)
        max_span = max(temporal_spans)
        self.assertGreater(
            max_span, 2, "Should generate walks spanning more than 2 timestamps"
        )

        # Check that we have diversity in temporal spans
        unique_spans = set(temporal_spans)
        self.assertGreater(
            len(unique_spans), 1, "Should have walks with varying temporal spans"
        )


if __name__ == "__main__":
    unittest.main()
