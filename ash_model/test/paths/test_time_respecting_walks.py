import unittest

from ash_model import ASH
from ash_model.paths import (
    all_time_respecting_s_walks,
    annotate_walks,
    temporal_s_dag,
    time_respecting_s_walks,
)


class TimeRespectingWalksCase(unittest.TestCase):
    @staticmethod
    def get_hypergraph():
        a = ASH()
        a.add_hyperedge([1, 2, 3], 0, 4)
        a.add_hyperedge([1, 4], 0, 1)
        a.add_hyperedge([1, 2, 3, 4], 2, 3)
        a.add_hyperedge([1, 3, 4], 2, 3)
        a.add_hyperedge([3, 4], 3, 4)
        return a

    def test_incidence(self):
        a = self.get_hypergraph()
        self.assertEqual(
            sorted(a.get_s_incident("e1", s=1, start=1, end=1)), [("e2", 1)]
        )
        self.assertEqual(
            sorted(a.get_s_incident("e1", s=1, start=2, end=2)), [("e3", 3), ("e4", 2)]
        )
        self.assertEqual(
            sorted(a.get_s_incident("e1", s=1, start=0, end=2)),
            [("e2", 1), ("e3", 3), ("e4", 2)],
        )

    def test_temporal_dag(self):
        a = self.get_hypergraph()
        dg, sources, targets = temporal_s_dag(a, s=2, start_from="e1", edge=True)

        # e1 can start time-respecting walks from t0, t1, and t2 (has s-incident neighbors in future)
        self.assertEqual(len(sources), 3)
        # Can reach e3 and e4 at t2 and t3, plus e5 at t3 and t4
        self.assertEqual(len(targets), 6)

        dg, sources, targets = temporal_s_dag(
            a, s=1, start_from="e1", start=0, end=1, edge=True
        )
        # In time window [0,1], e1 can only start from t0 (reaching e2 at t1)
        self.assertEqual(len(sources), 1)
        # Only e2_1 is reachable
        self.assertEqual(len(targets), 1)

    def test_time_respecting_s_walks(self):
        a = self.get_hypergraph()
        pts = time_respecting_s_walks(a, 1, start_from="e1", stop_at="e5")

        for p in pts:
            self.assertIsInstance(p, tuple)

        # When start=4 and end=4, there's only one timestamp, so no future timestamps
        # exist for time-respecting walks. This correctly returns 0 walks.
        self.assertEqual(
            len(
                time_respecting_s_walks(
                    a, 1, start_from="e1", stop_at="e5", start=4, end=4
                )
            ),
            0,
        )

        pts = time_respecting_s_walks(a, 1, start_from="e1", stop_at="e5", sample=0.5)
        for p in pts:
            self.assertIsInstance(p, tuple)

    def test_all_time_respecting_paths(self):
        a = self.get_hypergraph()
        pts = all_time_respecting_s_walks(a, s=1)

        for p in pts:
            self.assertIsInstance(p, tuple)

    def test_annotated_paths(self):
        a = self.get_hypergraph()
        pts = all_time_respecting_s_walks(a, s=1)

        for _, ap in pts.items():
            v = annotate_walks(ap)
            for k, i in v.items():
                self.assertIn(
                    k,
                    [
                        "shortest",
                        "fastest",
                        "foremost",
                        "heaviest",
                        "fastest_shortest",
                        "fastest_heaviest",
                        "shortest_fastest",
                        "shortest_heaviest",
                        "heaviest_shortest",
                        "heaviest_fastest",
                    ],
                )
                self.assertIsInstance(i, list)

    def test_complex_temporal_network(self):
        """Test time-respecting walks on a more complex temporal network."""
        h = ASH()
        # Build a network with 12 timestamps and more complex structure
        # Early phase (t=0-3)
        h.add_hyperedge([1, 2, 3], 0, 2)  # e1
        h.add_hyperedge([2, 3, 4], 1, 3)  # e2
        h.add_hyperedge([3, 4, 5], 2, 4)  # e3

        # Middle phase (t=4-7) - denser connections
        h.add_hyperedge([1, 4, 6], 4, 6)  # e4
        h.add_hyperedge([2, 5, 6], 4, 7)  # e5
        h.add_hyperedge([3, 5, 7], 5, 8)  # e6
        h.add_hyperedge([4, 6, 7], 6, 9)  # e7

        # Late phase (t=8-11)
        h.add_hyperedge([5, 7, 8], 8, 10)  # e8
        h.add_hyperedge([6, 7, 8], 9, 11)  # e9

        # Cross-phase bridges
        h.add_hyperedge([1, 5, 8], 5, 10)  # e10
        h.add_hyperedge([2, 6], 3, 9)  # e11

        # Test DAG construction with multiple sources
        dg, sources, targets = temporal_s_dag(
            h, s=1, start_from=["e1", "e2"], edge=True
        )

        self.assertGreater(len(sources), 0, "Should have sources")
        self.assertGreater(len(targets), 0, "Should have targets")
        self.assertGreater(dg.number_of_edges(), 0, "Should have edges in DAG")

        # Test walks enumeration from early to late network
        walks = time_respecting_s_walks(
            h, s=1, start_from="e1", stop_at=None, start=0, end=11
        )

        # Should generate walks across temporal range
        self.assertGreater(len(walks), 0, "Should generate walks in complex network")

        # Verify temporal properties of generated walks
        all_walks = [w for walk_list in walks.values() for w in walk_list]

        # Check for walks with good temporal span
        temporal_spans = [w[-1].tid - w[0].tid for w in all_walks if len(w) > 0]
        if temporal_spans:
            max_span = max(temporal_spans)
            self.assertGreater(
                max_span, 3, "Should have walks spanning multiple timestamps"
            )

        # Verify strict temporal ordering in all walks
        for walk in all_walks:
            timestamps = [e.tid for e in walk]
            self.assertEqual(
                timestamps,
                sorted(timestamps),
                msg=f"Walk does not have strictly increasing timestamps: {timestamps}",
            )


if __name__ == "__main__":
    unittest.main()
