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

        self.assertEqual(len(sources), 2)
        self.assertEqual(len(targets), 5)

        dg, sources, targets = temporal_s_dag(
            a, s=1, start_from="e1", start=0, end=1, edge=True
        )
        self.assertEqual(len(sources), 2)
        self.assertEqual(len(targets), 2)

    def test_time_respecting_s_walks(self):
        a = self.get_hypergraph()
        pts = time_respecting_s_walks(a, 1, start_from="e1", stop_at="e5")

        for p in pts:
            self.assertIsInstance(p, tuple)

        self.assertEqual(
            len(
                time_respecting_s_walks(
                    a, 1, start_from="e1", stop_at="e5", start=4, end=4
                )
            ),
            1,
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
