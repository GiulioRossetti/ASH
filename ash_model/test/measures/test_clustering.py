import unittest

from ash_model import ASH
from ash_model.measures import (
    average_s_local_clustering_coefficient,
    inclusiveness,
    s_intersections,
    s_local_clustering_coefficient,
)


class ClusteringTestCase(unittest.TestCase):
    def test_LCC(self):
        a = ASH()
        a.add_hyperedge([1, 2], 0)
        a.add_hyperedge([1, 3], 0)
        a.add_hyperedge([1, 4], 0)

        LCC = s_local_clustering_coefficient(a, 1, "e1")
        self.assertEqual(LCC, 1)

        a = ASH()
        a.add_hyperedge([1, 2, 3, 5], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)

        LCC = s_local_clustering_coefficient(a, 1, "e1")
        self.assertEqual(LCC, 1)

        a = ASH()
        a.add_hyperedge([1, 2, 6, 7], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)
        a.add_hyperedge([3, 4, 5, 6, 7], 0)

        LCC = s_local_clustering_coefficient(a, 1, "e1")
        self.assertEqual(LCC, 1)

        a = ASH()
        a.add_hyperedge([1, 2], 0)  # “e1”
        a.add_hyperedge([1, 3], 0)  # “e2”
        a.add_hyperedge([1, 4], 0)  # “e3”
        a.add_hyperedge([2, 5], 0)  # “e4”

        lcc = s_local_clustering_coefficient(a, 1, "e1")
        self.assertAlmostEqual(lcc, 1 / 3, places=6)

    def test_avg_LCC(self):
        a = ASH()
        a.add_hyperedge([1, 2], 0)
        a.add_hyperedge([1, 3], 0)
        a.add_hyperedge([1, 4], 0)

        LCC = average_s_local_clustering_coefficient(a, 1)
        self.assertEqual(LCC, 1)

        a = ASH()
        a.add_hyperedge([1, 2, 3, 5], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)

        LCC = average_s_local_clustering_coefficient(a, 1)
        self.assertEqual(LCC, 1)

        a = ASH()
        a.add_hyperedge([1, 2, 6, 7], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)
        a.add_hyperedge([3, 4, 5, 6, 7], 0)

        LCC = average_s_local_clustering_coefficient(a, 1)
        self.assertEqual(LCC, 1)  # TODO: CHECK THIS

    def test_s_intersections_and_inclusiveness(self):
        a = ASH()
        # Build at t=0 three hyperedges with overlaps:
        # e1={1,2,3}, e2={2,3,4}, e3={1,2}
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([2, 3, 4], 0)
        a.add_hyperedge([1, 2], 0)
        # s=2: intersections are between e1-e2 (overlap {2,3}) and e1-e3 (overlap {1,2})
        inter = s_intersections(a, s=2, start=0, end=0)
        self.assertEqual(inter, 2)
        # inclusiveness: e3={1,2} is subset of e1 -> non-facet count=1 over total 3 => 1/3
        incl = inclusiveness(a, start=0, end=0)
        self.assertAlmostEqual(incl, 1 / 3)

    def test_lcc_missing_hyperedge_and_small_k(self):
        a = ASH()
        # two hyperedges overlapping ⇒ line‑graph has 2 nodes (each degree 1)
        a.add_hyperedge([1, 2], 0)  # e1
        a.add_hyperedge([2, 3], 0)  # e2
        # hyperedge not in line graph
        self.assertEqual(s_local_clustering_coefficient(a, 1, "ex"), 0.0)
        # k<2 ⇒ 0
        self.assertEqual(s_local_clustering_coefficient(a, 1, "e1"), 0.0)

    def test_avg_lcc_empty_graph_zero_and_inclusiveness_empty(self):
        a = ASH()
        # no hyperedges at this time window ⇒ empty line graph
        self.assertEqual(
            average_s_local_clustering_coefficient(a, 1, start=5, end=5), 0.0
        )
        # inclusiveness with no hyperedges ⇒ 0
        self.assertEqual(inclusiveness(a, start=5, end=5), 0.0)
