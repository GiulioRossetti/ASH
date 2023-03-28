import unittest

from ash.measures import *


class ClusteringTestCase(unittest.TestCase):
    def test_LCC(self):
        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2], 0)
        a.add_hyperedge([1, 3], 0)
        a.add_hyperedge([1, 4], 0)

        LCC = s_local_clustering_coefficient(a, 1, "e1")
        self.assertEqual(LCC, 0)

        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2, 3, 5], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)

        LCC = s_local_clustering_coefficient(a, 1, "e1")
        self.assertEqual(LCC, 0)

        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2, 6, 7], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)
        a.add_hyperedge([3, 4, 5, 6, 7], 0)

        LCC = s_local_clustering_coefficient(a, 1, "e1")
        self.assertEqual(LCC, 1)

    def test_avg_LCC(self):
        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2], 0)
        a.add_hyperedge([1, 3], 0)
        a.add_hyperedge([1, 4], 0)

        LCC = average_s_local_clustering_coefficient(a, 1)
        self.assertEqual(LCC, 0)

        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2, 3, 5], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)

        LCC = average_s_local_clustering_coefficient(a, 1)
        self.assertEqual(LCC, 0)

        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2, 6, 7], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)
        a.add_hyperedge([3, 4, 5, 6, 7], 0)

        LCC = average_s_local_clustering_coefficient(a, 1)
        self.assertEqual(LCC, 1)

    def test_s_intersections(self):
        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2], 0)
        a.add_hyperedge([1, 3], 0)
        a.add_hyperedge([1, 4], 0)

        res = s_intersections(a, 1)
        self.assertEqual(res, 3)

    def test_inclusiveness(self):
        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2], 0)
        a.add_hyperedge([1, 3], 0)
        a.add_hyperedge([1, 4], 0)
        res = inclusiveness(a)
        self.assertEqual(res, 0)
        a.add_hyperedge([1, 2, 3, 5], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        res = inclusiveness(a)
        self.assertEqual(res, 0.6)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)
        a.add_hyperedge([3, 4, 5, 6, 7], 0)
        res = inclusiveness(a)
        self.assertEqual(res, 0.7142857142857143)
