import unittest

from ash_model.measures import *


class ClusteringTestCase(unittest.TestCase):
    def test_LCC(self):
        a = ASH(edge_attributes=True)
        a.add_hyperedge([1, 2], 0)
        a.add_hyperedge([1, 3], 0)
        a.add_hyperedge([1, 4], 0)

        LCC = s_local_clustering_coefficient(a, 1, "e1")
        self.assertEqual(LCC, 1)

        a = ASH(edge_attributes=True)
        a.add_hyperedge([1, 2, 3, 5], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)

        LCC = s_local_clustering_coefficient(a, 1, "e1")
        self.assertEqual(LCC, 1)

        a = ASH(edge_attributes=True)
        a.add_hyperedge([1, 2, 6, 7], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)
        a.add_hyperedge([3, 4, 5, 6, 7], 0)

        LCC = s_local_clustering_coefficient(a, 1, "e1")
        self.assertEqual(LCC, 1)

        a = ASH(edge_attributes=True)
        a.add_hyperedge([1, 2], 0)   # “e1”
        a.add_hyperedge([1, 3], 0)   # “e2”
        a.add_hyperedge([1, 4], 0)   # “e3”
        a.add_hyperedge([2, 5], 0)   # “e4”

        lcc = s_local_clustering_coefficient(a, 1, "e1")
        self.assertAlmostEqual(lcc, 1/3, places=6)

    def test_avg_LCC(self):
        a = ASH(edge_attributes=True)
        a.add_hyperedge([1, 2], 0)
        a.add_hyperedge([1, 3], 0)
        a.add_hyperedge([1, 4], 0)

        LCC = average_s_local_clustering_coefficient(a, 1)
        self.assertEqual(LCC, 1)

        a = ASH(edge_attributes=True)
        a.add_hyperedge([1, 2, 3, 5], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)

        LCC = average_s_local_clustering_coefficient(a, 1)
        self.assertEqual(LCC, 1)

        a = ASH(edge_attributes=True)
        a.add_hyperedge([1, 2, 6, 7], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)
        a.add_hyperedge([3, 4, 5, 6, 7], 0)

        LCC = average_s_local_clustering_coefficient(a, 1)
        self.assertEqual(LCC, 1) # TODO: CHECK THIS
    

    
