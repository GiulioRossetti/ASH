import unittest
from ash import ASH
import ash.utils as ut


class UtilsTestCase(unittest.TestCase):

    @staticmethod
    def get_hypergraph():
        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([15, 25], 0)
        a.add_hyperedge([1, 24, 34], 0)
        a.add_hyperedge([1, 2, 5, 6], 0)
        a.add_hyperedge([1, 2, 5], 1)
        a.add_hyperedge([3, 4, 5, 10], 1)
        a.add_hyperedge([3, 4, 5, 12], 1)
        return a

    def test_node_mapping(self):
        a = self.get_hypergraph()
        index_to_node, node_to_index = ut.get_node_mapping(a)
        self.assertIsInstance(node_to_index, dict)
        self.assertIsInstance(index_to_node, dict)

    def test_get_hyperedge_id_mapping(self):
        a = self.get_hypergraph()
        index_to_edge, edge_to_index = ut.get_hyperedge_id_mapping(a)
        self.assertIsInstance(index_to_edge, dict)
        self.assertIsInstance(edge_to_index, dict)

    def test_get_incidence_matrix(self):
        a = self.get_hypergraph()
        _, node_to_index = ut.get_node_mapping(a)
        _, edge_to_index = ut.get_hyperedge_id_mapping(a)
        M = ut.get_incidence_matrix(a, node_to_index, edge_to_index)
        self.assertIsInstance(M, dict)

        M = ut.get_incidence_matrix(a, node_to_index, edge_to_index, tid=0)
        self.assertIsInstance(M, dict)

    def test_get_hyperedge_weight_matrix(self):
        a = self.get_hypergraph()
        _, edge_to_index = ut.get_hyperedge_id_mapping(a)
        M = ut.get_hyperedge_weight_matrix(a, edge_to_index)
        self.assertIsInstance(M, dict)

        M = ut.get_hyperedge_weight_matrix(a, edge_to_index, tid=0)
        self.assertIsInstance(M, dict)

    def test_get_vertex_degree_matrix(self):
        a = self.get_hypergraph()
        M = ut.get_vertex_degree_matrix(a)
        self.assertIsInstance(M, dict)

        M = ut.get_vertex_degree_matrix(a, 1)
        self.assertIsInstance(M, dict)