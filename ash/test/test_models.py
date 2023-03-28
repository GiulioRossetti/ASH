import unittest

from ash.models import *


class ModelsTestCase(unittest.TestCase):

    def test_random_ASH(self):
        a = random_ASH(n_nodes=100, n_hyperedges=100, max_edge_size=10, n_tids=3,
                       attr_to_vals_dict={'party': ['L', 'R'], 'age': [20, 30, 40]})

        self.assertEqual(a.get_number_of_nodes(), 100)
        self.assertEqual(a.get_size(), 100 * 3)
        for tid in a.temporal_snapshots_ids():
            self.assertEqual(a.get_size(tid), 100)
        self.assertListEqual([he for he in a.get_hyperedge_id_set() if len(a.get_hyperedge_nodes(he)) > 10], [])
        self.assertEqual(len(a.temporal_snapshots_ids()), 3)

        for tid in a.temporal_snapshots_ids():
            for n in a.get_node_set(tid=tid):
                attr = a.get_node_attribute(n, 'party', tid=tid)
                self.assertIn(attr, ['L', 'R'])
                attr = a.get_node_attribute(n, 'age', tid=tid)
                self.assertIn(attr, [20, 30, 40])
