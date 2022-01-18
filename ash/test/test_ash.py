import unittest
from ash import ASH, NProfile


class ASHTestCase(unittest.TestCase):

    def test_add_node(self):
        a = ASH(hedge_removal=True)
        a.add_node(1, start=0, end=2, attr_dict={"label": "A"})
        self.assertEqual(a.has_node(1), True)

        a.add_node(2, start=4, end=10, attr_dict=NProfile(name="Giulio"))
        self.assertEqual(a.has_node(2), True)

    def test_add_nodes(self):
        a = ASH(hedge_removal=True)
        a.add_nodes([1, 2], start=0, end=2, node_attr_dict={1: {"label": "A"}, 2: {'label': 'B'}})
        self.assertEqual(a.has_node(1), True)
        self.assertEqual(a.has_node(2), True)
        self.assertEqual(a.has_node(1, tid=0), True)
        self.assertEqual(a.has_node(1, tid=3), False)
        self.assertEqual(a.has_node(1, tid=2), True)
        self.assertEqual(a.avg_number_of_nodes(), 2)

        self.assertEqual(a.get_node_snapshots(1), [0, 1, 2])

        self.assertEqual(a.coverage(), 1)
        self.assertEqual(a.node_contribution(1), 1)

    def test_node_attributes(self):
        a = ASH(hedge_removal=True)
        a.add_node(1, start=0, end=2, attr_dict={"label": "A"})

        attr = a.get_node_profile(1)
        self.assertEqual(attr, NProfile(**{'t': [[0, 2]], 'label': {0: 'A', 1: 'A', 2: 'A'}}))
        attr = a.get_node_profile(1, tid=0)
        self.assertEqual(attr, NProfile(**{'label': 'A'}))

        label = a.get_node_attribute(1, attribute_name='label')
        self.assertEqual(label, {0: 'A', 1: 'A', 2: 'A'})

        label = a.get_node_attribute(1, attribute_name='label', tid=0)
        self.assertEqual(label, 'A')

    def test_node_set(self):
        a = ASH(hedge_removal=True)
        a.add_node(1, start=0, end=0, attr_dict={"label": "A"})
        a.add_node(2, start=1, end=3, attr_dict={"label": "A"})
        a.add_node(3, start=3, end=4, attr_dict={"label": "A"})

        self.assertEqual(a.get_node_set(), {1, 2, 3})
        self.assertEqual(a.get_node_set(tid=0), {1})
        self.assertEqual(a.get_node_set(tid=3), {2, 3})
        self.assertEqual(a.get_node_set(tid=4), {3})

        self.assertEqual(a.get_number_of_nodes(), 3)
        self.assertEqual(a.get_number_of_nodes(0), 1)

    def test_node_iterator(self):
        a = ASH(hedge_removal=True)
        a.add_node(1, start=0, end=0, attr_dict={"label": "A"})
        a.add_node(2, start=1, end=3, attr_dict={"label": "A"})
        a.add_node(3, start=3, end=4, attr_dict={"label": "A"})

        iter_res = a.node_iterator()
        self.assertEqual(list(iter_res), [1, 2, 3])

        iter_res = a.node_iterator(tid=3)
        self.assertEqual(list(iter_res), [2, 3])
        iter_res = a.node_iterator(tid=4)
        self.assertEqual(list(iter_res), [3])

    def test_hyperedge(self):
        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2, 3], 0, 1)
        a.add_hyperedge([1, 2, 3], 6, 9)
        a.add_hyperedge([1, 2, 3], -3, -2)
        a.add_hyperedge([3, 4, 5], 3, 4)

        self.assertEqual(a.get_avg_number_of_hyperedges(), 1.0)
        self.assertEqual(a.hyperedge_contribution('e1'), 0.8)

        self.assertEqual(a.has_hyperedge([1, 2, 3]), True)
        self.assertEqual(a.has_hyperedge([1, 2, 4]), False)
        self.assertEqual(a.has_hyperedge([1, 2, 3], tid=0), True)
        self.assertEqual(a.has_hyperedge([1, 2, 3], tid=5), False)

        self.assertEqual(a.get_size(), 2)
        self.assertEqual(a.get_size(0), 1)

        self.assertEqual(a.get_number_of_neighbors(1), 3)
        self.assertEqual(a.get_number_of_neighbors(1, hyperedge_size=3), 3)
        self.assertEqual(a.get_number_of_neighbors(1, hyperedge_size=4), 0)
        self.assertEqual(a.get_number_of_neighbors(1, tid=0), 3)
        self.assertEqual(a.get_number_of_neighbors(1, tid=100), 0)

        self.assertEqual(a.get_degree(1), 1)
        self.assertEqual(a.get_degree(1, hyperedge_size=3), 1)
        self.assertEqual(a.get_degree(1, hyperedge_size=4), 0)
        self.assertEqual(a.get_degree(1, tid=0), 1)
        self.assertEqual(a.get_degree(1, tid=100), 0)

        hs = a.get_hyperedge_id_set()
        self.assertEqual(len(hs), 2)
        self.assertEqual(hs, {'e1', 'e2'})
        hs = a.get_hyperedge_id_set(tid=4)
        self.assertEqual(hs, {'e2'})

        a.add_hyperedges([[4, 5], [6, 7], [8, 9, 10]], start=3, end=10)
        hs = a.get_hyperedge_id_set()
        self.assertEqual(len(hs), 5)
        self.assertEqual(hs, {'e1', 'e2', 'e3', 'e4', 'e5'})

        self.assertEqual(a.get_hyperedge_id([1, 2, 3]), 'e1')
        self.assertEqual(a.get_hyperedge_nodes('e1'), [1, 2, 3])

        self.assertEqual(a.get_hyperedge_weight('e1'), 3)
        a.add_hyperedge([1, 2, 3], 9, 11)
        self.assertEqual(a.get_hyperedge_weight('e1'), 4)

        self.assertEqual(a.has_hyperedge([1, 2, 3]), True)
        self.assertEqual(a.has_hyperedge([1, 2, 3, 4]), False)
        self.assertEqual(a.has_hyperedge([1, 2, 3], tid=0), True)
        self.assertEqual(a.has_hyperedge([1, 2, 3], tid=100), False)

        self.assertEqual(a.has_hyperedge_id('e1'), True)
        self.assertEqual(a.has_hyperedge_id('e9'), False)
        self.assertEqual(a.has_hyperedge_id('e1', tid=0), True)
        self.assertEqual(a.has_hyperedge_id('e1', tid=100), False)

        self.assertEqual(a.uniformity(), 0.3157894736842105)

    def test_temporal_slice(self):
        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2, 3], 0, 1)
        a.add_hyperedge([1, 2, 3], 6, 9)
        a.add_hyperedge([1, 2, 5], 5, 10)
        a.add_hyperedge([3, 4, 5], 3, 4)

        b = a.hypergraph_temporal_slice(0)
        self.assertIsInstance(b, ASH)
        self.assertEqual(b.get_node_set(), {1, 2, 3})

        c = a.hypergraph_temporal_slice(5, 7)
        self.assertIsInstance(c, ASH)
        self.assertEqual(c.get_node_set(), {1, 2, 3, 5})

    def test_interactions(self):
        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2, 3], 0, 1)
        a.add_hyperedge([1, 2, 3], 6, 9)
        a.add_hyperedge([1, 2, 5], 5, 10)
        a.add_hyperedge([3, 4, 5], 3, 4)

        for he in a.stream_interactions():
            self.assertEqual(len(he), 3)
