import json
import unittest

from networkx.algorithms import bipartite
import networkx as nx

from ash_model import ASH, NProfile


class ASHTestCase(unittest.TestCase):
    def test_add_node(self):
        a = ASH()
        a.add_node(1, start=0, end=2, attr_dict={"label": "A"})
        self.assertEqual(a.has_node(1), True)

        a.add_node(2, start=4, end=10, attr_dict=NProfile(2, name="Giulio"))
        self.assertEqual(a.has_node(2), True)

    def test_add_nodes(self):
        a = ASH()
        a.add_nodes(
            [1, 2],
            start=0,
            end=2,
            node_attr_dict={1: {"label": "A"}, 2: {"label": "B"}},
        )
        self.assertEqual(a.has_node(1), True)
        self.assertEqual(a.has_node(2), True)
        self.assertEqual(a.has_node(1, start=0), True)
        self.assertEqual(a.has_node(1, start=3), False)
        self.assertEqual(a.has_node(1, start=2), True)
        self.assertEqual(a.avg_number_of_nodes(), 2)

        self.assertEqual(a.node_presence(1), [0, 1, 2])

        self.assertEqual(a.coverage(), 1)
        self.assertEqual(a.node_contribution(1), 1)

    def test_degree_dist(self):
        a = ASH()
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([15, 25], 0)
        a.add_hyperedge([1, 24, 34], 0)
        a.add_hyperedge([1, 2, 5, 6], 0)
        a.add_hyperedge([1, 2, 5], 1)
        a.add_hyperedge([3, 4, 5, 10], 1)
        a.add_hyperedge([3, 4, 5, 12], 1)

        self.assertDictEqual(a.degree_distribution(), {4: 2, 3: 2, 1: 7, 2: 1})
        self.assertDictEqual(a.degree_distribution(start=0), {3: 1, 2: 1, 1: 7})
        self.assertDictEqual(
            a.degree_distribution(start=0, end=1), a.degree_distribution()
        )

    def test_node_attributes(self):
        a = ASH()
        a.add_node(1, start=0, end=2, attr_dict={"label": "A"})

        attr = a.get_node_profile(1)
        self.assertEqual(
            attr, NProfile(1, **{"t": [[0, 2]], "label": {0: "A", 1: "A", 2: "A"}})
        )
        attr = a.get_node_profile(1, tid=0)
        self.assertEqual(attr, NProfile(1, **{"label": "A"}))

        label = a.get_node_attribute(1, attr_name="label")
        self.assertEqual(label, {0: "A", 1: "A", 2: "A"})

        label = a.get_node_attribute(1, attr_name="label", tid=0)
        self.assertEqual(label, "A")

    def test_node_profiles(self):
        a = ASH()
        a.add_node(1, start=0, end=2, attr_dict=NProfile(1, label="A"))

        attr = a.get_node_profile(1)
        self.assertEqual(
            attr, NProfile(1, **{"t": [[0, 2]], "label": {0: "A", 1: "A", 2: "A"}})
        )
        attr = a.get_node_profile(1, tid=0)
        self.assertEqual(attr, NProfile(1, **{"label": "A"}))

        label = a.get_node_attribute(1, attr_name="label")
        self.assertEqual(label, {0: "A", 1: "A", 2: "A"})

        label = a.get_node_attribute(1, attr_name="label", tid=0)
        self.assertEqual(label, "A")

        a.add_node(1, start=3, end=4, attr_dict=NProfile(1, label="B"))
        attr = a.get_node_profile(1, 3)
        self.assertEqual(attr, NProfile(1, **{"label": "B"}))

    def test_nodes(self):
        a = ASH()
        a.add_node(1, start=0, end=0, attr_dict={"label": "A"})
        a.add_node(2, start=1, end=3, attr_dict={"label": "A"})
        a.add_node(3, start=3, end=4, attr_dict={"label": "A"})

        self.assertEqual(a.nodes(), [1, 2, 3])
        self.assertEqual(a.nodes(start=0), [1])
        self.assertEqual(a.nodes(start=3), [2, 3])
        self.assertEqual(a.nodes(start=4), [3])

        self.assertEqual(a.number_of_nodes(), 3)
        self.assertEqual(a.number_of_nodes(start=0), 1)
        self.assertEqual(a.number_of_nodes(start=0, end=2), 2)

    def test_hyperedge(self):
        a = ASH()
        a.add_hyperedge([1, 2, 3], 0, 1)
        a.add_hyperedge([1, 2, 3], 6, 9)
        a.add_hyperedge([1, 2, 3], -3, -2)
        a.add_hyperedge([3, 4, 5], 3, 4)

        self.assertEqual(a.avg_number_of_hyperedges(), 1.0)
        self.assertEqual(a.hyperedge_contribution("e1"), 0.8)

        self.assertEqual(a.has_hyperedge([1, 2, 3]), True)
        self.assertEqual(a.has_hyperedge([1, 2, 4]), False)
        self.assertEqual(a.has_hyperedge([1, 2, 3], start=0), True)
        self.assertEqual(a.has_hyperedge([1, 2, 3], start=5), False)

        self.assertEqual(a.size(), 2)
        self.assertEqual(a.size(0), 1)

        self.assertEqual(a.number_of_neighbors(1), 2)
        self.assertEqual(a.number_of_neighbors(1, hyperedge_size=3), 2)
        self.assertEqual(a.number_of_neighbors(1, hyperedge_size=4), 0)
        self.assertEqual(a.number_of_neighbors(1, start=0), 2)
        self.assertEqual(a.number_of_neighbors(1, start=100), 0)

        self.assertEqual(a.degree(1), 1)
        self.assertEqual(a.degree(1, hyperedge_size=3), 1)
        self.assertEqual(a.degree(1, hyperedge_size=4), 0)
        self.assertEqual(a.degree(1, start=0), 1)
        self.assertEqual(a.degree(1, start=100), 0)

        hs = a.hyperedges()
        self.assertEqual(len(hs), 2)
        self.assertEqual(sorted(hs), ["e1", "e2"])
        hs = a.hyperedges(start=4)
        self.assertEqual(hs, ["e2"])

        hs = a.hyperedges(hyperedge_size=3)
        self.assertEqual(sorted(hs), ["e1", "e2"])
        hs = a.hyperedges(hyperedge_size=4)
        self.assertEqual(hs, list())
        hs = a.hyperedges(hyperedge_size=3, start=4)
        self.assertEqual(hs, ["e2"])
        hs = a.hyperedges(hyperedge_size=2, start=4)
        self.assertEqual(hs, list())

        a.add_hyperedges([[4, 5], [6, 7], [8, 9, 10]], start=3, end=10)
        hs = a.hyperedges()
        self.assertEqual(len(hs), 5)
        self.assertEqual(sorted(hs), ["e1", "e2", "e3", "e4", "e5"])

        self.assertEqual(a.get_hyperedge_id([1, 2, 3]), "e1")
        self.assertEqual(a.get_hyperedge_nodes("e1"), frozenset([1, 2, 3]))

        # self.assertEqual(a.get_hyperedge_weight("e1"), 3)
        a.add_hyperedge([1, 2, 3], 9, 11)
        # self.assertEqual(a.get_hyperedge_weight("e1"), 4)

        self.assertEqual(a.has_hyperedge([1, 2, 3]), True)
        self.assertEqual(a.has_hyperedge([1, 2, 3, 4]), False)
        self.assertEqual(a.has_hyperedge([1, 2, 3], start=0), True)
        self.assertEqual(a.has_hyperedge([1, 2, 3], start=100), False)

        self.assertEqual(a.has_hyperedge("e1"), True)
        self.assertEqual(a.has_hyperedge("e9"), False)
        self.assertEqual(a.has_hyperedge("e1", start=0), True)
        self.assertEqual(a.has_hyperedge("e1", start=100), False)

        self.assertEqual(a.uniformity(), 0.4845360824742268)

        a = ASH()
        a.add_hyperedge([1, 2, 3], 0, 1)
        a.add_hyperedge([1, 4], 0, 2)
        a.add_hyperedge([1, 2, 4], 2, 3)

        self.assertEqual(sorted(a.hyperedges()), ["e1", "e2", "e3"])
        self.assertEqual(sorted(list(a.hyperedges(start=0))), ["e1", "e2"])
        self.assertEqual(sorted(list(a.hyperedges(start=2))), ["e2", "e3"])
        self.assertEqual(sorted(list(a.hyperedges(start=3))), ["e3"])
        self.assertEqual(sorted(list(a.hyperedges(start=2, end=3))), ["e2", "e3"])

        self.assertEqual(sorted(list(a.hyperedges(start=3, end=3))), ["e3"])

    def test_temporal_slice(self):
        a = ASH()
        a.add_hyperedge([1, 2, 3], 0, 1)
        a.add_hyperedge([1, 2, 3], 6, 9)
        a.add_hyperedge([1, 2, 5], 5, 10)
        a.add_hyperedge([3, 4, 5], 3, 4)

        b, old_to_new = a.temporal_slice(0)
        self.assertIsInstance(b, ASH)
        self.assertEqual(b.nodes(), [1, 2, 3])

        b, old_to_new = a.temporal_slice(0, 0)
        self.assertIsInstance(b, ASH)
        self.assertEqual(b.nodes(), [1, 2, 3])

        c, old_to_new = a.temporal_slice(5, 7)
        self.assertIsInstance(c, ASH)
        self.assertEqual(sorted(c.nodes()), [1, 2, 3, 5])

    def test_interactions(self):
        a = ASH()
        a.add_hyperedge([1, 2, 3], 0, 1)
        a.add_hyperedge([1, 2, 3], 6, 9)
        a.add_hyperedge([1, 2, 5], 5, 10)
        a.add_hyperedge([3, 4, 5], 3, 4)

        for he in a.stream_interactions():
            self.assertEqual(len(he), 3)

    def test_size_distribution(self):
        a = ASH()
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([15, 25], 0)
        a.add_hyperedge([1, 24, 34], 0)
        a.add_hyperedge([1, 2, 5, 6], 0)
        a.add_hyperedge([1, 2, 5], 1)
        a.add_hyperedge([3, 4, 5, 10], 1)
        a.add_hyperedge([3, 4, 5, 12], 1)

        self.assertDictEqual(a.hyperedge_size_distribution(), {2: 1, 3: 3, 4: 3})
        self.assertDictEqual(a.hyperedge_size_distribution(start=0), {2: 1, 3: 2, 4: 1})
        self.assertDictEqual(
            a.hyperedge_size_distribution(start=0, end=1), {2: 1, 3: 3, 4: 3}
        )

        self.assertEqual(a.degree_by_hyperedge_size(1), {3: 3, 4: 1})
        self.assertEqual(a.degree_by_hyperedge_size(1, start=1), {3: 1})

        self.assertEqual(a.s_degree(1, 4), 1)
        self.assertEqual(a.s_degree(1, 4, 0), 1)

    def test_star(self):
        a = ASH()
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([15, 25], 0)
        a.add_hyperedge([1, 24, 34], 0)
        a.add_hyperedge([1, 2, 5, 6], 0)
        a.add_hyperedge([1, 2, 5], 1)
        a.add_hyperedge([3, 4, 5, 10], 1)
        a.add_hyperedge([3, 4, 5, 12], 1)

        self.assertEqual(sorted(a.star(1)), ["e1", "e3", "e4", "e5"])
        self.assertEqual(sorted(a.star(1, start=0)), ["e1", "e3", "e4"])
        self.assertEqual(a.star(1, start=0, hyperedge_size=4), ["e4"])

    def test_line_graph(self):
        a = ASH()
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([1, 4], 0)
        a.add_hyperedge([2, 3, 4], 0)
        a.add_hyperedge([1, 3], 1)
        a.add_hyperedge([3, 4], 1)

        g = a.s_line_graph()
        g2 = nx.Graph()
        edges = [
            ("e1", "e2"),
            ("e1", "e4"),
            ("e1", "e3"),
            ("e1", "e5"),
            ("e2", "e3"),
            ("e2", "e4"),
            ("e2", "e5"),
            ("e3", "e4"),
            ("e3", "e5"),
            ("e4", "e5"),
        ]
        g2.add_edges_from(edges)

        self.assertEqual(nx.is_isomorphic(g, g2), True)

        g = a.s_line_graph(start=0, end=0)

        eds = sorted([("e1", "e2"), ("e1", "e3"), ("e2", "e3")])
        res = sorted([tuple(sorted(i)) for i in list(g.edges())])

        self.assertListEqual(res, eds)

    def test_bipartite(self):
        a = ASH()
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([1, 4], 0)
        a.add_hyperedge([2, 3, 4], 0)
        a.add_hyperedge([1, 3], 1)
        a.add_hyperedge([3, 4], 1)

        g = a.bipartite_projection()
        self.assertEqual(bipartite.is_bipartite(g), True)
        edges, nodes = bipartite.sets(g)
        self.assertEqual(sorted(edges), ["e1", "e2", "e3", "e4", "e5"])
        self.assertEqual(sorted(nodes),[1, 2, 3, 4])

        g = a.bipartite_projection(start=0, end=0)
        self.assertEqual(bipartite.is_bipartite(g), True)

    def test_dual(self):
        a = ASH()
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([1, 4], 0)
        a.add_hyperedge([2, 3, 4], 0)
        a.add_hyperedge([1, 3], 1)
        a.add_hyperedge([3, 4], 1)

        dual, mapping  = a.dual_hypergraph()
        self.assertIsInstance(dual, ASH)
        self.assertEqual(dual.number_of_nodes(), 5)
        self.assertEqual(dual.number_of_hyperedges(), 4)

    
    def test_s_incident(self):
        a = ASH()
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([1, 4], 0)
        a.add_hyperedge([1, 2, 4], 0)

        self.assertEqual(sorted(a.get_s_incident("e1", s=1)), [("e2", 1), ("e3", 2)])
        self.assertEqual(sorted(a.get_s_incident("e1", s=2)), [("e3", 2)])
        self.assertEqual(sorted(a.get_s_incident("e1", s=3)), [])

    def test_hyper_subgraph(self):
        a = ASH()
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([1, 4], 0)
        a.add_hyperedge([2, 3, 4], 0)
        a.add_hyperedge([1, 3], 1)
        a.add_hyperedge([3, 4], 1)

        a.add_node(1, start=0, end=1, attr_dict=NProfile(node_id=1, party="L", age=37))
        a.add_node(1, start=0, end=1, attr_dict=NProfile(node_id=2, party="L", age=20))

        b, eid_map = a.induced_hypergraph(("e1", "e2"))
        self.assertEqual(len(eid_map), 2)

    def test_removal(self):
        for backend in ["dense", "interval"]:
            
            a = ASH(backend=backend)
            a.add_hyperedge([1, 2, 3], 0)
            a.add_hyperedge([1, 4], 0)
            a.add_hyperedge([2, 3, 4], 0)
            a.add_hyperedge([1, 3], 1)
            a.add_hyperedge([3, 4], 1)
            a.add_hyperedge([3, 4], 2)

            a.remove_hyperedge("e1")
            self.assertEqual(a.has_hyperedge("e1"), False)
            self.assertEqual(len(a.hyperedges()), 4)

            a.remove_hyperedge("e5", start=0)
            self.assertEqual(a.has_hyperedge("e5"), True)
            self.assertEqual(a.has_hyperedge("e5", start=0), False)
            self.assertEqual(a.has_hyperedge("e5", start=1), True)

            a = ASH()
            a.add_hyperedge([1, 2, 3], 0)
            a.add_hyperedge([1, 4], 0)
            a.add_hyperedge([2, 3, 4], 0)
            a.add_hyperedge([1, 3], 1)
            a.add_hyperedge([3, 4], 1)
            a.add_hyperedge([3, 4], 2)

            a.remove_node(1)
            self.assertEqual(a.has_node(1), False)
            self.assertEqual(len(a.nodes()), 3)
            self.assertEqual(len(a.hyperedges()), 2)

            a.add_node(1, start=0, end=1, attr_dict=NProfile(node_id=1, party="L", age=37))
            a.remove_unlabelled_nodes("party")
            self.assertEqual(len(a.nodes()), 1)



