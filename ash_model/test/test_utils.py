import unittest

import ash_model.utils as ut

from ash_model.utils import (
    clique_projection,
    bipartite_projection,
    line_graph_projection,
    dual_hypergraph_projection,
    clique_projection_by_time,
    bipartite_projection_by_time,
    line_graph_projection_by_time,
    dual_hypergraph_projection_by_time,
)
from ash_model import ASH, NProfile
from scipy import sparse


import unittest
import networkx as nx

from ash_model import ASH


def get_hypergraph():
    a = ASH(edge_attributes=True)
    a.add_hyperedge([1, 2, 3], 0)
    a.add_hyperedge([15, 25], 0)
    a.add_hyperedge([1, 24, 34], 0)
    a.add_hyperedge([1, 2, 5, 6], 0)
    a.add_hyperedge([1, 2, 5], 1)
    a.add_hyperedge([3, 4, 5, 10], 1)
    a.add_hyperedge([3, 4, 5, 12], 1)
    return a


class MatricesTestCase(unittest.TestCase):

    def setUp(self):

        self.h = get_hypergraph()

    def test_incidence(self):
        # global incidence covers all snapshots (0 and 1)
        M = ut.incidence_matrix(self.h)

        # There are 9 unique nodes at t=0 plus possibly new at t=1 (node 4,10,12)
        # Unique nodes overall: {1,2,3,5,6,15,24,25,34,4,10,12} => 12
        # Total hyperedges overall: 7
        self.assertIsInstance(M, sparse.csr_matrix)
        self.assertEqual(M.shape, (12, 7))

        # check that all nodes and hyperedges are present
        node_map = ut.get_node_id_mapping(self.h)
        he_map = ut.get_hyperedge_id_mapping(self.h)

        for edge in self.h.hyperedges():
            eidx = he_map[edge]
            for node in self.h.get_hyperedge_nodes(edge):
                nidx = node_map[node]
                # check that the incidence matrix has a 1 at (node, edge)
                self.assertEqual(M[nidx, eidx], 1)

            # check that the incidence matrix has 0s where it should
            for node in self.h.nodes():
                if node not in self.h.get_hyperedge_nodes(edge):
                    nidx = node_map[node]
                    self.assertEqual(M[nidx, eidx], 0)

    def test_incidence_by_time(self):
        mats = ut.incidence_matrix_by_time(self.h)
        # should have two keys: 0 and 1
        self.assertCountEqual(list(mats.keys()), [0, 1])
        node_map = ut.get_node_id_mapping(self.h)
        he_map = ut.get_hyperedge_id_mapping(self.h)
        for t in [0, 1]:
            M = mats[t]
            self.assertIsInstance(M, sparse.csr_matrix)

            # shape
            self.assertEqual(M.shape, (12, 7))

            # check that if a node is in a hyperedge at time t, it has a 1 in the incidence matrix, else 0
            for edge in self.h.hyperedges(start=t, end=t):
                eidx = he_map[edge]
                for node in self.h.get_hyperedge_nodes(edge):
                    nidx = node_map[node]
                    self.assertEqual(M[nidx, eidx], 1)

                # check that the incidence matrix has 0s where it should
                for node in self.h.nodes():
                    if node not in self.h.get_hyperedge_nodes(edge):
                        nidx = node_map[node]
                        self.assertEqual(M[nidx, eidx], 0)

    def test_adjacency_global(self):
        A = ut.adjacency_matrix(self.h)
        self.assertIsInstance(A, sparse.csr_matrix)
        # adjacency is symmetric, shape 12×12, diagonal zeros
        self.assertEqual(A.shape, (12, 12))
        self.assertTrue((A.diagonal() == 0).all())

        node_map = ut.get_node_id_mapping(self.h)
        i1, i2 = node_map[1], node_map[2]
        i15, i25 = node_map[15], node_map[25]
        # Nodes 1 and 2 share three edges total ([1,2,3], [1,2,5,6], [1,2,5])
        self.assertEqual(A[i1, i2], 3)
        # 15 and 25 only share the one edge at t=0
        self.assertEqual(A[i15, i25], 1)
        # 1 and 15 never co-occur
        self.assertEqual(A[i1, i15], 0)

    def test_adjacency_by_time(self):
        ams = ut.adjacency_matrix_by_time(self.h)
        self.assertCountEqual(list(ams.keys()), [0, 1])

        for t in [0, 1]:
            A = ams[t]
            self.assertIsInstance(A, sparse.csr_matrix)

            # for each time step, the diagonal should be zero
            self.assertTrue((A.diagonal() == 0).all())

            # check that all node pairs that share an edge have a non-zero value
            node_map = ut.get_node_id_mapping(self.h)

            # check that nodes that are not connected have a zero value
            for node in self.h.nodes():
                nidx = node_map[node]
                for other_node in self.h.nodes():
                    if node != other_node:
                        other_nidx = node_map[other_node]
                        if A[nidx, other_nidx] > 0:
                            # they should be connected by at least one edge
                            connected = False
                            for edge in self.h.hyperedges(start=t, end=t, as_ids=False):
                                if node in edge and other_node in edge:
                                    connected = True
                                    break
                            self.assertTrue(connected)
                        else:
                            # they should not be connected by any edge
                            connected = False
                            for edge in self.h.hyperedges(start=t, end=t, as_ids=False):
                                if node in edge and other_node in edge:
                                    connected = True
                                    break
                            self.assertFalse(connected)


class TestCliqueProjection(unittest.TestCase):
    def setUp(self):
        # build hypergraph with overlapping hyperedges at time 0
        self.h = ASH()
        # nodes 1–3 in a triangle and 2–4 in an edge
        self.h.add_hyperedge([1, 2, 3], start=0)
        self.h.add_hyperedge([2, 4], start=0)
        # attach a node attribute for testing keep_attrs
        for n in [1, 2, 3, 4]:
            self.h.add_node(n, start=0, attr_dict=NProfile(n, color=str(n)))

    def test_structure_default(self):
        G = clique_projection(self.h, start=0, end=0, keep_attrs=False)

        expected = [
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
        ]
        edges = sorted([tuple(sorted((u, v))) for u, v in G.edges()])
        self.assertListEqual(edges, expected)

    def test_keep_attrs(self):
        G = clique_projection(self.h, start=0, end=0, keep_attrs=True)
        # node attributes should propagate
        for n in G.nodes():
            self.assertIn("color", G.nodes[n])
            self.assertEqual(G.nodes[n]["color"], {0: str(n)})

    def test_by_time(self):
        # add hyperedge at time 1
        self.h.add_hyperedge([1, 4], start=1)
        by_time = clique_projection_by_time(self.h, keep_attrs=False)
        # should have entries for times 0 and 1
        self.assertCountEqual(list(by_time.keys()), [0, 1])
        # compare each to direct call
        for t, Gt in by_time.items():
            direct = clique_projection(self.h, start=t, end=t)
            self.assertTrue(
                nx.is_isomorphic(
                    Gt, direct, edge_match=lambda a, b: a["weight"] == b["weight"]
                )
            )


class TestBipartiteProjection(unittest.TestCase):
    def setUp(self):
        self.h = ASH()
        # two hyperedges at t=0
        self.h.add_hyperedge([1, 2], start=0)
        self.h.add_hyperedge([2, 3], start=0)
        # node attributes
        for n in [1, 2, 3]:
            self.h.add_node(n, start=0, attr_dict=NProfile(n, label=f"N{n}"))

    def test_structure_default(self):
        G = bipartite_projection(self.h, start=0, end=0, keep_attrs=False)
        # check bipartite sets
        nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == 0}
        edges = {n for n, d in G.nodes(data=True) if d.get("bipartite") == 1}
        self.assertEqual(nodes, {1, 2, 3})
        # hyperedge IDs are strings like 'e1','e2'
        self.assertEqual(edges, {"e1", "e2"})
        # edges connect each node to its hyperedges
        for he in edges:
            members = set(self.h.get_hyperedge_nodes(he))
            self.assertEqual(set(G.neighbors(he)), members)

    def test_keep_attrs(self):
        G = bipartite_projection(self.h, start=0, end=0, keep_attrs=True)
        # only graph-nodes corresponding to ASH nodes get attrs

        for n in [1, 2, 3]:
            self.assertIn("label", G.nodes[n])
            self.assertEqual(G.nodes[n]["label"], {0: f"N{n}"})

    def test_by_time(self):
        self.h.add_hyperedge([3, 4], start=1)
        by_time = bipartite_projection_by_time(self.h, keep_attrs=False)
        self.assertCountEqual(list(by_time.keys()), [0, 1])
        for t, Gt in by_time.items():
            direct = bipartite_projection(self.h, start=t, end=t)
            self.assertTrue(nx.is_isomorphic(Gt, direct))


class TestLineGraphProjection(unittest.TestCase):
    def setUp(self):
        self.h = ASH()
        # three hyperedges at t=0
        self.h.add_hyperedge([1, 2], start=0)  # e1
        self.h.add_hyperedge([2, 3], start=0)  # e2
        self.h.add_hyperedge([4, 5], start=0)  # e3

    def test_default_s1(self):
        G = line_graph_projection(self.h, s=1, start=0, end=0, keep_attrs=False)
        self.assertListEqual(sorted(G.nodes()), ["e1", "e2", "e3"])
        # e1 and e2 share node 2
        self.assertIn(("e1", "e2"), G.edges())
        # no edge between e1 and e3
        self.assertNotIn(("e1", "e3"), G.edges())

    def test_s2_threshold(self):
        # add a hyperedge sharing two nodes with e1
        self.h.add_hyperedge([1, 2, 6], start=0)  # e4
        G = line_graph_projection(self.h, s=2, start=0, end=0)
        # only e1 and e4 share two nodes
        self.assertIn(("e1", "e4"), G.edges())
        self.assertNotIn(("e2", "e4"), G.edges())

    def test_by_time(self):
        self.h.add_hyperedge([2, 3], start=1)
        by_time = line_graph_projection_by_time(self.h, s=1, keep_attrs=False)
        self.assertCountEqual(list(by_time.keys()), [0, 1])
        for t, Gt in by_time.items():
            direct = line_graph_projection(self.h, s=1, start=t, end=t)
            self.assertTrue(nx.is_isomorphic(Gt, direct))


class TestDualHypergraphProjection(unittest.TestCase):
    def setUp(self):
        self.h = ASH()
        # hyperedges at t=0
        self.h.add_hyperedge([1, 2], start=0)  # original edge e1
        self.h.add_hyperedge([2, 3], start=0)  # e2

    def test_dual_structure(self):
        dual, mapping = dual_hypergraph_projection(self.h, start=0, end=0)
        # mapping maps original nodes to dual hyperedge IDs
        self.assertEqual(set(mapping.keys()), {1, 2, 3})
        # dual hyperedges correspond to original nodes
        for node, new_eid in mapping.items():
            members = set(dual.get_hyperedge_nodes(new_eid))
            # should be the list of original hyperedges containing 'node'
            expected = {
                he
                for he in self.h.hyperedges(0, 0)
                if node in self.h.get_hyperedge_nodes(he)
            }
            self.assertEqual(members, expected)

    def test_by_time(self):
        # add an edge at time 1
        self.h.add_hyperedge([3, 4], start=1)
        by_time = dual_hypergraph_projection_by_time(self.h)
        self.assertCountEqual(list(by_time.keys()), [0, 1])
        for t, (dual_t, mapping_t) in by_time.items():
            dual_direct, map_direct = dual_hypergraph_projection(self.h, start=t, end=t)
            self.assertCountEqual(mapping_t.items(), map_direct.items())
            # compare dual hyperedge memberships
            for new_eid in mapping_t.values():
                self.assertEqual(
                    set(dual_t.get_hyperedge_nodes(new_eid)),
                    set(dual_direct.get_hyperedge_nodes(new_eid)),
                )


class TestFromNetworkxGraph(unittest.TestCase):
    def test_type_error_on_non_graph(self):
        with self.assertRaises(TypeError):
            ut.from_networkx_graph("not a graph", start=0)

    def test_basic_conversion_without_attrs(self):
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3)])
        h = ut.from_networkx_graph(G, start=5, end=6, keep_attrs=False)
        # hyperedges at t=5 and t=6
        hes5 = set(h.hyperedges(5, 6))
        self.assertEqual(len(hes5), 2)
        expected = {frozenset({1, 2}), frozenset({2, 3})}
        self.assertEqual({h.get_hyperedge_nodes(e) for e in hes5}, expected)
        # nodes present both times, no attrs
        for t in (5, 6):
            self.assertCountEqual(h.nodes(t), [1, 2, 3])
            for n in h.nodes(t):
                self.assertEqual(h.get_node_attributes(n, tid=t), {})

    def test_basic_conversion_with_attrs(self):
        G = nx.Graph()
        G.add_node(1, color="red")
        G.add_node(2, size=10)
        G.add_edge(1, 2)
        h = ut.from_networkx_graph(G, start=0, keep_attrs=True)
        # node profiles at t=0 keep attrs
        for n, data in G.nodes(data=True):
            self.assertEqual(h.get_node_attributes(n, tid=0), data)
        # one hyperedge
        hes = h.hyperedges(0)
        self.assertEqual(len(hes), 1)
        he_nodes = h.get_hyperedge_nodes(hes[0])
        self.assertEqual(he_nodes, frozenset({1, 2}))


class TestFromNetworkxGraphList(unittest.TestCase):
    def test_graph_list_conversion(self):
        G0 = nx.Graph()
        G0.add_edges_from([(1, 2)])
        G1 = nx.Graph()
        G1.add_edges_from([(2, 3)])
        h = ut.from_networkx_graph_list([G0, G1], keep_attrs=False)
        # time 0: one edge, nodes {1,2}
        self.assertCountEqual(h.hyperedges(0), h.hyperedges(0, 0))
        self.assertCountEqual(h.nodes(0), [1, 2])
        # time 1: one edge, nodes {2,3}
        self.assertCountEqual(h.hyperedges(1), h.hyperedges(1, 1))
        self.assertCountEqual(h.nodes(1), [2, 3])
        # global counts
        self.assertEqual(len(h.hyperedges()), 2)
        self.assertCountEqual(h.nodes(), [1, 2, 3])


class TestFromNetworkxMaximalCliques(unittest.TestCase):
    def test_type_error_on_non_graph(self):
        with self.assertRaises(TypeError):
            ut.from_networkx_maximal_cliques(123, start=0)

    def test_cliques_conversion(self):
        G = nx.Graph()
        # clique of size 3 and edge of size 2
        G.add_edges_from([(1, 2), (2, 3), (1, 3), (3, 4)])
        # add attr on node 4
        G.nodes[4]["role"] = "X"
        h = ut.from_networkx_maximal_cliques(G, start=2, end=3)
        # expected maximal cliques: {1,2,3} and {3,4}
        expected = [set(c) for c in nx.find_cliques(G) if len(c) > 1]
        hes = [set(h.get_hyperedge_nodes(e)) for e in h.hyperedges(2, 3)]
        self.assertCountEqual(hes, expected)
        # node attrs at t=2 and t=3
        for t in (2, 3):
            for n, data in G.nodes(data=True):
                self.assertEqual(h.get_node_attributes(n, tid=t), data)


class TestFromNetworkxMaximalCliquesList(unittest.TestCase):
    def test_sequence_conversion(self):
        G0 = nx.Graph()
        G0.add_edges_from([(1, 2), (2, 3), (1, 3)])
        G1 = nx.Graph()
        G1.add_edges_from([(3, 4)])
        h = ut.from_networkx_maximal_cliques_list([G0, G1])
        # time 0: one clique {1,2,3}
        hes0 = [set(h.get_hyperedge_nodes(e)) for e in h.hyperedges(0, 0)]
        self.assertEqual(hes0, [set([1, 2, 3])])
        # time 1: one clique {3,4}
        hes1 = [set(h.get_hyperedge_nodes(e)) for e in h.hyperedges(1, 1)]
        self.assertEqual(hes1, [set([3, 4])])
        # all nodes present across snapshots
        self.assertCountEqual(h.nodes(), [1, 2, 3, 4])


class TestFromNetworkxBipartite(unittest.TestCase):
    def setUp(self):
        # bipartite: nodes 1,2 on left; 'A','B' on right
        self.G = nx.Graph()
        left, right = {1, 2}, {"A", "B"}
        self.G.add_nodes_from(left, bipartite=0)
        self.G.add_nodes_from(right, bipartite=1)
        self.G.add_edges_from([(1, "A"), (2, "A"), (2, "B")])

    def test_type_error_on_non_graph(self):
        with self.assertRaises(TypeError):
            ut.from_networkx_bipartite("nope", start=0)

    def test_value_error_on_non_bipartite(self):
        H = nx.Graph()
        H.add_edges_from([(1, 2), (2, 3), (1, 3)])
        with self.assertRaises(ValueError):
            ut.from_networkx_bipartite(H, start=0)

    def test_basic_bipartite_conversion(self):
        h = ut.from_networkx_bipartite(self.G, start=7, end=8, keep_attrs=False)
        # get bipartite sets
        nodes, edges = nx.bipartite.sets(self.G)
        # check nodes and hyperedges
        self.assertListEqual(sorted(h.nodes()), sorted(list(nodes)))
        self.assertListEqual(sorted(h.hyperedges()), ["e1", "e2"])


class TestFromNetworkxBipartiteList(unittest.TestCase):
    def test_sequence_bipartite_conversion(self):
        h = get_hypergraph()
        # make it connected
        h.add_hyperedge(h.nodes(), start=0, end=1)
        g = ut.bipartite_projection(h, keep_attrs=False)

        glist = [g, g.copy(), g.copy()]

        h2 = ut.from_networkx_bipartite_list(glist, keep_attrs=False)
        print(h2.hyperedges(as_ids=False))
        self.assertListEqual(sorted(h2.nodes()), sorted(h.nodes()))
        self.assertCountEqual(sorted(h2.hyperedges()), sorted(h.hyperedges()))

        # check time
        self.assertListEqual(h2.temporal_snapshots_ids(), [0, 1, 2])
        for t in h.temporal_snapshots_ids():
            # all nodes present at each time step
            self.assertCountEqual(h2.nodes(t), h2.nodes(t))
            # hyperedges are the same as in the bipartite projection
            for he in h2.hyperedges(t):
                nodes = h2.get_hyperedge_nodes(he)
                # self.assertIn(set(nodes), [set(g.neighbors(r)) for r in g.nodes() if g.nodes[r]['bipartite'] == 1])


if __name__ == "__main__":
    unittest.main()
