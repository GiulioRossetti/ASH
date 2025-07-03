import unittest

from ash_model.measures import *


class SCentralitiesCase(unittest.TestCase):
    @staticmethod
    def get_hypergraph():
        a = ASH()
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([1, 4], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        a.add_hyperedge([1, 3, 4], 1)
        a.add_hyperedge([3, 4], 1)
        return a

    def test_centralities(self):
        a = self.get_hypergraph()

        # Betweenness centrality
        self.assertDictEqual(
            s_betweenness_centrality(a, s=1),
            {"e1": 0.0, "e2": 0.0, "e3": 0.0, "e4": 0.0, "e5": 0.0},
        )
        self.assertDictEqual(
            s_betweenness_centrality(a, s=1, edges=False),
            {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        )

        # Closeness centrality
        self.assertDictEqual(
            s_closeness_centrality(a, s=1),
            {"e1": 1.0, "e2": 1.0, "e3": 1.0, "e4": 1.0, "e5": 1.0},
        )
        self.assertDictEqual(
            s_closeness_centrality(a, s=1, edges=False),
            {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        )

        # Eccentricity
        self.assertDictEqual(
            s_eccentricity(a, s=1),
            {"e1": 1.0, "e2": 1.0, "e3": 1.0, "e4": 1.0, "e5": 1.0},
        )
        self.assertDictEqual(
            s_eccentricity(a, s=1, edges=False),
            {1: 1, 2: 1, 3: 1, 4: 1},
        )

        # Harmonic centrality
        self.assertDictEqual(
            s_harmonic_centrality(a, s=1),
            {"e1": 4.0, "e2": 4.0, "e3": 4.0, "e4": 4.0, "e5": 4.0},
        )
        self.assertDictEqual(
            s_harmonic_centrality(a, s=1, edges=False),
            {1: 3.0, 2: 3.0, 3: 3.0, 4: 3.0},
        )

        # Katz centrality rounded to 3 decimals
        katz_edge = s_katz(a, s=1)
        katz_edge = {k: round(v, 3) for k, v in katz_edge.items()}
        self.assertDictEqual(
            katz_edge,
            {"e1": 0.447, "e2": 0.447, "e3": 0.447, "e4": 0.447, "e5": 0.447},
        )
        katz_node = s_katz(a, s=1, edges=False)
        katz_node = {k: round(v, 3) for k, v in katz_node.items()}
        self.assertDictEqual(
            katz_node,
            {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5},
        )

        # Load centrality
        self.assertDictEqual(
            s_load_centrality(a, s=1),
            {"e1": 0.0, "e2": 0.0, "e3": 0.0, "e4": 0.0, "e5": 0.0},
        )
        self.assertDictEqual(
            s_load_centrality(a, s=1, edges=False),
            {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        )

        # Eigenvector centrality rounded to 3 decimals
        eig_edge = s_eigenvector_centrality(a, s=1)
        eig_edge = {k: round(v, 3) for k, v in eig_edge.items()}
        self.assertDictEqual(
            eig_edge,
            {"e1": 0.447, "e2": 0.447, "e3": 0.447, "e4": 0.447, "e5": 0.447},
        )
        eig_node = s_eigenvector_centrality(a, s=1, edges=False)
        eig_node = {k: round(v, 3) for k, v in eig_node.items()}
        self.assertDictEqual(
            eig_node,
            {1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5},
        )

        # Information centrality rounded to 3 decimals
        info_edge = s_information_centrality(a, s=1)
        info_edge = {k: round(v, 3) for k, v in info_edge.items()}
        self.assertDictEqual(
            info_edge,
            {"e1": 0.625, "e2": 0.625, "e3": 0.625, "e4": 0.625, "e5": 0.625},
        )
        info_node = s_information_centrality(a, s=1, edges=False)
        info_node = {k: round(v, 3) for k, v in info_node.items()}
        self.assertDictEqual(
            info_node,
            {1: 0.667, 2: 0.667, 3: 0.667, 4: 0.667},
        )

        # Second-order centrality rounded to 3 decimals
        sec_edge = s_second_order_centrality(a, s=1)
        sec_edge = {k: round(v, 3) for k, v in sec_edge.items()}
        self.assertDictEqual(
            sec_edge,
            {"e1": 3.464, "e2": 3.464, "e3": 3.464, "e4": 3.464, "e5": 3.464},
        )
        sec_node = s_second_order_centrality(a, s=1, edges=False)
        sec_node = {k: round(v, 3) for k, v in sec_node.items()}
        self.assertDictEqual(
            sec_node,
            {1: 2.449, 2: 2.449, 3: 2.449, 4: 2.449},
        )
