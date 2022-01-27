import unittest
from ash import ASH
from ash.measures import *


class SCentralitiesCase(unittest.TestCase):
    @staticmethod
    def get_hypergraph():
        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([1, 4], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        a.add_hyperedge([1, 3, 4], 1)
        a.add_hyperedge([3, 4], 1)
        return a

    def test_centralities(self):
        a = self.get_hypergraph()
        self.assertDictEqual(
            s_betweenness_centrality(a, s=1),
            {"e1": 0.0, "e2": 0.0, "e3": 0.0, "e4": 0.0, "e5": 0.0},
        )
        self.assertDictEqual(
            s_betweenness_centrality(a, s=1, edges=False),
            {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        )

        self.assertDictEqual(
            s_closeness_centrality(a, s=1),
            {"e1": 1.0, "e2": 1.0, "e3": 1.0, "e4": 1.0, "e5": 1.0},
        )
        self.assertDictEqual(
            s_closeness_centrality(a, s=1, edges=False),
            {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
        )

        self.assertDictEqual(
            s_eccentricity(a, s=1),
            {"e1": 1.0, "e2": 1.0, "e3": 1.0, "e4": 1.0, "e5": 1.0},
        )
        self.assertDictEqual(
            s_eccentricity(a, s=1, edges=False), {1: 1, 2: 1, 3: 1, 4: 1}
        )

        self.assertDictEqual(
            s_harmonic_centrality(a, s=1),
            {"e1": 4.0, "e3": 4.0, "e5": 4.0, "e4": 4.0, "e2": 4.0},
        )
        self.assertDictEqual(
            s_harmonic_centrality(a, s=1, edges=False), {1: 3.0, 2: 3.0, 3: 3.0, 4: 3.0}
        )

        self.assertDictEqual(
            s_katz(a, s=1),
            {
                "e1": 0.447213595499958,
                "e2": 0.4472135954999579,
                "e3": 0.447213595499958,
                "e4": 0.4472135954999579,
                "e5": 0.4472135954999579,
            },
        )
        self.assertDictEqual(
            s_katz(a, s=1, edges=False),
            {1: 0.49999999999999983, 2: 0.5, 3: 0.5, 4: 0.49999999999999994},
        )

        self.assertDictEqual(
            s_load_centrality(a, s=1),
            {"e1": 0.0, "e2": 0.0, "e3": 0.0, "e4": 0.0, "e5": 0.0},
        )
        self.assertDictEqual(
            s_load_centrality(a, s=1, edges=False), {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        )

        self.assertDictEqual(
            s_eigenvector_centrality(a, s=1),
            {
                "e1": 0.447213595499958,
                "e2": 0.447213595499958,
                "e3": 0.447213595499958,
                "e4": 0.44721359549995787,
                "e5": 0.4472135954999579,
            },
        )
        self.assertDictEqual(
            s_eigenvector_centrality(a, s=1, edges=False),
            {
                1: 0.4999999999999999,
                2: 0.5000000000000002,
                3: 0.4999999999999999,
                4: 0.5,
            },
        )

        self.assertDictEqual(
            s_information_centrality(a, s=1),
            {"e1": 0.625, "e2": 0.625, "e3": 0.625, "e4": 0.625, "e5": 0.625},
        )
        self.assertDictEqual(
            s_information_centrality(a, s=1, edges=False),
            {
                1: 0.6666666666666666,
                2: 0.6666666666666666,
                3: 0.6666666666666666,
                4: 0.6666666666666666,
            },
        )

        self.assertDictEqual(
            s_second_order_centrality(a, s=1),
            {
                "e1": 3.4641016151377535,
                "e2": 3.4641016151377544,
                "e3": 3.4641016151377535,
                "e4": 3.4641016151377544,
                "e5": 3.4641016151377544,
            },
        )
        self.assertDictEqual(
            s_second_order_centrality(a, s=1, edges=False),
            {
                1: 2.4494897427831774,
                2: 2.4494897427831774,
                3: 2.4494897427831774,
                4: 2.4494897427831774,
            },
        )
