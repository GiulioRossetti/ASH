import unittest
import networkx as nx
from ash import ASH
from ash.algorithms import *


class SWalksCase(unittest.TestCase):
    @staticmethod
    def get_hypergraph():
        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([1, 4], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        a.add_hyperedge([1, 3, 4], 1)
        a.add_hyperedge([3, 4], 1)
        return a

    # Hyperedge walks

    def test_shortest_s_walk(self):

        a = self.get_hypergraph()
        s_w = shortest_s_walk(a, 2, "e1", "e2")
        self.assertEqual(len(s_w), 3)
        s_w = shortest_s_walk(a, 3, "e1", "e2")
        self.assertEqual(len(s_w), 0)

        s_w = shortest_s_walk(a, 2, "e1", "e2", weight=True)
        self.assertEqual(len(s_w), 3)

        s_w = shortest_s_walk(a, 2, "e1")
        self.assertDictEqual(
            s_w,
            {
                "e1": ["e1"],
                "e3": ["e1", "e3"],
                "e4": ["e1", "e4"],
                "e2": ["e1", "e3", "e2"],
                "e5": ["e1", "e3", "e5"],
            },
        )

        s_w = shortest_s_walk(a, 2)
        self.assertDictEqual(
            s_w,
            {
                "e1": {
                    "e1": ["e1"],
                    "e3": ["e1", "e3"],
                    "e4": ["e1", "e4"],
                    "e2": ["e1", "e3", "e2"],
                    "e5": ["e1", "e3", "e5"],
                },
                "e3": {
                    "e3": ["e3"],
                    "e1": ["e3", "e1"],
                    "e2": ["e3", "e2"],
                    "e4": ["e3", "e4"],
                    "e5": ["e3", "e5"],
                },
                "e4": {
                    "e4": ["e4"],
                    "e1": ["e4", "e1"],
                    "e2": ["e4", "e2"],
                    "e3": ["e4", "e3"],
                    "e5": ["e4", "e5"],
                },
                "e2": {
                    "e2": ["e2"],
                    "e3": ["e2", "e3"],
                    "e4": ["e2", "e4"],
                    "e1": ["e2", "e3", "e1"],
                    "e5": ["e2", "e3", "e5"],
                },
                "e5": {
                    "e5": ["e5"],
                    "e3": ["e5", "e3"],
                    "e4": ["e5", "e4"],
                    "e1": ["e5", "e3", "e1"],
                    "e2": ["e5", "e3", "e2"],
                },
            },
        )

        s_w = shortest_s_walk(a, 3)
        self.assertDictEqual(
            s_w,
            {
                "e1": {"e1": ["e1"], "e3": ["e1", "e3"], "e4": ["e1", "e3", "e4"]},
                "e3": {"e3": ["e3"], "e1": ["e3", "e1"], "e4": ["e3", "e4"]},
                "e4": {"e4": ["e4"], "e3": ["e4", "e3"], "e1": ["e4", "e3", "e1"]},
            },
        )

    def test_s_distance(self):
        a = self.get_hypergraph()
        self.assertEqual(s_distance(a, 1, "e1", "e2"), 1)
        self.assertEqual(s_distance(a, 2, "e1", "e2"), 2)
        self.assertEqual(s_distance(a, 3, "e1", "e2"), None)

        self.assertDictEqual(
            s_distance(a, 2, "e1"), {"e1": 0, "e2": 2, "e3": 1, "e4": 1, "e5": 2}
        )
        self.assertDictEqual(
            s_distance(a, 2, "e1", weight=True),
            {"e1": 0, "e2": 4, "e3": 3, "e4": 2, "e5": 4},
        )

        self.assertListEqual(
            list(s_distance(a, 2)),
            [
                ("e1", {"e1": 0, "e2": 2, "e3": 1, "e4": 1, "e5": 2}),
                ("e3", {"e1": 1, "e2": 1, "e3": 0, "e4": 1, "e5": 1}),
                ("e4", {"e1": 1, "e2": 1, "e3": 1, "e4": 0, "e5": 1}),
                ("e2", {"e1": 2, "e2": 0, "e3": 1, "e4": 1, "e5": 2}),
                ("e5", {"e1": 2, "e2": 2, "e3": 1, "e4": 1, "e5": 0}),
            ],
        )

    def test_average_s_distance(self):
        a = self.get_hypergraph()
        self.assertEqual(average_s_distance(a, 2), 1.3)
        self.assertEqual(average_s_distance(a, 3), 1.3333333333333333)
        self.assertEqual(average_s_distance(a, 2, start=0, end=0), 1.3333333333333333)

    def test_has_s_walk(self):
        a = self.get_hypergraph()
        self.assertEqual(has_s_walk(a, 2, "e1", "e2"), True)
        self.assertEqual(has_s_walk(a, 8, "e1", "e2"), False)
        self.assertEqual(has_s_walk(a, 2, "e1", "e2", start=0, end=0), True)
        self.assertEqual(has_s_walk(a, 2, "e1", "e2", start=1, end=1), False)

    def test_diameter(self):
        a = self.get_hypergraph()
        self.assertEqual(s_diameter(a, 2), 2)
        self.assertEqual(s_diameter(a, 3), 2)
        self.assertEqual(s_diameter(a, 2, start=0, end=0), 2)
        self.assertEqual(s_diameter(a, 2, start=0, end=0, weight=True), 5)

    def test_s_components(self):
        a = self.get_hypergraph()
        self.assertEqual(list(s_components(a, 2)), [{"e5", "e3", "e1", "e2", "e4"}])
        self.assertEqual(list(s_components(a, 3)), [{"e1", "e3", "e4"}])
        self.assertEqual(list(s_components(a, 2, start=0, end=0)), [{"e1", "e2", "e3"}])

    # Node walks

    def test_node_shortest_s_walk(self):

        a = self.get_hypergraph()
        s_w = shortest_node_s_walk(a, 2, 1, 4)
        self.assertEqual(len(s_w), 2)
        s_w = shortest_node_s_walk(a, 3, 1, 4)
        self.assertEqual(len(s_w), 2)

        s_w = shortest_node_s_walk(a, 2, 1, 4, weight=True)
        self.assertEqual(len(s_w), 2)

        s_w = shortest_node_s_walk(a, 2, 1, 4, start=1, end=1, weight=True)
        self.assertEqual(len(s_w), 0)

        s_w = shortest_node_s_walk(a, 1, 1)
        self.assertDictEqual(s_w, {1: [1], 2: [1, 2], 3: [1, 3], 4: [1, 4]})

        s_w = shortest_node_s_walk(a, 2)
        self.assertDictEqual(
            s_w, {1: [1, 2, 3, 4], 2: [2, 1, 3, 4], 3: [3, 1, 2, 4], 4: [4, 1, 3, 2]}
        )

        s_w = shortest_node_s_walk(a, 3)
        self.assertDictEqual(s_w, {1: [1, 3, 4], 3: [3, 1, 4], 4: [4, 1, 3]})

    def test_node_s_distance(self):
        a = self.get_hypergraph()
        self.assertEqual(node_s_distance(a, 1, 1, 2), 1)
        self.assertEqual(node_s_distance(a, 2, 1, 2), 1)
        self.assertEqual(node_s_distance(a, 3, 1, 2), None)

        self.assertDictEqual(node_s_distance(a, 2, 1), {1: 0, 3: 1, 4: 1, 2: 1})
        self.assertDictEqual(
            node_s_distance(a, 2, 1, weight=True), {1: 0, 2: 2, 3: 3, 4: 3}
        )

        self.assertListEqual(
            list(node_s_distance(a, 2)),
            [
                (1, {1: 0, 4: 1, 3: 1, 2: 1}),
                (2, {2: 0, 1: 1, 3: 1, 4: 2}),
                (3, {3: 0, 1: 1, 4: 1, 2: 1}),
                (4, {4: 0, 1: 1, 3: 1, 2: 2}),
            ],
        )

    def test_node_average_s_distance(self):
        a = self.get_hypergraph()
        self.assertEqual(average_node_s_distance(a, 2), 1.1666666666666667)
        self.assertEqual(average_node_s_distance(a, 3), 1.0)
        self.assertEqual(
            average_node_s_distance(a, 2, start=0, end=0), 1.3333333333333333
        )

    def test_node_has_s_walk(self):
        a = self.get_hypergraph()
        self.assertEqual(has_node_s_walk(a, 2, 1, 2), True)
        self.assertEqual(has_node_s_walk(a, 8, 1, 2), False)
        self.assertEqual(has_node_s_walk(a, 2, 1, 2, start=0, end=0), True)
        self.assertEqual(has_node_s_walk(a, 2, 1, 4, start=1, end=1), False)

    def test_node_diameter(self):
        a = self.get_hypergraph()
        self.assertEqual(node_s_diameter(a, 2), 2)
        self.assertEqual(node_s_diameter(a, 3), 1)
        self.assertEqual(node_s_diameter(a, 2, start=0, end=0), 2)
        self.assertEqual(node_s_diameter(a, 2, start=0, end=0, weight=True), 4)

    def test_node_s_components(self):
        a = self.get_hypergraph()
        self.assertEqual(list(node_s_components(a, 2)), [{1, 2, 3, 4}])
        self.assertEqual(list(node_s_components(a, 3)), [{1, 3, 4}])
        self.assertEqual(list(node_s_components(a, 2, start=0, end=0)), [{1, 3, 4}])

    def test_is_path(self):
        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2], 0)
        a.add_hyperedge([1, 3], 0)
        a.add_hyperedge([1, 4], 0)
        self.assertEqual(is_s_path(a, ["e1", "e2", "e3"]), False)

        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2, 3, 5], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)
        self.assertEqual(is_s_path(a, ["e1", "e2", "e3"]), False)

        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2, 6, 7], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)
        a.add_hyperedge([3, 4, 5, 6, 7], 0)
        self.assertEqual(is_s_path(a, ["e1", "e2", "e3"]), True)

    def test_closed_s_walk(self):
        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2], 0)
        a.add_hyperedge([1, 3], 0)
        a.add_hyperedge([1, 4], 0)
        for w in closed_s_walk(a, 1, "e1"):
            self.assertEqual(is_s_path(a, w), False)

        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2, 3, 5], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)
        for w in closed_s_walk(a, 1, "e1"):
            self.assertEqual(is_s_path(a, w), False)

        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2, 6, 7], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)
        a.add_hyperedge([3, 4, 5, 6, 7], 0)
        for w in closed_s_walk(a, 1, "e1"):
            self.assertEqual(is_s_path(a, w), True)
