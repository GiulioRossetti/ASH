import unittest

from ash_model import ASH
from ash_model.paths import (
    average_s_distance,
    all_shortest_s_path_lengths,
    all_shortest_s_paths,
    all_simple_paths,
    closed_s_walk,
    has_s_walk,
    is_s_path,
    s_components,
    s_diameter,
    s_distance,
    shortest_s_path,
    shortest_s_walk,
)


class SWalksCase(unittest.TestCase):
    @staticmethod
    def get_hypergraph():
        a = ASH()
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([1, 4], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        a.add_hyperedge([1, 3, 4], 1)
        a.add_hyperedge([3, 4], 1)
        return a

    def _assert_same_vertices(self, got, want):
        """recursively check that the same keys are present at every level"""
        self.assertEqual(set(got.keys()), set(want.keys()))
        for k in got:
            if isinstance(want[k], dict):  # nested dict → recurse
                self.assertIsInstance(got[k], dict)
                self._assert_same_vertices(got[k], want[k])

    def _assert_min_lengths(self, got, want):
        """recursively check that every walk has the same (minimal) length"""
        for k, v in want.items():
            if isinstance(v, dict):  # nested dict → recurse
                self._assert_min_lengths(got[k], v)
            else:  # v is the reference walk
                self.assertEqual(len(got[k]), len(v))  # same length
                self.assertEqual(got[k][0], v[0])  # correct start vertex
                self.assertEqual(got[k][-1], k)  # correct end vertex

    def test_shortest_s_walk(self):

        a = self.get_hypergraph()
        s_w = shortest_s_walk(a, 2, "e1", "e2")
        self.assertEqual(len(s_w), 3)
        s_w = shortest_s_walk(a, 3, "e1", "e2")
        self.assertEqual(len(s_w), 0)

        s_w = shortest_s_walk(a, 2, "e1", "e2", weight=True)
        self.assertEqual(len(s_w), 3)

        # ---------- 1 source vertex -----------------------------------------------
        exp_1 = {
            "e1": ["e1"],
            "e3": ["e1", "e3"],
            "e4": ["e1", "e4"],
            "e2": ["e1", "e3", "e2"],
            "e5": ["e1", "e3", "e5"],
        }
        s_w = shortest_s_walk(a, 2, "e1")
        self._assert_same_vertices(s_w, exp_1)
        self._assert_min_lengths(s_w, exp_1)

        # ---------- all vertices, length ≤ 2 --------------------------------------
        exp_2 = {
            "e1": exp_1,
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
        }
        s_w = shortest_s_walk(a, 2)
        self._assert_same_vertices(s_w, exp_2)
        self._assert_min_lengths(s_w, exp_2)

        # ---------- length ≤ 3 -----------------------------------------------------
        exp_3 = {
            "e1": {"e1": ["e1"], "e3": ["e1", "e3"], "e4": ["e1", "e3", "e4"]},
            "e3": {"e3": ["e3"], "e1": ["e3", "e1"], "e4": ["e3", "e4"]},
            "e4": {"e4": ["e4"], "e3": ["e4", "e3"], "e1": ["e4", "e3", "e1"]},
        }
        s_w = shortest_s_walk(a, 3)
        self._assert_same_vertices(s_w, exp_3)
        self._assert_min_lengths(s_w, exp_3)

    def test_s_distance(self):
        a = self.get_hypergraph()
        self.assertEqual(s_distance(a, 1, "e1", "e2"), 1)
        self.assertEqual(s_distance(a, 2, "e1", "e2"), 2)
        self.assertEqual(s_distance(a, 3, "e1", "e2"), None)

        self.assertDictEqual(
            s_distance(a, 2, "e1"), {"e1": 0, "e2": 2, "e3": 1, "e4": 1, "e5": 2}
        )
        result = sorted(
            [(k, v) for k, v in s_distance(a, 2, "e1", weight=False).items()],
            key=lambda x: x[0],
        )
        check = sorted(
            {"e1": 0, "e2": 2, "e3": 1, "e4": 1, "e5": 2}.items(), key=lambda x: x[0]
        )
        self.assertListEqual(
            result,
            check,
        )
        # raise ValueError(f"{list(s_distance(a, 2))}")
        self.assertListEqual(
            sorted(s_distance(a, 2).items()),
            sorted(
                [
                    ("e1", {"e1": 0, "e2": 2, "e3": 1, "e4": 1, "e5": 2}),
                    ("e3", {"e1": 1, "e2": 1, "e3": 0, "e4": 1, "e5": 1}),
                    ("e4", {"e1": 1, "e2": 1, "e3": 1, "e4": 0, "e5": 1}),
                    ("e2", {"e1": 2, "e2": 0, "e3": 1, "e4": 1, "e5": 2}),
                    ("e5", {"e1": 2, "e2": 2, "e3": 1, "e4": 1, "e5": 0}),
                ]
            ),
        )

    def test_average_s_distance(self):
        a = self.get_hypergraph()
        self.assertEqual(average_s_distance(a, 2), 1.3)
        self.assertEqual(round(average_s_distance(a, 3), 2), 1.33)
        self.assertEqual(round(average_s_distance(a, 2, start=0, end=0), 2), 1.33)

    def test_has_s_walk(self):
        a = self.get_hypergraph()
        self.assertEqual(has_s_walk(a, 2, "e1", "e2"), True)
        self.assertEqual(has_s_walk(a, 8, "e1", "e2"), False)
        self.assertEqual(has_s_walk(a, 2, "e1", "e2", start=0, end=0), True)
        self.assertEqual(has_s_walk(a, 2, "e1", "e2", start=1, end=1), False)

    def test_diameter(self):
        a = self.get_hypergraph()
        # def get_hypergraph():
        # a = ASH()
        # a.add_hyperedge([1, 2, 3], 0) e1
        # a.add_hyperedge([1, 4], 0) e2
        # a.add_hyperedge([1, 2, 3, 4], 0) e3
        # a.add_hyperedge([1, 3, 4], 1) e4
        # a.add_hyperedge([3, 4], 1) e5
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
        s_w = shortest_s_walk(a, 2, 1, 4, edge=False)
        self.assertEqual(len(s_w), 2)
        s_w = shortest_s_walk(a, 3, 1, 4, edge=False)
        self.assertEqual(len(s_w), 2)

        s_w = shortest_s_walk(a, 2, 1, 4, weight=True, edge=False)
        self.assertEqual(len(s_w), 2)

        s_w = shortest_s_walk(a, 2, 1, 4, start=1, end=1, weight=True, edge=False)
        self.assertEqual(len(s_w), 0)

        s_w = shortest_s_walk(a, 1, 1, edge=False)
        expected = {1: [1], 2: [1, 2], 3: [1, 3], 4: [1, 4]}
        for k, v in s_w.items():
            self.assertCountEqual(v, expected[k])

        s_w = shortest_s_walk(a, 2, edge=False)
        expected = {1: [1, 2, 3, 4], 2: [2, 1, 3, 4], 3: [3, 1, 2, 4], 4: [4, 1, 3, 2]}
        for k, v in s_w.items():
            self.assertCountEqual(v, expected[k])

        s_w = shortest_s_walk(a, 3, edge=False)
        expected = {1: [1, 3, 4], 3: [3, 1, 4], 4: [4, 1, 3]}
        for k, v in s_w.items():
            self.assertCountEqual(v, expected[k])

    def test_node_s_distance(self):
        a = self.get_hypergraph()
        self.assertEqual(s_distance(a, 1, 1, 2, edge=False), 1)
        self.assertEqual(s_distance(a, 2, 1, 2, edge=False), 1)
        self.assertEqual(s_distance(a, 3, 1, 2, edge=False), None)

        self.assertDictEqual(
            s_distance(a, s=2, fr=1, edge=False), {1: 0, 3: 1, 4: 1, 2: 1}
        )
        self.assertDictEqual(
            s_distance(a, 2, 1, weight=True, edge=False), {1: 0, 2: 2, 3: 3, 4: 3}
        )
        expected = {
            1: {1: 0, 4: 1, 3: 1, 2: 1},
            2: {2: 0, 1: 1, 3: 1, 4: 2},
            3: {3: 0, 1: 1, 4: 1, 2: 1},
            4: {4: 0, 1: 1, 3: 1, 2: 2},
        }

        result = s_distance(a, 2, edge=False)
        for src, dist_dict in result.items():
            for trg, dist in dist_dict.items():
                real = expected[src][trg]
                self.assertEqual(
                    dist,
                    real,
                    f"Distance from {src} to {trg} should be {real}, got {dist}",
                )

    def test_node_average_s_distance(self):
        a = self.get_hypergraph()
        self.assertEqual(average_s_distance(a, 2, edge=False), 1.1666666666666667)
        self.assertEqual(average_s_distance(a, 3, edge=False), 1.0)
        self.assertEqual(
            average_s_distance(a, 2, start=0, end=0, edge=False), 1.3333333333333333
        )

    def test_node_has_s_walk(self):
        a = self.get_hypergraph()
        self.assertEqual(has_s_walk(a, 2, 1, 2, edge=False), True)
        self.assertEqual(has_s_walk(a, 8, 1, 2, edge=False), False)
        self.assertEqual(has_s_walk(a, 2, 1, 2, start=0, end=0, edge=False), True)
        self.assertEqual(has_s_walk(a, 2, 1, 4, start=1, end=1, edge=False), False)

    def test_node_diameter(self):
        a = self.get_hypergraph()
        self.assertEqual(s_diameter(a, 2, edge=False), 2)
        self.assertEqual(s_diameter(a, 3, edge=False), 1)
        self.assertEqual(s_diameter(a, 2, start=0, end=0, edge=False), 2)
        self.assertEqual(s_diameter(a, 2, start=0, end=0, weight=True, edge=False), 4)

    def test_node_s_components(self):
        a = self.get_hypergraph()
        self.assertEqual(list(s_components(a, 2, edge=False)), [{1, 2, 3, 4}])
        self.assertEqual(list(s_components(a, 3, edge=False)), [{1, 3, 4}])
        self.assertEqual(
            list(s_components(a, 2, start=0, end=0, edge=False)), [{1, 3, 4}]
        )

    def test_is_path(self):
        a = ASH()
        a.add_hyperedge([1, 2], 0)
        a.add_hyperedge([1, 3], 0)
        a.add_hyperedge([1, 4], 0)
        self.assertEqual(is_s_path(a, ["e1", "e2", "e3"]), False)

        a = ASH()
        a.add_hyperedge([1, 2, 3, 5], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)
        self.assertEqual(is_s_path(a, ["e1", "e2", "e3"]), False)

        a = ASH()
        a.add_hyperedge([1, 2, 6, 7], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)
        a.add_hyperedge([3, 4, 5, 6, 7], 0)
        self.assertEqual(is_s_path(a, ["e1", "e2", "e3"]), True)

    def test_closed_s_walk(self):
        a = ASH()
        a.add_hyperedge([1, 2], 0)
        a.add_hyperedge([1, 3], 0)
        a.add_hyperedge([1, 4], 0)
        for w in closed_s_walk(a, 1, "e1"):
            self.assertEqual(is_s_path(a, w), False)

        a = ASH()
        a.add_hyperedge([1, 2, 3, 5], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)
        for w in closed_s_walk(a, 1, "e1"):
            self.assertEqual(is_s_path(a, w), False)

        a = ASH()
        a.add_hyperedge([1, 2, 6, 7], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)
        a.add_hyperedge([3, 4, 5, 6, 7], 0)
        for w in closed_s_walk(a, 1, "e1"):
            self.assertEqual(is_s_path(a, w), True)

    def test_all_simple_s_paths(self):
        a = ASH()
        a.add_hyperedge([1, 2, 6, 7], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)
        a.add_hyperedge([3, 4, 5, 6, 7], 0)

        for p in all_simple_paths(a, 1, "e1", "e2"):
            self.assertIsInstance(p, list)

    def test_s_shortest_paths(self):
        a = ASH()
        a.add_hyperedge([1, 2, 6, 7], 0)
        a.add_hyperedge([1, 2, 3, 4, 5], 0)
        a.add_hyperedge([3, 4, 5, 6, 7], 0)

        for p in shortest_s_path(a, 1, "e1", "e2"):
            self.assertEqual(sorted(p), ["e1", "e2"])

        for k, v in all_shortest_s_paths(a, 1, "e1").items():
            self.assertIsInstance(k, tuple)
            self.assertIsInstance(v, list)

        for k, v in all_shortest_s_paths(a, 1).items():
            self.assertIsInstance(k, tuple)
            self.assertIsInstance(v, list)

        for k, v in all_shortest_s_path_lengths(a, 1).items():
            self.assertEqual(len(v), 3)
