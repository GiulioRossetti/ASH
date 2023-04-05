import unittest

from ash.measures import *


class AttributeAnalysisCase(unittest.TestCase):
    @staticmethod
    def get_hypergraph():
        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([1, 4], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        a.add_hyperedge([1, 3, 4], 1)
        a.add_hyperedge([3, 4], 1)

        a.add_node(
            1,
            start=0,
            end=0,
            attr_dict=NProfile(node_id=1, party="L", age=37, gender="M"),
        )
        a.add_node(
            1,
            start=1,
            end=1,
            attr_dict=NProfile(node_id=1, party="R", age=37, gender="M"),
        )
        a.add_node(
            2,
            start=0,
            end=0,
            attr_dict=NProfile(node_id=2, party="L", age=20, gender="F"),
        )
        a.add_node(
            3,
            start=0,
            end=1,
            attr_dict=NProfile(node_id=3, party="L", age=11, gender="F"),
        )
        a.add_node(
            4,
            start=0,
            end=1,
            attr_dict=NProfile(node_id=4, party="R", age=45, gender="M"),
        )
        return a

    def test_profiles(self):
        a = ASH(hedge_removal=True)
        a.add_node(1, start=1, end=5, attr_dict=NProfile(1, party="L", age=37))
        a.add_node(2, start=1, end=5, attr_dict=NProfile(2, party="L", age=20))
        a.add_node(3, start=1, end=5, attr_dict=NProfile(3, party="L", age=11))
        a.add_node(4, start=1, end=5, attr_dict=NProfile(4, party="R", age=45))
        a.add_hyperedge([1, 2, 3, 4], 1, 4)

        self.assertEqual(
            hyperedge_aggregate_node_profile(a, "e1", 1).get_attributes(),
            {"age": 28.25},
        )
        self.assertEqual(
            hyperedge_aggregate_node_profile(a, "e1", 1).get_statistic("age", "std"),
            {"std": 13.442005058770064},
        )
        self.assertEqual(
            hyperedge_most_frequent_node_attribute_value(a, "e1", "party", 1), {"L": 3}
        )

    def test_hyperedge_profile_purity(self):
        a = self.get_hypergraph()

        for tid in a.temporal_snapshots_ids():
            hes = a.get_hyperedge_id_set(tid=tid)
            for he in hes:
                res = hyperedge_profile_purity(a, he, tid)
                self.assertListEqual(sorted(list(res.keys())), ["gender", "party"])
            res = average_hyperedge_profile_purity(a, tid=tid, by_label=False)
            self.assertListEqual(sorted(list(res.keys())), ["gender", "party"])
            res = average_hyperedge_profile_purity(a, tid=tid, by_label=True)
            self.assertListEqual(sorted(list(res.keys())), ["gender", "party"])

    def test_hyperedge_profile_entropy(self):
        a = self.get_hypergraph()

        for tid in a.temporal_snapshots_ids():
            hes = a.get_hyperedge_id_set(tid=tid)
            for he in hes:
                res = hyperedge_profile_entropy(a, he, tid)
                self.assertListEqual(sorted(list(res.keys())), ["gender", "party"])
            res = average_hyperedge_profile_entropy(a, tid)
            self.assertListEqual(sorted(list(res.keys())), ["gender", "party"])

    def test_star_profile_homogeneity(self):
        a = self.get_hypergraph()

        hom = star_profile_homogeneity(a, node_id=1, tid=0)
        self.assertEqual(hom, {"gender": 0.6666666666666666, "party": 1.0})
        for tid in a.temporal_snapshots_ids():
            for method in ["aggregate", "collapse"]:
                for n in a.get_node_set(tid=tid):
                    res = star_profile_homogeneity(a, n, tid, method)
                    self.assertListEqual(sorted(list(res.keys())), ["gender", "party"])

                res = average_star_profile_homogeneity(
                    a, tid=tid, method=method, by_label=False
                )
                self.assertListEqual(sorted(list(res.keys())), ["gender", "party"])
                res = average_star_profile_homogeneity(
                    a, tid=tid, method=method, by_label=True
                )
                self.assertListEqual(sorted(list(res.keys())), ["gender", "party"])

    def test_star_profile_entropy(self):
        a = self.get_hypergraph()

        ent = star_profile_entropy(a, node_id=1, tid=0)
        self.assertEqual(ent, {"party": 0, "gender": 0.9182958340544896})
        for tid in a.temporal_snapshots_ids():
            nodes = a.get_node_set(tid=tid)
            for method in ["aggregate", "collapse"]:
                for n in nodes:
                    res = star_profile_entropy(a, n, tid, method)
                    self.assertListEqual(sorted(list(res.keys())), ["gender", "party"])
                res = average_star_profile_entropy(a, tid, method=method)
                self.assertListEqual(sorted(list(res.keys())), ["gender", "party"])

    def test_consistency(self):
        a = self.get_hypergraph()
        res = consistency(a)

        self.assertIsInstance(res, dict)
        self.assertEqual(res[1]["party"], 0)
        self.assertEqual(res[1]["gender"], 1)

    def test_average_group_degree(self):
        a = self.get_hypergraph()

        res = average_group_degree(a, tid=0)
        self.assertDictEqual(
            res,
            {
                "party": {"R": 2.0, "L": 2.3333333333333335},
                "gender": {"M": 2.5, "F": 2.0},
            },
        )

        res = average_group_degree(a, tid=0, hyperedge_size=2)
        self.assertDictEqual(
            res,
            {
                "party": {"R": 1.0, "L": 0.3333333333333333},
                "gender": {"M": 1.0, "F": 0.0},
            },
        )
        for tid in a.temporal_snapshots_ids():
            res = average_group_degree(a, tid=tid)
            self.assertIsInstance(res, dict)
