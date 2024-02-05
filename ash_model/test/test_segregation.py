import unittest

from ash_model.measures import *


class SegregationCase(unittest.TestCase):
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

    def test_hyperedge_affinity_count(self):
        a = self.get_hypergraph()

        internal, external, other = hyperedge_affinity_count(a, 0, "party", "linear")
        self.assertListEqual(sorted(list(internal.values())), [0, 0.75])
        self.assertListEqual(sorted(list(external.values())), [2, 2.25])
        self.assertListEqual(sorted(list(other.values())), [1])

        internal, external, other = hyperedge_affinity_count(a, 0, "party", "majority")
        self.assertListEqual(sorted(list(internal.values())), [0, 1])
        self.assertListEqual(sorted(list(external.values())), [2, 2])
        self.assertListEqual(sorted(list(other.values())), [1])

        internal, external, other = hyperedge_affinity_count(a, 0, "party", "strict")
        self.assertListEqual(sorted(list(internal.values())), [0, 1])
        self.assertListEqual(sorted(list(external.values())), [2, 2])
        self.assertListEqual(sorted(list(other.values())), [1])

    def test_ei_index(self):
        a = self.get_hypergraph()
        res = ei_index(a, 0, criterion="linear")

        for ct in ["linear", "majority", "strict"]:

            for tid in a.temporal_snapshots_ids():
                res = ei_index(a, tid, criterion=ct)
                self.assertListEqual(sorted(list(res.keys())), ["gender", "party"])

    def test_guptas_Q(self):
        a = self.get_hypergraph()
        res = ei_index(a, 0, criterion="linear")

        for ct in ["linear", "majority", "strict"]:

            for tid in a.temporal_snapshots_ids():
                res = guptas_Q(a, tid, criterion=ct)
                self.assertListEqual(sorted(list(res.keys())), ["gender", "party"])
