import unittest
from ash import ASH, NProfile
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

        a.add_node(1, start=0, end=0, attr_dict=NProfile(node_id=1, party="L", age=37, gender="M"))
        a.add_node(1, start=1, end=1, attr_dict=NProfile(node_id=1, party="R", age=37, gender="M"))
        a.add_node(2, start=0, end=0, attr_dict=NProfile(node_id=2, party="L", age=20, gender="F"))
        a.add_node(3, start=0, end=1, attr_dict=NProfile(node_id=3, party="L", age=11, gender="F"))
        a.add_node(4, start=0, end=1, attr_dict=NProfile(node_id=4, party="R", age=45, gender="M"))
        return a

    def test_hyperedge_profile_purity(self):
        a = self.get_hypergraph()

        for tid in a.temporal_snapshots_ids():
            hes = a.get_hyperedge_id_set(tid=tid)
            for he in hes:
                res = hyperedge_profile_purity(a, he, tid)
                self.assertListEqual(sorted(list(res.keys())), ['gender', 'party'])
