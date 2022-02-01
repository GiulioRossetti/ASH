import unittest
from ash import ASH, NProfile
from ash.measures import *
import json
import os


class HyperConformityTestCase(unittest.TestCase):
    @staticmethod
    def get_hypergraph():
        a = ASH()
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([1, 4], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        a.add_hyperedge([1, 3, 4], 0)
        a.add_hyperedge([3, 4], 0)

        a.add_node(
            1,
            start=0,
            end=0,
            attr_dict=NProfile(node_id=1, party="L", age=37, gender="M", rank="one"),
        )
        a.add_node(
            2,
            start=0,
            end=0,
            attr_dict=NProfile(node_id=2, party="L", age=20, gender="F", rank="one"),
        )
        a.add_node(
            3,
            start=0,
            end=0,
            attr_dict=NProfile(node_id=3, party="L", age=11, gender="F", rank="two"),
        )
        a.add_node(
            4,
            start=0,
            end=0,
            attr_dict=NProfile(node_id=4, party="R", age=45, gender="M", rank="three"),
        )
        return a

    def test_hyper_conformity(self):
        a = self.get_hypergraph()

        full_res = hyper_conformity(
            h=a,
            alphas=list(np.arange(1, 4, 0.2)),
            labels=["party", "gender"],
            profile_size=1,
        )

        with open(f"conformity.json", "w") as o:
            json.dump(full_res, o)

        for res in full_res:
            for k, v in res.items():
                for z, t in v.items():
                    for _, val in t.items():
                        self.assertTrue(-1 <= val <= 1)

        os.remove("conformity.json")

        hierarchy = {"rank": {"one": 1, "two": 2, "three": 3}}
        full_res = hyper_conformity(
            h=a,
            alphas=list(np.arange(1, 4, 0.2)),
            labels=["party", "gender", "rank"],
            profile_size=1,
            hierarchies=hierarchy,
        )

        with open(f"conformity_hierarchy.json", "w") as o:
            json.dump(full_res, o)

        for res in full_res:
            for k, v in res.items():
                for z, t in v.items():
                    for _, val in t.items():
                        self.assertTrue(-1 <= val <= 1)

        os.remove("conformity_hierarchy.json")
