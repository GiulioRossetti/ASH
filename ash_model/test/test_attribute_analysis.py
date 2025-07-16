import unittest
import math
import numpy as np

from ash_model.measures.attribute_analysis import (
    hyperedge_profile_purity,
    hyperedge_profile_entropy,
    star_profile_entropy,
    star_profile_homogeneity,
    average_group_degree,
    attribute_consistency,
)
from ash_model.measures import ASH, NProfile


class AttributeAnalysisCase(unittest.TestCase):
    @staticmethod
    def get_hypergraph():
        a = ASH()
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

    def test_hyperedge_profile_purity(self):
        a = self.get_hypergraph()
        # find the hyperedge [3,4] at tid=1
        hes = list(a.hyperedges(start=1))
        he = next(h for h in hes if set(a.get_hyperedge_nodes(h)) == {3, 4})

        purity = hyperedge_profile_purity(a, he, 1)
        # for party, labels ['L','R'] => most common 'L'
        # for gender, labels ['F','M'] => most common 'F'
        self.assertEqual(purity["party"], {"L": 0.5})
        self.assertEqual(purity["gender"], {"F": 0.5})

        # keys should be exactly these two
        self.assertListEqual(sorted(purity.keys()), ["gender", "party"])

    def test_hyperedge_profile_entropy(self):
        a = self.get_hypergraph()
        hes = list(a.hyperedges(start=1))
        he = next(h for h in hes if set(a.get_hyperedge_nodes(h)) == {3, 4})

        ent = hyperedge_profile_entropy(a, he, tid=1)
        # for ['L','R'] with base=2 ⇒ entropy=1.0
        self.assertAlmostEqual(ent["party"], 1.0)
        self.assertAlmostEqual(ent["gender"], 1.0)

        self.assertListEqual(sorted(ent.keys()), ["gender", "party"])

    def test_star_profile_entropy(self):
        a = self.get_hypergraph()
        # collapse method at tid=0, node=1: star nodes {1,2,3,4}
        res = star_profile_entropy(a, node_id=1, tid=0, method="collapse")
        # party labels ['L','L','L','R']
        p = -(2 / 3 * math.log(2 / 3, 2) + 1 / 3 * math.log(1 / 3, 2))
        self.assertAlmostEqual(res["party"], p, places=6)
        # gender labels ['M','F','F','M']

        self.assertAlmostEqual(res["gender"], 0.918, places=3)

    def test_star_profile_homogeneity(self):
        a = self.get_hypergraph()
        # collapse method at tid=0, node=1: 3 hyperedges.  attributes: gender(1)='M', party(1)='L'
        res = star_profile_homogeneity(a, node_id=1, tid=0, method="collapse")
        # party: 2 of the 3 neighbors have the same party 'L', 1 has 'R' ⇒ homogeneity=2/3
        self.assertAlmostEqual(res["party"], 2 / 3)
        # gender: 1 of the 3 neighbors have the same gender 'M', 2 has 'F' ⇒ homogeneity= 1/3
        self.assertAlmostEqual(res["gender"], 1 / 3)

    def test_average_group_degree(self):
        a = self.get_hypergraph()
        out = average_group_degree(a, tid=0)
        # at tid=0, degrees are:
        #   party L: nodes 1,2,3 have degrees 3,2,2 ⇒ mean 7/3
        #   party R: node 4 has degree 2 ⇒ mean 2
        self.assertAlmostEqual(out["party"]["L"], 7 / 3)
        self.assertAlmostEqual(out["party"]["R"], 2.0)
        # gender M: nodes 1,4 => 3,2 ⇒ 2.5
        # gender F: nodes 2,3 => 2,2 ⇒ 2.0
        self.assertAlmostEqual(out["gender"]["M"], 2.5)
        self.assertAlmostEqual(out["gender"]["F"], 2.0)

    def test_attribute_consistency(self):
        a = self.get_hypergraph()
        cons = attribute_consistency(a)
        # Node 1: party changed L→R ⇒ consistency=0; gender stayed M⇒1
        self.assertAlmostEqual(cons[1]["party"], 0.0)
        self.assertAlmostEqual(cons[1]["gender"], 1.0)
        # Node 2 only at tid=0 ⇒ consistency=1 for both
        self.assertAlmostEqual(cons[2]["party"], 1.0)
        self.assertAlmostEqual(cons[2]["gender"], 1.0)
        # Node 3 (L→L, F→F) ⇒ both 1
        self.assertAlmostEqual(cons[3]["party"], 1.0)
        self.assertAlmostEqual(cons[3]["gender"], 1.0)
        # Node 4 (R→R, M→M) ⇒ both 1
        self.assertAlmostEqual(cons[4]["party"], 1.0)
        self.assertAlmostEqual(cons[4]["gender"], 1.0)
