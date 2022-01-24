import unittest
from ash import ASH, NProfile
from ash.readwrite import *
import os


class IOTestCase(unittest.TestCase):

    @staticmethod
    def get_hypergraph():
        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([1, 4], 0)
        a.add_hyperedge([1, 2, 3, 4], 0)
        a.add_hyperedge([1, 3, 4], 1)
        a.add_hyperedge([3, 4], 1)

        a.add_node(1, start=0, end=0, attr_dict=NProfile(node_id=1, party="L", age=37))
        a.add_node(1, start=1, end=1, attr_dict=NProfile(node_id=1, party="R", age=37))
        a.add_node(2, start=0, end=0, attr_dict=NProfile(node_id=2, party="L", age=20))
        a.add_node(3, start=0, end=1, attr_dict=NProfile(node_id=3, party="L", age=11))
        a.add_node(4, start=0, end=1, attr_dict=NProfile(node_id=4, party="R", age=45))
        return a

    def test_read_write_profile_csv(self):
        a = self.get_hypergraph()

        write_profiles_to_csv(a, "test_profiles.csv")
        res = read_profiles_from_csv("test_profiles.csv")

        for k, v in res.items():
            self.assertEqual(a.has_node(k), True)
            for t, p in v.items():
                self.assertEqual(a.has_node(k, t), True)
                profile = a.get_node_profile(k, t)
                self.assertEqual(a.get_node_profile(k, t) == profile, True)
        os.remove("test_profiles.csv")

    def test_read_write_profile_json(self):
        a = self.get_hypergraph()

        write_profiles_to_jsonl(a, "test_profiles.json")
        res = read_profiles_from_jsonl("test_profiles.json")

        for k, v in res.items():
            self.assertEqual(a.has_node(k), True)
            for t, p in v.items():
                self.assertEqual(a.has_node(k, t), True)
                profile = a.get_node_profile(k, t)
                self.assertEqual(a.get_node_profile(k, t) == profile, True)

        os.remove("test_profiles.json")

    def test_read_write_sh_csv(self):
        a = self.get_hypergraph()

        write_sh_to_csv(a, "test_sh.csv")
        b = read_sh_from_csv("test_sh.csv")

        self.assertEqual(a.get_hyperedge_id_set(), b.get_hyperedge_id_set())
        self.assertEqual(sorted(list(a.node_iterator())), sorted(list(b.node_iterator())))
        self.assertEqual(sorted(a.temporal_snapshots_ids()), sorted(b.temporal_snapshots_ids()))
        for tid in a.temporal_snapshots_ids():
            self.assertEqual(len(a.get_hyperedge_id_set(tid)), len(b.get_hyperedge_id_set(tid)))
            self.assertEqual(len(list(a.node_iterator(tid))), len(list(b.node_iterator(tid))))

        os.remove("test_sh.csv")

    def test_read_write_ash(self):
        a = self.get_hypergraph()
        write_ash_to_json(a, "test_ash.json")
        b = read_ash_from_json("test_ash.json")

        self.assertEqual(a.get_hyperedge_id_set(), b.get_hyperedge_id_set())
        self.assertEqual(sorted(list(a.node_iterator())), sorted(list(b.node_iterator())))
        self.assertEqual(sorted(a.temporal_snapshots_ids()), sorted(b.temporal_snapshots_ids()))
        for tid in a.temporal_snapshots_ids():
            self.assertEqual(len(a.get_hyperedge_id_set(tid)), len(b.get_hyperedge_id_set(tid)))
            self.assertEqual(len(list(a.node_iterator(tid))), len(list(b.node_iterator(tid))))

        for node in a.node_iterator():
            tids = a.get_node_presence(node)
            for tid in tids:
                p1 = a.get_node_profile(node, tid)
                p2 = b.get_node_profile(node, tid)
                self.assertEqual(p1 == p2, True)

        os.remove("test_ash.json")
