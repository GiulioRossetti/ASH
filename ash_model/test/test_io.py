import os
import gzip
import json
import tempfile
import unittest

import numpy as np

from ash_model import ASH, NProfile
from ash_model.readwrite import (
    write_profiles_to_csv,
    read_profiles_from_csv,
    write_profiles_to_jsonl,
    read_profiles_from_jsonl,
    write_sh_to_csv,
    read_sh_from_csv,
    write_ash_to_json,
    read_ash_from_json,
    read_hif,
    write_hif,
)


class IOTestCase(unittest.TestCase):
    def setUp(self):
        # a small hypergraph with node‐ and edge‐attributes and multi‐span edges
        self.a = ASH()
        # hyperedge e1 present at t=0..1, weight=3
        self.a.add_hyperedge([1, 2], start=0, end=1, weight=3)
        # hyperedge e2 present only at t=2
        self.a.add_hyperedge([2, 3, 4], start=2, weight=5)
        # nodes with profiles changing across time
        self.a.add_node(1, start=0, end=1, attr_dict=NProfile(1, color="red", score=10))
        self.a.add_node(
            1, start=2, end=2, attr_dict=NProfile(1, color="blue", score=20)
        )
        self.a.add_node(
            2, start=0, end=2, attr_dict=NProfile(2, color="green", score=5)
        )

    def _roundtrip_file(self, write_fn, read_fn, *args, **kwargs):
        # helper: write to temp, read back, return result
        tf = tempfile.NamedTemporaryFile(delete=False)
        tf.close()
        try:
            write_fn(self.a, tf.name, *args, **kwargs)
            return read_fn(tf.name, *args, **kwargs)
        finally:
            os.remove(tf.name)

    def test_profiles_csv_roundtrip_default(self):
        # write and read back
        res = self._roundtrip_file(write_profiles_to_csv, read_profiles_from_csv)
        # Should have exactly nodes 1 and 2:
        self.assertCountEqual(res.keys(), [1, 2, 3, 4])
        # Node 1: tids 0,1,2
        self.assertCountEqual(res[1].keys(), [0, 1, 2])
        # Check types: score remains int, color is str
        p0 = res[1][0]
        attrs = p0.get_attributes()
        self.assertIsInstance(attrs["score"], int)
        self.assertIsInstance(attrs["color"], str)
        # Check actual values
        self.assertEqual(attrs["color"], "red")
        self.assertEqual(attrs["score"], 10)

    def test_profiles_csv_custom_delimiter_and_header(self):
        # use semicolon
        tf = tempfile.NamedTemporaryFile(delete=False)
        tf.close()
        try:
            write_profiles_to_csv(self.a, tf.name, delimiter=";")
            # check header line
            with open(tf.name) as f:
                header = f.readline().strip().split(";")
            # must start with node_id, tid, then sorted attribute names
            self.assertEqual(header[0:2], ["node_id", "tid"])
            self.assertEqual(set(header[2:]), {"color", "score"})
            # now roundtrip
            res = read_profiles_from_csv(tf.name, delimiter=";")
            self.assertCountEqual(res.keys(), [1, 2, 3, 4])
        finally:
            os.remove(tf.name)

    def test_profiles_csv_empty_graph(self):
        empty = ASH()
        tf = tempfile.NamedTemporaryFile(delete=False)
        tf.close()
        try:
            write_profiles_to_csv(empty, tf.name)
            # file should either not exist or only header
            lines = open(tf.name).read().splitlines()
            # no header written (no nodes)
            self.assertTrue(len(lines) in (0, 1))
        finally:
            os.remove(tf.name)

    def test_profiles_jsonl_roundtrip(self):
        # uncompressed
        res = self._roundtrip_file(
            write_profiles_to_jsonl, read_profiles_from_jsonl, compress=False
        )
        self.assertCountEqual(res.keys(), [1, 2, 3, 4])
        # compressed
        res2 = self._roundtrip_file(
            write_profiles_to_jsonl, read_profiles_from_jsonl, compress=True
        )
        self.assertCountEqual(res2.keys(), [1, 2, 3, 4])

    def test_sh_csv_roundtrip_and_header(self):
        # write hyperedges
        tf = tempfile.NamedTemporaryFile(delete=False)
        tf.close()
        try:
            write_sh_to_csv(self.a, tf.name)
            # header must be present
            with open(tf.name) as f:
                hdr = f.readline().strip()
            self.assertEqual(hdr, "nodes\tstart,end")
            # roundtrip
            b = read_sh_from_csv(tf.name)
            # same hyperedges
            sets_a = [
                frozenset(self.a.get_hyperedge_nodes(h)) for h in self.a.hyperedges()
            ]
            sets_b = [frozenset(b.get_hyperedge_nodes(h)) for h in b.hyperedges()]

            self.assertCountEqual(sets_b, sets_a)
            # presence times match
            sets_a = [
                frozenset(self.a.hyperedge_presence(h)) for h in self.a.hyperedges()
            ]
            sets_b = [frozenset(b.hyperedge_presence(h)) for h in b.hyperedges()]
            self.assertCountEqual(sets_b, sets_a)

        finally:
            os.remove(tf.name)

    def test_ash_json_roundtrip_uncompressed_and_compressed(self):
        # uncompressed
        for compress in (False, True):
            # write
            tf = tempfile.NamedTemporaryFile(delete=False)
            tf.close()
            try:
                write_ash_to_json(self.a, tf.name, compress=compress)
                b = read_ash_from_json(tf.name, compress=compress)
                # same nodes
                self.assertCountEqual(b.nodes(), self.a.nodes())
                # same hyperedges
                self.assertCountEqual(b.hyperedges(), self.a.hyperedges())
                # same snapshots
                self.assertEqual(
                    sorted(b.temporal_snapshots_ids()),
                    sorted(self.a.temporal_snapshots_ids()),
                )
                # edge attributes (weight) preserved
                weights_a = [
                    self.a.get_hyperedge_weight(h) for h in self.a.hyperedges()
                ]
                weights_b = [b.get_hyperedge_weight(h) for h in b.hyperedges()]
                self.assertCountEqual(weights_a, weights_b)

                # node profiles equal
                for n in self.a.nodes():
                    for t in self.a.node_presence(n):
                        p1 = self.a.get_node_profile(n, t)
                        p2 = b.get_node_profile(n, t)
                        self.assertEqual(p1, p2)
            finally:
                os.remove(tf.name)

    def test_read_write_missing_file_raises(self):
        # read functions should raise if file not found
        with self.assertRaises(FileNotFoundError):
            read_profiles_from_csv("no_such_file.csv")
        with self.assertRaises(FileNotFoundError):
            read_profiles_from_jsonl("no_such_file.jsonl")
        with self.assertRaises(FileNotFoundError):
            read_sh_from_csv("no_such_file.csv")
        with self.assertRaises(FileNotFoundError):
            read_ash_from_json("no_such_file.json")

    def test_hif_ash(self):
        import fastjsonschema
        import json

        base = os.path.join(os.path.dirname(__file__), "hif_data")
        schema_path = os.path.join(base, "hif_schema.json")
        with open(schema_path) as f:
            schema = json.load(f)
        validator = fastjsonschema.compile(schema)

        # write
        tf = tempfile.NamedTemporaryFile(delete=False)
        tf.close()

        write_hif(self.a, tf.name)

        with open(tf.name) as tmp:
            data = json.loads(tmp.read())

        try:
            validator(data)
        except Exception:
            raise ValueError("invalid hif")

        a2 = read_hif(tf.name)

        # check equivalence
        try:
            # same nodes
            self.assertCountEqual(a2.nodes(), self.a.nodes())

            # same hyperedges (by node sets, since edge IDs may differ)
            sets_a = [
                frozenset(self.a.get_hyperedge_nodes(h)) for h in self.a.hyperedges()
            ]
            sets_a2 = [frozenset(a2.get_hyperedge_nodes(h)) for h in a2.hyperedges()]
            self.assertCountEqual(sets_a2, sets_a)

            # same temporal snapshots
            self.assertEqual(
                sorted(a2.temporal_snapshots_ids()),
                sorted(self.a.temporal_snapshots_ids()),
            )

            # check hyperedge presence times match for corresponding edges
            for h1 in self.a.hyperedges():
                nodes1 = frozenset(self.a.get_hyperedge_nodes(h1))
                presence1 = frozenset(self.a.hyperedge_presence(h1))

                # find corresponding edge in a2
                found = False
                for h2 in a2.hyperedges():
                    nodes2 = frozenset(a2.get_hyperedge_nodes(h2))
                    if nodes1 == nodes2:
                        presence2 = frozenset(a2.hyperedge_presence(h2))
                        if presence1 == presence2:
                            # check edge attributes match
                            weight1 = self.a.get_hyperedge_weight(h1)
                            weight2 = a2.get_hyperedge_weight(h2)
                            self.assertEqual(weight1, weight2)
                            found = True
                            break

                self.assertTrue(found, f"Could not find matching edge for {nodes1}")

            # check node profiles are preserved
            for n in self.a.nodes():
                for t in self.a.node_presence(n):
                    if n in a2.nodes() and t in a2.node_presence(n):
                        p1 = self.a.get_node_profile(n, t)
                        p2 = a2.get_node_profile(n, t)
                        self.assertEqual(p1, p2)

        finally:
            os.remove(tf.name)

    def test_hif_static(self):
        base = os.path.join(os.path.dirname(__file__), "hif_data")
        data_path = os.path.join(base, "email-enron.hif")
        h = read_hif(data_path)
        self.assertIsInstance(h, ASH)
        self.assertListEqual(h.temporal_snapshots_ids(), [0])
        self.assertGreaterEqual(h.number_of_nodes(), 143)
        self.assertGreaterEqual(h.number_of_hyperedges(), 1436)


if __name__ == "__main__":
    unittest.main()
