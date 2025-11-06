import unittest

import numpy as np
from scipy import sparse

from ash_model import ASH
from ash_model.paths.randwalks import (
    random_walk_probabilities,
    random_walks,
    time_respecting_random_walks,
)


class RandomWalksTestCase(unittest.TestCase):
    def setUp(self):
        # Build a tiny hypergraph with one 3-node edge and one 2-node edge
        self.h = ASH()
        self.h.add_hyperedge([1, 2, 3], start=0)  # e1
        self.h.add_hyperedge([2, 4], start=0)  # e2

    def test_random_walk_probabilities_exact(self):
        # With s=1 (default), nodes are connected if they co-occur in at least 1 hyperedge.
        # The weight is the number of co-occurrences (not len(edge)-1).
        # For hyperedge [1,2,3]: each pair co-occurs 1 time
        # For hyperedge [2,4]: pair (2,4) co-occurs 1 time
        T, mapping = random_walk_probabilities(self.h)
        self.assertIsInstance(T, sparse.csr_matrix)
        dense = T.toarray()
        # Build expected using mapping order dynamically
        idx_of = mapping  # node -> row/col index
        n = len(idx_of)
        expected = np.zeros((n, n), dtype=float)
        # from 1: neighbors 2,3 with 1 co-occurrence each
        expected[idx_of[1], idx_of[2]] += 1
        expected[idx_of[1], idx_of[3]] += 1
        # from 2: neighbors 1,3 (1 each) and 4 (1)
        expected[idx_of[2], idx_of[1]] += 1
        expected[idx_of[2], idx_of[3]] += 1
        expected[idx_of[2], idx_of[4]] += 1
        # from 3: neighbors 1,2 (1 each)
        expected[idx_of[3], idx_of[1]] += 1
        expected[idx_of[3], idx_of[2]] += 1
        # from 4: neighbor 2 (1)
        expected[idx_of[4], idx_of[2]] += 1
        # row-normalize
        row_sums = expected.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        expected = expected / row_sums
        np.testing.assert_allclose(dense, expected, rtol=1e-6)

    def test_random_walks_shape_and_ids(self):
        # generate 2 walks per start node, length 4
        walks = random_walks(
            self.h,
            num_walks=2,
            walk_length=4,
            p=1.0,
            q=1.0,
            start=None,
            end=None,
            threads=1,
        )
        # we have 4 nodes ⇒ total walks = 2 * 4
        self.assertIsInstance(walks, np.ndarray)
        self.assertEqual(walks.shape, (8, 4))
        # all entries must be one of {1,2,3,4}
        unique = set(walks.flatten().tolist())
        self.assertTrue(unique.issubset({1, 2, 3, 4}))

    def test_random_walks_with_single_start(self):
        # only start from node 2
        walks = random_walks(
            self.h, start_from=2, num_walks=3, walk_length=5, p=1.0, q=1.0, threads=1
        )
        # only one start node ⇒ 3 walks
        self.assertEqual(walks.shape, (3, 5))
        self.assertTrue(all((w[0] == 2 for w in walks)))  # each walk begins at 2
        # IDs valid
        self.assertTrue(set(walks.flatten()).issubset({1, 2, 3, 4}))

    def test_time_respecting_random_walks_with_explicit_edges(self):
        # Need multi-timestamp hypergraph for time-respecting walks
        h = ASH()
        h.add_hyperedge([1, 2, 3], start=0, end=1)  # e1 at t0-t1
        h.add_hyperedge([2, 4], start=1, end=2)  # e2 at t1-t2

        walks = time_respecting_random_walks(
            h,
            s=1,
            start_from="e1",
            stop_at=None,
            num_walks=2,
            walk_length=2,
            p=1.0,
            q=1.0,
            edge=True,
            threads=-1,
        )

        # Should have walks from e1 to e2 (time-respecting)
        self.assertGreaterEqual(len(walks), 1)
        self.assertIn("e1", list(walks.keys())[0][0])  # Starts from e1

    def test_tr_rw_terminate_at_sink_edge_mode_empty(self):
        # Build a graph with a single hyperedge at t=5 only, so no overlaps and no forward chain
        h = ASH()
        h.add_hyperedge([10, 11], start=5)  # e1 at t=5
        res = time_respecting_random_walks(
            h,
            s=1,
            start_from="e1",
            num_walks=1,
            walk_length=3,
            edge=True,
            terminate_at_sink=True,
            start=5,
            end=5,
        )
        self.assertIsInstance(res, dict)
        self.assertEqual(res, {})  # no path produced

    def test_tr_rw_stop_at_node_mode(self):
        # Need multi-timestamp hypergraph for time-respecting walks
        h = ASH()
        h.add_hyperedge([1, 2], start=0)  # t=0
        h.add_hyperedge([1, 2], start=1)  # t=1
        h.add_hyperedge([2, 3], start=1)  # t=1

        walks = time_respecting_random_walks(
            h,
            s=1,
            start_from=1,
            stop_at=2,
            num_walks=10,
            walk_length=5,
            edge=False,
            start=0,
            end=1,
        )
        # Should generate some walks from 1 to 2
        self.assertGreaterEqual(walks.shape[0], 1)
        # one walk with exactly one step to node 2
        self.assertGreaterEqual(walks.shape[0], 1)
        self.assertGreaterEqual(walks.shape[1], 1)
        for w in walks:
            self.assertEqual(str(w[-1]), "2")

    def test_tr_rw_start_from_none_node_mode(self):
        # Need multi-timestamp hypergraph for time-respecting walks
        h = ASH()
        h.add_hyperedge([1, 2], start=0)  # t=0
        h.add_hyperedge([2, 3], start=0)  # t=0
        h.add_hyperedge([1, 2], start=1)  # t=1
        h.add_hyperedge([2, 3], start=1)  # t=1

        walks = time_respecting_random_walks(
            h,
            s=1,
            start_from=None,
            num_walks=5,
            walk_length=2,
            edge=False,
            start=0,
            end=1,
        )
        self.assertIsInstance(walks, np.ndarray)
        # At least one short walk should be produced
        self.assertGreaterEqual(len(walks), 1)
        for w in walks:
            self.assertTrue(len(w) >= 1)
            self.assertTrue({str(x) for x in w}.issubset({"1", "2", "3"}))

    def test_tr_rw_waiting_node_mode(self):
        # Node 1 exists at t=0, and can reach node 2 at t=1
        h = ASH()
        h.add_hyperedge([1], start=0)  # t=0: node 1 alone
        h.add_hyperedge([1, 2], start=1)  # t=1: nodes 1 and 2 together
        h.add_hyperedge([2, 3], start=1)  # t=1: nodes 2 and 3 together

        walks = time_respecting_random_walks(
            h,
            s=1,
            start_from=1,
            num_walks=10,
            walk_length=2,
            edge=False,
            start=0,
            end=1,
        )
        self.assertIsInstance(walks, np.ndarray)
        self.assertGreaterEqual(walks.shape[0], 1)
        # Walks should reach other nodes
        for w in walks:
            self.assertGreaterEqual(len(w), 1)

    def test_tr_rw_waiting_edge_mode_self_loop_path(self):
        # For time-respecting walks, we need edges that can transition forward in time
        # A single isolated edge across timestamps cannot form a time-respecting walk
        # as it has no s-incident neighbors at future times
        h = ASH()
        h.add_hyperedge([7], start=0)  # e1 at t=0
        h.add_hyperedge([7], start=1)  # e1 at t=1 (no overlap with other edges)

        res = time_respecting_random_walks(
            h,
            s=1,
            start_from="e1",
            num_walks=1,
            walk_length=3,
            edge=True,
            start=0,
            end=1,
        )
        # With no s-incident neighbors, no time-respecting walks can be formed
        self.assertIsInstance(res, dict)
        # The result may be empty or contain only trivial paths
        self.assertTrue(len(res) == 0 or all(len(paths) == 0 for paths in res.values()))

    def test_random_walks_with_s_parameter(self):
        # Test that s parameter filters connections based on co-occurrence threshold
        h = ASH()
        # Add hyperedges where some pairs co-occur multiple times
        h.add_hyperedge([1, 2, 3], start=0)  # 1-2, 1-3, 2-3 co-occur once
        h.add_hyperedge([1, 2, 4], start=0)  # 1-2 co-occurs twice now, 1-4, 2-4 once
        h.add_hyperedge(
            [1, 2, 5], start=0
        )  # 1-2 co-occurs three times now, 1-5, 2-5 once

        # With s=1 (default), all pairs with at least 1 co-occurrence are connected
        T_s1, mapping_s1 = random_walk_probabilities(h, s=1)
        dense_s1 = T_s1.toarray()
        idx = mapping_s1
        # Node 1 should have transitions to 2,3,4,5
        self.assertGreater(dense_s1[idx[1], idx[2]], 0)  # 1->2
        self.assertGreater(dense_s1[idx[1], idx[3]], 0)  # 1->3
        self.assertGreater(dense_s1[idx[1], idx[4]], 0)  # 1->4
        self.assertGreater(dense_s1[idx[1], idx[5]], 0)  # 1->5

        # With s=2, only pairs that co-occur at least twice are connected
        T_s2, mapping_s2 = random_walk_probabilities(h, s=2)
        dense_s2 = T_s2.toarray()
        idx2 = mapping_s2
        # Only 1-2 and 2-1 should have transitions (they co-occur 3 times)
        self.assertGreater(dense_s2[idx2[1], idx2[2]], 0)  # 1->2
        self.assertGreater(dense_s2[idx2[2], idx2[1]], 0)  # 2->1
        # But 1 should NOT connect to 3,4,5 (they co-occur only once each)
        self.assertEqual(dense_s2[idx2[1], idx2[3]], 0)  # 1->3
        self.assertEqual(dense_s2[idx2[1], idx2[4]], 0)  # 1->4
        self.assertEqual(dense_s2[idx2[1], idx2[5]], 0)  # 1->5

        # With s=3, only pairs that co-occur at least 3 times are connected
        T_s3, mapping_s3 = random_walk_probabilities(h, s=3)
        dense_s3 = T_s3.toarray()
        idx3 = mapping_s3
        # Only 1-2 and 2-1 (co-occur exactly 3 times)
        self.assertGreater(dense_s3[idx3[1], idx3[2]], 0)  # 1->2
        self.assertGreater(dense_s3[idx3[2], idx3[1]], 0)  # 2->1

        # With s=4, no pairs qualify (max co-occurrence is 3)
        T_s4, mapping_s4 = random_walk_probabilities(h, s=4)
        dense_s4 = T_s4.toarray()
        # Matrix should be all zeros
        self.assertEqual(dense_s4.sum(), 0)

    def test_random_walks_edge_mode_with_s_parameter(self):
        # Test s parameter on hyperedge line graph
        h = ASH()
        h.add_hyperedge([1, 2], start=0)  # e1
        h.add_hyperedge([2, 3], start=0)  # e2: shares node 2 with e1
        h.add_hyperedge([1, 2, 3], start=0)  # e3: shares 1,2 with e1 and 2,3 with e2

        # With s=1, hyperedges connected if they share at least 1 node
        T_s1, mapping_s1 = random_walk_probabilities(h, s=1, edge=True)
        dense_s1 = T_s1.toarray()
        # All hyperedges should be connected (they all share at least 1 node)
        self.assertGreater(dense_s1.sum(), 0)

        # With s=2, hyperedges must share at least 2 nodes
        T_s2, mapping_s2 = random_walk_probabilities(h, s=2, edge=True)
        dense_s2 = T_s2.toarray()
        idx2 = mapping_s2
        # e1 and e3 share nodes [1,2] (2 nodes) - should be connected
        # e2 and e3 share nodes [2,3] (2 nodes) - should be connected
        # e1 and e2 share only node 2 (1 node) - should NOT be connected with s=2
        e1_id = h.get_hyperedge_id([1, 2])
        e2_id = h.get_hyperedge_id([2, 3])
        e3_id = h.get_hyperedge_id([1, 2, 3])

        # e1-e3 should be connected
        if e1_id in idx2 and e3_id in idx2:
            self.assertGreater(dense_s2[idx2[e1_id], idx2[e3_id]], 0)
        # e2-e3 should be connected
        if e2_id in idx2 and e3_id in idx2:
            self.assertGreater(dense_s2[idx2[e2_id], idx2[e3_id]], 0)
        # e1-e2 should NOT be connected (only 1 shared node)
        if e1_id in idx2 and e2_id in idx2:
            self.assertEqual(dense_s2[idx2[e1_id], idx2[e2_id]], 0)
