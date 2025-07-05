import unittest

import numpy as np
from scipy import sparse

from ash_model import ASH
from ash_model.paths.randwalks import (
    random_walk_probabilities,
    random_walks,
    # time_respecting_random_walks,
)


class RandomWalksTestCase(unittest.TestCase):
    def setUp(self):
        # Build a tiny hypergraph with one 3‐node edge and one 2‐node edge
        self.h = ASH()
        self.h.add_hyperedge([1, 2, 3], start=0)  # e1
        self.h.add_hyperedge([2, 4], start=0)  # e2

    def test_random_walk_probabilities_exact(self):
        # For hyperedge [1,2,3]: weight per pair = 2
        # For hyperedge [2,4]: weight per pair = 1
        # Node order: [1,2,3,4]
        T, mapping = random_walk_probabilities(self.h)
        # must be csr
        self.assertIsInstance(T, sparse.csr_matrix)
        # mapping correctness
        self.assertEqual(mapping, {1: 0, 2: 1, 3: 2, 4: 3})
        dense = T.toarray()
        # Manually build expected raw counts:
        #   from 1: only edge e1 → neighbors 2,3 each get +2 ⇒ total 4 → p=2/4=0.5
        #   from 2: in e1 neighbors 1,3 (+2 each), in e2 neighbor 4 (+1) ⇒ totals [1:2,3:2,4:1], sum=5
        #              p(2→1)=2/5, p(2→3)=2/5, p(2→4)=1/5
        #   from 3: only e1 → neighbors 1,2 each +2 ⇒ sum=4 → p=0.5 each
        #   from 4: only e2 → neighbor 2 gets +1 ⇒ sum=1 → p(4→2)=1
        expected = np.array(
            [
                [0.0, 0.5, 0.5, 0.0],
                [2 / 5, 0.0, 2 / 5, 1 / 5],
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
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

    """
    def test_time_respecting_random_walks_basic(self):
        # With s=1, hyperedge_from=None ⇒ start from both hyperedges ['e1','e2']
        walks = time_respecting_random_walks(
            self.h,
            s=1,
            hyperedge_from=None,
            hyperedge_to=None,
            num_walks=2,
            walk_length=3,
            p=1.0,
            q=1.0,
            threads=-1,
        )
        # we have 2 hyperedges ⇒ total walks = 2 * 2
        self.assertEqual(walks.shape, (4, 3))
        # IDs should be hyperedge IDs (strings 'e1' or 'e2')
        uniq = set(walks.flatten())
        self.assertTrue(uniq.issubset({"e1", "e2"}))

    def test_time_respecting_random_walks_with_explicit_edges(self):
        # start only from e2
        walks = time_respecting_random_walks(
            self.h,
            s=1,
            hyperedge_from="e2",
            hyperedge_to=None,
            num_walks=4,
            walk_length=2,
            threads=1,
        )
        # only one start edge ⇒ 4 walks
        self.assertEqual(walks.shape, (4, 2))
        # first column is always 'e2'
        self.assertTrue(all(row[0] == "e2" for row in walks))
        # values valid
        self.assertTrue(set(walks.flatten()).issubset({"e1", "e2"}))
    """
