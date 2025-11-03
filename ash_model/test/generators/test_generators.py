import unittest
import random
import numpy as np

from ash_model import ASH, NProfile
from ash_model.generators.random import random_hypergraph, random_ash
from ash_model.generators.homophily_driven import (
    _compute_probs_hba,
    _random_hyperedge,
    _truncated_pareto,
    _sample_to_prob_distr,
    ba_with_homophily,
)


class RandomGeneratorsTestCase(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 10
        self.size_distr = {1: 3, 2: 4, 3: 2}
        self.time_steps = 5
        self.node_attrs = {"color": ["red", "green", "blue"], "age": [20, 30, 40]}
        self.seed = 42

    def test_random_hypergraph_determinism(self):
        h1 = random_hypergraph(
            self.num_nodes, self.size_distr, self.node_attrs, seed=self.seed
        )
        h2 = random_hypergraph(
            self.num_nodes, self.size_distr, self.node_attrs, seed=self.seed
        )
        self.assertEqual(set(h1.nodes()), set(h2.nodes()))
        self.assertEqual(set(h1.hyperedges()), set(h2.hyperedges()))
        for e in h1.hyperedges():
            self.assertEqual(h1.hyperedge_presence(e), h2.hyperedge_presence(e))
        for n in h1.nodes():
            self.assertEqual(
                h1.get_node_attributes(n, tid=0), h2.get_node_attributes(n, tid=0)
            )

    def test_random_hypergraph_structure(self):
        h = random_hypergraph(
            self.num_nodes, self.size_distr, node_attrs=None, seed=self.seed
        )
        self.assertEqual(set(h.nodes()), set(range(self.num_nodes)))
        total_edges = sum(self.size_distr.values())
        self.assertEqual(h.number_of_hyperedges(start=0, end=0), total_edges)
        sizes = [len(h.get_hyperedge_nodes(e)) for e in h.hyperedges(0, 0)]
        for size, count in self.size_distr.items():
            self.assertEqual(sizes.count(size), count)

    def test_random_ash_determinism(self):
        G1 = random_ash(
            self.num_nodes,
            self.size_distr,
            self.time_steps,
            self.node_attrs,
            seed=self.seed,
        )
        G2 = random_ash(
            self.num_nodes,
            self.size_distr,
            self.time_steps,
            self.node_attrs,
            seed=self.seed,
        )
        for n in range(self.num_nodes):
            pres1 = G1.node_presence(n, as_intervals=False)
            pres2 = G2.node_presence(n, as_intervals=False)
            self.assertEqual(pres1, pres2)
            for t in pres1:
                self.assertEqual(
                    G1.get_node_attributes(n, tid=t), G2.get_node_attributes(n, tid=t)
                )
        hes1 = sorted(G1.hyperedges())
        hes2 = sorted(G2.hyperedges())
        self.assertEqual(hes1, hes2)
        for e in hes1:
            self.assertEqual(
                G1.hyperedge_presence(e, as_intervals=False),
                G2.hyperedge_presence(e, as_intervals=False),
            )

    def test_random_ash_structure(self):
        G = random_ash(
            self.num_nodes,
            self.size_distr,
            self.time_steps,
            node_attrs=None,
            seed=self.seed,
        )
        for n in range(self.num_nodes):
            intervals = G.node_presence(n, as_intervals=True)
            self.assertEqual(intervals, [(0, self.time_steps - 1)])
        for e in G.hyperedges():
            intervals = G.hyperedge_presence(e, as_intervals=True)
            self.assertEqual(intervals, [(0, self.time_steps - 1)])
        for t in range(self.time_steps):
            self.assertEqual(
                G.number_of_hyperedges(start=t, end=t), sum(self.size_distr.values())
            )
        distr = G.hyperedge_size_distribution(start=0, end=0)
        self.assertEqual(distr, self.size_distr)


class HomophilyDrivenGeneratorsTestCase(unittest.TestCase):
    def test_truncated_pareto_errors(self):
        with self.assertRaises(ValueError):
            _truncated_pareto(alpha=1.0, size=5, xmin=0, xmax=10)
        with self.assertRaises(ValueError):
            _truncated_pareto(alpha=1.0, size=5, xmin=2, xmax=1)
        with self.assertRaises(ValueError):
            _truncated_pareto(alpha=0, size=5, xmin=2, xmax=10)

    def test_truncated_pareto_output(self):
        size = 1000
        xmin, xmax, alpha = 2.0, 5.0, 1.5
        samples = _truncated_pareto(
            alpha=alpha, size=size, xmin=xmin, xmax=xmax, rng=np.random.default_rng(0)
        )
        self.assertEqual(len(samples), size)
        self.assertTrue((samples >= xmin).all() and (samples <= xmax).all())

    def test_sample_to_prob_distr(self):
        # zero-sum sample => uniform
        uniform = _sample_to_prob_distr([0, 0, 0])
        self.assertEqual(uniform, [1 / 3, 1 / 3, 1 / 3])
        # normal sample
        sample = [2, 3, 5]
        probs = _sample_to_prob_distr(sample)
        self.assertAlmostEqual(sum(probs), 1.0)
        self.assertEqual(probs, [2 / 10, 3 / 10, 5 / 10])

    def test_compute_probs_hba(self):
        # build small ASH with 2 existing nodes and 1 new node
        h = ASH()
        # node 0: red, node 1: blue
        h.add_node(0, start=0, end=0, attr_dict=NProfile(node_id=0, color="red"))
        h.add_node(1, start=0, end=0, attr_dict=NProfile(node_id=1, color="blue"))
        # new node 2: red
        h.add_node(2, start=0, end=0, attr_dict=NProfile(node_id=2, color="red"))
        hr = 0.8
        probs = _compute_probs_hba(h, node=2, homophily_rate=hr)
        # should get [hr, 1-hr]
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(probs[0], hr, places=6)
        self.assertAlmostEqual(probs[1], 1 - hr, places=6)

    def test_ba_with_homophily_errors(self):
        with self.assertRaises(ValueError):
            ba_with_homophily(5, m=1, homophily_rate=-0.1, minority_size=0.5, n0=1)
        with self.assertRaises(ValueError):
            ba_with_homophily(5, m=1, homophily_rate=0.5, minority_size=1.5, n0=1)

    def test_ba_with_homophily_basic(self):
        random.seed(0)
        np.random.seed(0)
        num_nodes, m, hr, ms, n0 = 4, 1, 0.5, 0.5, 1
        # fixed two-size distribution for determinism
        size_prob_distr = [0.5, 0.5]
        h = ba_with_homophily(num_nodes, m, hr, ms, n0, size_prob_distr=size_prob_distr)
        self.assertIsInstance(h, ASH)
        # every node got a color attribute
        for n in range(num_nodes):
            c = h.get_node_attribute(n, "color", tid=0)
            self.assertIn(c, {"red", "blue"})
        # coverage and uniformity should be 1.0 (all present at t=0)
        self.assertEqual(h.coverage(), 1.0)
        self.assertEqual(h.uniformity(), 1.0)
        # each hyperedge size between 2 and 1+len(size_prob_distr)
        for e in h.hyperedges(start=0, end=0):
            sz = len(h.get_hyperedge_nodes(e))
            self.assertIn(sz, {2, 3})
