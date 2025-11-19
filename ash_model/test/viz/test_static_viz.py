import unittest

import matplotlib

matplotlib.use("Agg")  # headless

from ash_model.classes import ASH
from ash_model.viz.static import (
    plot_s_degrees,
    plot_hyperedge_size_distribution,
    plot_degree_distribution,
    plot_s_ranks,
)


class TestStaticViz(unittest.TestCase):
    def setUp(self):
        """Create a test hypergraph with varied structure."""
        self.h = ASH()

        # Create hyperedges with varying sizes
        self.h.add_hyperedge([1, 2], 0, 5)
        self.h.add_hyperedge([1, 3], 1, 4)
        self.h.add_hyperedge([2, 3, 4], 2, 5)
        self.h.add_hyperedge([4, 5], 5, 10)
        self.h.add_hyperedge([5, 6, 7], 6, 9)
        self.h.add_hyperedge([3, 4, 5], 5, 8)
        self.h.add_hyperedge([7, 8], 10, 15)
        self.h.add_hyperedge([6, 7, 8, 9], 11, 14)
        self.h.add_hyperedge([1, 2, 3, 4, 5], 5, 10)

    def test_s_degrees_loglog(self):
        """Test s-degrees plot with loglog scale."""
        ax = plot_s_degrees(self.h, smax=3, loglog=True)
        self.assertIsNotNone(ax)
        self.assertTrue(hasattr(ax, "lines"))
        self.assertGreater(len(ax.lines), 0)

    def test_s_degrees_linear(self):
        """Test s-degrees plot with linear scale."""
        ax = plot_s_degrees(self.h, smax=2, loglog=False)
        self.assertIsNotNone(ax)
        self.assertTrue(hasattr(ax, "lines"))
        self.assertGreater(len(ax.lines), 0)

    def test_hyperedge_size_distribution(self):
        """Test hyperedge size distribution plot."""
        ax = plot_hyperedge_size_distribution(self.h)
        self.assertIsNotNone(ax)
        self.assertTrue(hasattr(ax, "patches"))
        self.assertGreater(len(ax.patches), 0)

    def test_hyperedge_size_distribution_filtered(self):
        """Test hyperedge size distribution with min/max filters."""
        ax = plot_hyperedge_size_distribution(self.h, min_size=2, max_size=4)
        self.assertIsNotNone(ax)
        self.assertTrue(hasattr(ax, "patches"))

    def test_degree_distribution_loglog(self):
        """Test degree distribution with loglog scale."""
        ax = plot_degree_distribution(self.h, loglog=True)
        self.assertIsNotNone(ax)
        self.assertTrue(hasattr(ax, "lines"))
        self.assertGreater(len(ax.lines), 0)

    def test_degree_distribution_linear(self):
        """Test degree distribution with linear scale."""
        ax = plot_degree_distribution(self.h, loglog=False)
        self.assertIsNotNone(ax)
        self.assertTrue(hasattr(ax, "lines"))
        self.assertGreater(len(ax.lines), 0)

    def test_s_ranks_loglog(self):
        """Test s-ranks plot with loglog scale."""
        ax = plot_s_ranks(self.h, smax=3, loglog=True)
        self.assertIsNotNone(ax)
        self.assertTrue(hasattr(ax, "lines"))
        self.assertGreater(len(ax.lines), 0)

    def test_s_ranks_linear(self):
        """Test s-ranks plot with linear scale."""
        ax = plot_s_ranks(self.h, smax=2, loglog=False)
        self.assertIsNotNone(ax)
        self.assertTrue(hasattr(ax, "lines"))
        self.assertGreater(len(ax.lines), 0)


if __name__ == "__main__":
    unittest.main()
