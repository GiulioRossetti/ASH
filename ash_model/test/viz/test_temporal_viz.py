import unittest

import matplotlib

matplotlib.use("Agg")  # headless

from ash_model.classes import ASH
from ash_model.viz.temporal import (
    plot_hyperedge_activity_series,
    plot_node_activity_series,
    plot_presence_timeline,
    plot_inter_event_time_distribution,
    plot_hyperedge_lifespan_distribution,
    plot_node_lifespan_distribution,
)


class TestTemporalViz(unittest.TestCase):
    def setUp(self):
        """Create a test hypergraph with temporal structure."""
        self.h = ASH()
        # Build temporal structure with varied patterns
        # Hyperedge e1: nodes 1,2 active t=0..2
        self.h.add_hyperedge([1, 2], 0, 2)
        # Hyperedge e2: nodes 2,3 active t=1..3
        self.h.add_hyperedge([2, 3], 1, 3)
        # Hyperedge e3: nodes 3,4 active t=3 only
        self.h.add_hyperedge([3, 4], 3, 3)
        # Hyperedge e4: nodes 1,3 active t=2..5
        self.h.add_hyperedge([1, 3], 2, 5)
        # Hyperedge e5: larger hyperedge
        self.h.add_hyperedge([1, 2, 3, 4], 4, 6)

    def test_hyperedge_activity_series(self):
        """Test hyperedge activity series plot."""
        ax = plot_hyperedge_activity_series(self.h)
        self.assertIsNotNone(ax)
        self.assertTrue(hasattr(ax, "lines"))
        self.assertGreater(len(ax.lines), 0)

    def test_hyperedge_activity_series_normalized(self):
        """Test normalized hyperedge activity series plot."""
        ax = plot_hyperedge_activity_series(self.h, normalize=True)
        self.assertIsNotNone(ax)
        self.assertTrue(hasattr(ax, "lines"))
        self.assertGreater(len(ax.lines), 0)

    def test_node_activity_series(self):
        """Test node activity series plot."""
        ax = plot_node_activity_series(self.h)
        self.assertIsNotNone(ax)
        self.assertTrue(hasattr(ax, "lines"))
        self.assertGreater(len(ax.lines), 0)

    def test_node_activity_series_normalized(self):
        """Test normalized node activity series plot."""
        ax = plot_node_activity_series(self.h, normalize=True)
        self.assertIsNotNone(ax)
        self.assertTrue(hasattr(ax, "lines"))
        self.assertGreater(len(ax.lines), 0)

    def test_presence_timeline_hyperedges(self):
        """Test presence timeline for hyperedges."""
        hs = list(self.h.hyperedges())
        ax = plot_presence_timeline(self.h, hyperedges=hs)
        self.assertIsNotNone(ax)

    def test_presence_timeline_nodes(self):
        """Test presence timeline for nodes."""
        ax = plot_presence_timeline(self.h, nodes=[1, 2, 3])
        self.assertIsNotNone(ax)

    def test_inter_event_distribution(self):
        """Test inter-event time distribution plot."""
        ax = plot_inter_event_time_distribution(self.h)
        self.assertIsNotNone(ax)

    def test_hyperedge_lifespan_distribution(self):
        """Test hyperedge lifespan distribution plot."""
        ax = plot_hyperedge_lifespan_distribution(self.h)
        self.assertIsNotNone(ax)

    def test_node_lifespan_distribution(self):
        """Test node lifespan distribution plot."""
        ax = plot_node_lifespan_distribution(self.h)
        self.assertIsNotNone(ax)


if __name__ == "__main__":
    unittest.main()
