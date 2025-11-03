import unittest

import matplotlib

matplotlib.use("Agg")  # headless

from ash_model.classes import ASH
from ash_model.viz.temporal import (
    plot_activity_series,
    plot_node_activity,
    plot_presence_timeline,
    plot_inter_event_time_distribution,
    plot_hyperedge_lifespan_distribution,
)


class TestTemporalViz(unittest.TestCase):
    def setUp(self):
        self.h = ASH()
        # Build tiny temporal structure
        # Hyperedge e1: nodes 1,2 active t=0..2
        self.h.add_hyperedge([1, 2], 0, 2)
        # Hyperedge e2: nodes 2,3 active t=1..3
        self.h.add_hyperedge([2, 3], 1, 3)
        # Hyperedge e3: nodes 3,4 active t=3 only
        self.h.add_hyperedge([3, 4], 3, 3)

    def test_activity_series(self):
        ax = plot_activity_series(self.h)
        self.assertIsNotNone(ax)

    def test_node_activity(self):
        ax = plot_node_activity(self.h, [1, 2])
        self.assertIsNotNone(ax)

    def test_presence_timeline_hyperedges(self):
        hs = list(self.h.hyperedges())
        ax = plot_presence_timeline(self.h, hyperedges=hs)
        self.assertIsNotNone(ax)

    def test_presence_timeline_nodes(self):
        ax = plot_presence_timeline(self.h, nodes=[1, 2, 3])
        self.assertIsNotNone(ax)

    def test_inter_event_distribution(self):
        ax = plot_inter_event_time_distribution(self.h)
        self.assertIsNotNone(ax)

    def test_lifespan_distribution(self):
        ax = plot_hyperedge_lifespan_distribution(self.h)
        self.assertIsNotNone(ax)


if __name__ == "__main__":
    unittest.main()
