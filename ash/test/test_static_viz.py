import unittest

from ash.viz import *


class StaticVizTestCase(unittest.TestCase):
    @staticmethod
    def get_hypergraph():
        a = ASH(hedge_removal=True)
        a.add_hyperedge([1, 2, 3], 0)
        a.add_hyperedge([15, 25], 0)
        a.add_hyperedge([1, 24, 34], 0)
        a.add_hyperedge([1, 2, 5, 6], 0)
        a.add_hyperedge([1, 2, 5], 1)
        a.add_hyperedge([3, 4, 5, 10], 1)
        a.add_hyperedge([3, 4, 5, 12], 1)
        return a

    def test_plot_s_degrees(self):
        a = self.get_hypergraph()
        ax = plot_s_degrees(a, smax=3)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_hyperedge_size_distribution(self):
        a = self.get_hypergraph()
        ax = plot_hyperedge_size_distribution(a)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_degree_distribution(self):
        a = self.get_hypergraph()
        ax = plot_degree_distribution(a)
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_s_ranks(self):
        a = self.get_hypergraph()
        ax = plot_s_ranks(a, smax=3)
        self.assertIsInstance(ax, plt.Axes)
