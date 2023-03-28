import unittest

from ash.viz import *


class TemporalVizTestCase(unittest.TestCase):
    @staticmethod
    def get_hypergraph():
        a = ASH(hedge_removal=True)
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

    def test_plot_structure_dynamics(self):
        a = self.get_hypergraph()

        ax = plot_structure_dynamics(a,
                                     average_s_local_clustering_coefficient,
                                     func_params={'s': 2})
        self.assertIsInstance(ax, plt.Axes)
        ax = plot_structure_dynamics(a,
                                     inclusiveness,
                                     func_params={})
        self.assertIsInstance(ax, plt.Axes)

    def test_plot_attribute_dynamics(self):
        a = self.get_hypergraph()
        ax = plot_attribute_dynamics(a,
                                     attr_name='party',
                                     func=average_hyperedge_profile_entropy)
        self.assertIsInstance(ax, plt.Axes)
        ax = plot_attribute_dynamics(a,
                                     attr_name='party',
                                     func=average_hyperedge_profile_purity,
                                     func_params={'by_label': True}
                                     )
        self.assertIsInstance(ax, plt.Axes)
        ax = plot_attribute_dynamics(a,
                                     attr_name='party',
                                     func=average_hyperedge_profile_purity,
                                     func_params={'by_label': False}
                                     )
        self.assertIsInstance(ax, plt.Axes)
        ax = plot_attribute_dynamics(a,
                                     attr_name='party',
                                     func=average_star_profile_homogeneity,
                                     func_params={'by_label': True}
                                     )
        self.assertIsInstance(ax, plt.Axes)
        ax = plot_attribute_dynamics(a,
                                     attr_name='party',
                                     func=average_star_profile_homogeneity,
                                     func_params={'by_label': False}
                                     )
        self.assertIsInstance(ax, plt.Axes)

    # def test_plot_temporal_attribute_distribution(self):
    #    a = self.get_hypergraph()
    #    for attr_name in ['gender', 'party']:
    #        ax = plot_temporal_attribute_distribution(a, attr_name)
    #        self.assertIsInstance(ax, plt.Axes)

    def test_plot_consistency(self):
        a = self.get_hypergraph()
        ax = plot_consistency(a)
        self.assertIsInstance(ax, plt.Axes)
