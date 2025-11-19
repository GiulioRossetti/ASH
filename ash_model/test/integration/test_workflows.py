"""Integration tests for complete analysis workflows."""

import unittest
import numpy as np

from ash_model import ASH, NProfile
from ash_model.measures import attribute_consistency, average_group_degree


class WorkflowIntegrationTestCase(unittest.TestCase):
    """Test complete analysis workflows from data loading to results."""

    def test_attribute_analysis_workflow(self):
        """Test complete workflow for attribute homophily analysis."""
        # 1. Create hypergraph with attributed nodes
        h = ASH()

        # Add nodes with group and numeric attributes
        for i in range(1, 11):
            group = "A" if i <= 5 else "B"
            h.add_node(i, start=0, end=5, attr_dict={"group": group, "value": i * 10})

        # 2. Add hyperedges with some homophilic structure
        # Group A mostly connects within itself
        h.add_hyperedge([1, 2, 3], start=0)
        h.add_hyperedge([2, 3, 4], start=1)
        h.add_hyperedge([4, 5], start=2)

        # Group B mostly connects within itself
        h.add_hyperedge([6, 7, 8], start=0)
        h.add_hyperedge([7, 8, 9], start=1)
        h.add_hyperedge([9, 10], start=2)

        # Some cross-group connections
        h.add_hyperedge([5, 6], start=3)

        # 3. Compute homophily measures
        consistency = attribute_consistency(h, node=1)
        avg_degree_data = average_group_degree(h, tid=0)
        avg_degree_A = avg_degree_data["group"]["A"]
        avg_degree_B = avg_degree_data["group"]["B"]

        # 4. Verify results make sense
        self.assertIsInstance(consistency, dict)
        self.assertGreater(avg_degree_A, 0)
        self.assertGreater(avg_degree_B, 0)

    def test_temporal_evolution_workflow(self):
        """Test workflow analyzing temporal evolution of hypergraph structure."""
        h = ASH()

        # Build evolving hypergraph
        for t in range(10):
            # Add new nodes over time
            new_nodes = list(range(t * 3 + 1, (t + 1) * 3 + 1))
            for node in new_nodes:
                h.add_node(node, start=t, end=t + 2)

            # Add hyperedges connecting recent nodes
            if t > 0:
                h.add_hyperedge(new_nodes, start=t)

        # Analyze temporal evolution
        snapshots = h.temporal_snapshots_ids()
        node_counts = [h.number_of_nodes(start=t, end=t) for t in snapshots]
        edge_counts = [h.number_of_hyperedges(start=t, end=t) for t in snapshots]

        # Verify growth
        self.assertGreater(len(snapshots), 0)
        self.assertGreater(max(node_counts), min(node_counts))
        self.assertGreater(max(edge_counts), 0)

        # Check temporal metrics
        coverage = h.coverage()
        uniformity = h.uniformity()

        self.assertGreaterEqual(coverage, 0.0)
        self.assertLessEqual(coverage, 1.0)
        self.assertGreaterEqual(uniformity, 0.0)
        self.assertLessEqual(uniformity, 1.0)

    def test_clustering_analysis_workflow(self):
        """Test workflow for degree-based structural analysis."""
        h = ASH()

        # Create hypergraph with clustered structure
        # Cluster 1
        h.add_hyperedge([1, 2, 3], start=0)
        h.add_hyperedge([2, 3, 4], start=0)
        h.add_hyperedge([1, 3, 4], start=0)

        # Cluster 2
        h.add_hyperedge([5, 6, 7], start=0)
        h.add_hyperedge([6, 7, 8], start=0)

        # Bridge
        h.add_hyperedge([4, 5], start=0)

        # Compute degree-based metrics
        degree_results = {}
        for node in h.nodes(start=0):
            deg = h.degree(node, start=0)
            degree_results[node] = deg

        # Verify all nodes have valid degrees
        for node, deg in degree_results.items():
            self.assertIsNotNone(deg)
            self.assertGreaterEqual(deg, 0)

    def test_projection_and_analysis_workflow(self):
        """Test workflow using projections for analysis."""
        h = ASH()

        # Build bipartite-like structure
        # Authors: 1, 2, 3, 4
        # Papers: represented as hyperedges
        h.add_hyperedge([1, 2], start=0)  # Paper 1
        h.add_hyperedge([2, 3], start=0)  # Paper 2
        h.add_hyperedge([1, 2, 3], start=0)  # Paper 3
        h.add_hyperedge([3, 4], start=0)  # Paper 4

        # Create clique projection (co-authorship network)
        from ash_model.utils import clique_projection

        G = clique_projection(h, start=0)

        # Analyze projected graph
        self.assertGreater(G.number_of_nodes(), 0)
        self.assertGreater(G.number_of_edges(), 0)

        # Check edge weights (number of co-authorships)
        for u, v, data in G.edges(data=True):
            self.assertIn("weight", data)
            self.assertGreater(data["weight"], 0)

    def test_profile_aggregation_workflow(self):
        """Test workflow with node profile aggregation."""
        h = ASH()

        # Add node with evolving attributes
        h.add_node(1, start=0, attr_dict=NProfile(node_id=1, value=10, category="A"))
        h.add_node(1, start=1, attr_dict=NProfile(node_id=1, value=20, category="A"))
        h.add_node(1, start=2, attr_dict=NProfile(node_id=1, value=30, category="B"))
        h.add_node(1, start=3, attr_dict=NProfile(node_id=1, value=40, category="B"))

        # Get aggregated profile
        agg_profile = h.get_node_profile(1)  # No tid = aggregation

        # Check aggregation
        self.assertIsNotNone(agg_profile)
        attrs = agg_profile.get_attributes()

        # Numeric should be averaged
        self.assertIn("value", attrs)
        self.assertAlmostEqual(attrs["value"], 25.0, delta=1.0)

        # Categorical should be most frequent
        self.assertIn("category", attrs)
        # B appears twice, A appears twice, so could be either
        self.assertIn(attrs["category"], ["A", "B"])

    def test_io_roundtrip_workflow(self):
        """Test complete workflow saving and loading hypergraph."""
        import tempfile
        import os
        from ash_model.readwrite import write_ash_to_json, read_ash_from_json

        # Create hypergraph
        h_original = ASH()
        h_original.add_node(1, start=0, end=2, attr_dict={"name": "Alice"})
        h_original.add_node(2, start=0, end=2, attr_dict={"name": "Bob"})
        h_original.add_hyperedge([1, 2], start=0, end=1, weight=2.5)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            write_ash_to_json(h_original, temp_path)

            # Load back
            h_loaded = read_ash_from_json(temp_path)

            # Verify preservation
            self.assertEqual(set(h_original.nodes()), set(h_loaded.nodes()))
            self.assertEqual(set(h_original.hyperedges()), set(h_loaded.hyperedges()))
            self.assertEqual(
                h_original.get_node_attributes(1), h_loaded.get_node_attributes(1)
            )
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_generator_and_analysis_workflow(self):
        """Test workflow generating synthetic hypergraph and analyzing it."""
        from ash_model.generators import random_ash

        # Generate random hypergraph
        h = random_ash(
            num_nodes=20,
            size_distr={2: 10, 3: 10, 4: 5, 5: 5},  # 30 total hyperedges
            time_steps=5,
            seed=42,
        )

        # Verify structure
        self.assertEqual(h.number_of_nodes(), 20)
        self.assertGreaterEqual(h.number_of_hyperedges(), 1)

        # Analyze degree distribution
        degree_dist = h.degree_distribution()
        self.assertIsInstance(degree_dist, dict)
        self.assertGreater(len(degree_dist), 0)

        # Analyze size distribution
        size_dist = h.hyperedge_size_distribution()
        self.assertIsInstance(size_dist, dict)
        self.assertGreater(len(size_dist), 0)

        # Check all sizes respect max_size constraint
        for size in size_dist.keys():
            self.assertLessEqual(size, 5)


if __name__ == "__main__":
    unittest.main()
