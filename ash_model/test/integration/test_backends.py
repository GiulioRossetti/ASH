"""Integration tests comparing Dense and Interval backends."""

import unittest
import numpy as np

from ash_model import ASH


class BackendIntegrationTestCase(unittest.TestCase):
    """Test that Dense and Interval backends produce identical results."""
    
    def test_identical_node_operations(self):
        """Test that both backends handle node operations identically."""
        h_dense = ASH(backend="dense")
        h_interval = ASH(backend="interval")
        
        # Add same nodes to both
        for i in range(1, 6):
            h_dense.add_node(i, start=0, end=10)
            h_interval.add_node(i, start=0, end=10)
        
        # Verify same nodes present
        self.assertEqual(set(h_dense.nodes()), set(h_interval.nodes()))
        self.assertEqual(
            h_dense.number_of_nodes(),
            h_interval.number_of_nodes()
        )
    
    def test_identical_hyperedge_operations(self):
        """Test that both backends handle hyperedge operations identically."""
        h_dense = ASH(backend="dense")
        h_interval = ASH(backend="interval")
        
        # Add same hyperedges
        edges = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
        for edge in edges:
            h_dense.add_hyperedge(edge, start=0, end=5)
            h_interval.add_hyperedge(edge, start=0, end=5)
        
        # Verify same hyperedges
        self.assertEqual(
            set(h_dense.hyperedges()),
            set(h_interval.hyperedges())
        )
        self.assertEqual(
            h_dense.number_of_hyperedges(),
            h_interval.number_of_hyperedges()
        )
    
    def test_identical_temporal_queries(self):
        """Test that temporal queries return same results."""
        h_dense = ASH(backend="dense")
        h_interval = ASH(backend="interval")
        
        # Add temporal data
        h_dense.add_hyperedge([1, 2], start=0, end=2)
        h_dense.add_hyperedge([2, 3], start=1, end=3)
        h_dense.add_hyperedge([1, 3], start=2, end=4)
        
        h_interval.add_hyperedge([1, 2], start=0, end=2)
        h_interval.add_hyperedge([2, 3], start=1, end=3)
        h_interval.add_hyperedge([1, 3], start=2, end=4)
        
        # Check temporal snapshots
        self.assertEqual(
            h_dense.temporal_snapshots_ids(),
            h_interval.temporal_snapshots_ids()
        )
        
        # Check node presence
        for t in range(5):
            self.assertEqual(
                set(h_dense.nodes(start=t, end=t)),
                set(h_interval.nodes(start=t, end=t))
            )
    
    def test_identical_node_presence(self):
        """Test that node_presence returns identical results."""
        h_dense = ASH(backend="dense")
        h_interval = ASH(backend="interval")
        
        # Add node with complex temporal pattern
        h_dense.add_node(1, start=0, end=2)
        h_dense.add_node(1, start=5, end=7)
        h_dense.add_node(1, start=10, end=12)
        
        h_interval.add_node(1, start=0, end=2)
        h_interval.add_node(1, start=5, end=7)
        h_interval.add_node(1, start=10, end=12)
        
        # Check presence as times
        self.assertEqual(
            h_dense.node_presence(1),
            h_interval.node_presence(1)
        )
        
        # Check presence as intervals
        self.assertEqual(
            h_dense.node_presence(1, as_intervals=True),
            h_interval.node_presence(1, as_intervals=True)
        )
    
    def test_identical_hyperedge_presence(self):
        """Test that hyperedge_presence returns identical results."""
        h_dense = ASH(backend="dense")
        h_interval = ASH(backend="interval")
        
        # Add hyperedge with gaps
        h_dense.add_hyperedge([1, 2, 3], start=0, end=2)
        h_dense.add_hyperedge([1, 2, 3], start=5, end=7)
        
        h_interval.add_hyperedge([1, 2, 3], start=0, end=2)
        h_interval.add_hyperedge([1, 2, 3], start=5, end=7)
        
        # Get hyperedge id
        eid_dense = h_dense.get_hyperedge_id([1, 2, 3])
        eid_interval = h_interval.get_hyperedge_id([1, 2, 3])
        
        # Check presence
        self.assertEqual(
            h_dense.hyperedge_presence(eid_dense),
            h_interval.hyperedge_presence(eid_interval)
        )
        
        self.assertEqual(
            h_dense.hyperedge_presence(eid_dense, as_intervals=True),
            h_interval.hyperedge_presence(eid_interval, as_intervals=True)
        )
    
    def test_identical_degree_calculations(self):
        """Test that degree calculations are identical."""
        h_dense = ASH(backend="dense")
        h_interval = ASH(backend="interval")
        
        # Build same hypergraph
        edges = [[1, 2], [2, 3], [1, 2, 3], [3, 4]]
        for edge in edges:
            h_dense.add_hyperedge(edge, start=0)
            h_interval.add_hyperedge(edge, start=0)
        
        # Check degrees for all nodes
        for node in [1, 2, 3, 4]:
            self.assertEqual(
                h_dense.degree(node, start=0),
                h_interval.degree(node, start=0)
            )
    
    def test_identical_star_operations(self):
        """Test that star() returns identical results."""
        h_dense = ASH(backend="dense")
        h_interval = ASH(backend="interval")
        
        edges = [[1, 2, 3], [1, 4], [2, 3, 4]]
        for edge in edges:
            h_dense.add_hyperedge(edge, start=0)
            h_interval.add_hyperedge(edge, start=0)
        
        # Check star for each node
        for node in [1, 2, 3, 4]:
            self.assertEqual(
                set(h_dense.star(node, start=0)),
                set(h_interval.star(node, start=0))
            )
    
    def test_identical_stream_interactions(self):
        """Test that stream_interactions produces same events."""
        h_dense = ASH(backend="dense")
        h_interval = ASH(backend="interval")
        
        # Add temporal hyperedges
        h_dense.add_hyperedge([1, 2], start=0, end=2)
        h_dense.add_hyperedge([2, 3], start=1, end=3)
        
        h_interval.add_hyperedge([1, 2], start=0, end=2)
        h_interval.add_hyperedge([2, 3], start=1, end=3)
        
        # Collect streams
        stream_dense = list(h_dense.stream_interactions())
        stream_interval = list(h_interval.stream_interactions())
        
        # Should have same events
        self.assertEqual(len(stream_dense), len(stream_interval))
        self.assertEqual(set(stream_dense), set(stream_interval))
    
    def test_identical_with_attributes(self):
        """Test that backends handle attributes identically."""
        h_dense = ASH(backend="dense")
        h_interval = ASH(backend="interval")
        
        # Add nodes with attributes
        attrs = {"name": "Alice", "age": 30}
        h_dense.add_node(1, start=0, end=2, attr_dict=attrs)
        h_interval.add_node(1, start=0, end=2, attr_dict=attrs)
        
        # Check attributes
        self.assertEqual(
            h_dense.get_node_attributes(1),
            h_interval.get_node_attributes(1)
        )
        
        # Add hyperedge with attributes
        hedge_attrs = {"weight": 2.5, "type": "collab"}
        h_dense.add_hyperedge([1, 2], start=0, **hedge_attrs)
        h_interval.add_hyperedge([1, 2], start=0, **hedge_attrs)
        
        eid_dense = h_dense.get_hyperedge_id([1, 2])
        eid_interval = h_interval.get_hyperedge_id([1, 2])
        
        self.assertEqual(
            h_dense.get_hyperedge_attributes(eid_dense),
            h_interval.get_hyperedge_attributes(eid_interval)
        )
    
    def test_performance_characteristics(self):
        """Test that backends have expected performance characteristics for different patterns."""
        import time
        
        # Dense pattern: many short-lived entities
        h_dense_backend = ASH(backend="dense")
        h_interval_backend = ASH(backend="interval")
        
        # Add many short-lived hyperedges
        n_edges = 100
        
        start_dense = time.time()
        for i in range(n_edges):
            h_dense_backend.add_hyperedge([i, i+1], start=i, end=i)
        time_dense = time.time() - start_dense
        
        start_interval = time.time()
        for i in range(n_edges):
            h_interval_backend.add_hyperedge([i, i+1], start=i, end=i)
        time_interval = time.time() - start_interval
        
        # Both should complete reasonably fast (< 1 second)
        self.assertLess(time_dense, 1.0)
        self.assertLess(time_interval, 1.0)
        
        # Verify same result
        self.assertEqual(
            h_dense_backend.number_of_hyperedges(),
            h_interval_backend.number_of_hyperedges()
        )


if __name__ == '__main__':
    unittest.main()
