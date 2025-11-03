"""Tests for hypergraph segregation measures (RWHS)."""

import unittest
import numpy as np

from ash_model import ASH
from ash_model.measures.hyper_segregation import rwhs, temporal_rwhs


class HyperSegregationTestCase(unittest.TestCase):
    """Test cases for Random Walk Hypergraph Similarity (RWHS) measures."""
    
    def setUp(self):
        """Create test hypergraphs with attributes."""
        # Simple hypergraph with 2 groups
        self.h = ASH()
        self.h.add_node(1, start=0, end=2, attr_dict={"group": "A", "value": 10})
        self.h.add_node(2, start=0, end=2, attr_dict={"group": "A", "value": 20})
        self.h.add_node(3, start=0, end=2, attr_dict={"group": "B", "value": 30})
        self.h.add_node(4, start=0, end=2, attr_dict={"group": "B", "value": 40})
        
        self.h.add_hyperedge([1, 2], start=0, end=2)
        self.h.add_hyperedge([3, 4], start=0, end=2)
        self.h.add_hyperedge([1, 2, 3], start=1, end=2)
    
    def test_rwhs_meet_basic(self):
        """Test RWHS meet method returns valid scores."""
        scores = rwhs(self.h, tid=0, num_walks=10, walk_length=5, method='meet')
        
        # Should return dict of dicts
        self.assertIsInstance(scores, dict)
        
        # Scores should be between 0 and 1
        for node, attr_scores in scores.items():
            self.assertIn(node, [1, 2, 3, 4])
            for attr, score in attr_scores.items():
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
    
    def test_rwhs_jump_basic(self):
        """Test RWHS jump method returns valid scores."""
        scores = rwhs(self.h, tid=0, num_walks=10, walk_length=5, method='jump')
        
        # Should return dict of dicts
        self.assertIsInstance(scores, dict)
        
        # Scores should be between 0 and 1
        for node, attr_scores in scores.items():
            for attr, score in attr_scores.items():
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
    
    def test_rwhs_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        with self.assertRaises(ValueError):
            rwhs(self.h, tid=0, method='invalid')
    
    def test_rwhs_with_s_parameter(self):
        """Test RWHS with different s-incidence thresholds."""
        # Add more co-occurrences for some pairs
        h = ASH()
        h.add_node(1, start=0, attr_dict={"group": "A"})
        h.add_node(2, start=0, attr_dict={"group": "A"})
        h.add_node(3, start=0, attr_dict={"group": "B"})
        
        # 1-2 co-occur in multiple hyperedges
        h.add_hyperedge([1, 2], start=0)
        h.add_hyperedge([1, 2, 3], start=0)
        h.add_hyperedge([1, 3], start=0)
        
        # With s=1, all connections included
        scores_s1 = rwhs(h, tid=0, s=1, num_walks=20, walk_length=5, method='meet')
        
        # With s=2, only strong connections (1-2 co-occur twice)
        scores_s2 = rwhs(h, tid=0, s=2, num_walks=20, walk_length=5, method='meet')
        
        # Both should produce results
        self.assertGreater(len(scores_s1), 0)
        self.assertGreater(len(scores_s2), 0)
    
    def test_rwhs_homogeneous_vs_heterogeneous(self):
        """Test that homogeneous groups have higher RWHS scores."""
        # Homogeneous hypergraph (all same group)
        h_homo = ASH()
        for i in range(1, 5):
            h_homo.add_node(i, start=0, attr_dict={"group": "A"})
        h_homo.add_hyperedge([1, 2, 3, 4], start=0)
        
        # Heterogeneous hypergraph (mixed groups)
        h_hetero = ASH()
        for i in [1, 2]:
            h_hetero.add_node(i, start=0, attr_dict={"group": "A"})
        for i in [3, 4]:
            h_hetero.add_node(i, start=0, attr_dict={"group": "B"})
        h_hetero.add_hyperedge([1, 2, 3, 4], start=0)
        
        scores_homo = rwhs(h_homo, tid=0, num_walks=50, walk_length=10, method='meet')
        scores_hetero = rwhs(h_hetero, tid=0, num_walks=50, walk_length=10, method='meet')
        
        # Homogeneous should have higher average group similarity
        if 1 in scores_homo and "group" in scores_homo[1]:
            homo_score = scores_homo[1]["group"]
            if 1 in scores_hetero and "group" in scores_hetero[1]:
                hetero_score = scores_hetero[1]["group"]
                self.assertGreaterEqual(homo_score, hetero_score)
    
    def test_temporal_rwhs_basic(self):
        """Test temporal RWHS on hyperedge walks."""
        # Create temporal hypergraph
        h = ASH()
        h.add_node(1, start=0, end=2, attr_dict={"group": "A"})
        h.add_node(2, start=0, end=2, attr_dict={"group": "A"})
        h.add_node(3, start=0, end=2, attr_dict={"group": "B"})
        h.add_node(4, start=0, end=2, attr_dict={"group": "B"})
        
        h.add_hyperedge([1, 2], start=0, end=1)
        h.add_hyperedge([2, 3], start=1, end=2)
        h.add_hyperedge([3, 4], start=2, end=2)
        
        scores = temporal_rwhs(h, s=1, tid=0, num_walks=10, walk_length=5, method='meet')
        
        # Should return dict of dicts
        self.assertIsInstance(scores, dict)
        
        # Scores should be between 0 and 1
        for node, attr_scores in scores.items():
            for attr, score in attr_scores.items():
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
    
    def test_temporal_rwhs_jump_method(self):
        """Test temporal RWHS with jump method."""
        h = ASH()
        h.add_node(1, start=0, end=2, attr_dict={"group": "A"})
        h.add_node(2, start=0, end=2, attr_dict={"group": "A"})
        h.add_node(3, start=0, end=2, attr_dict={"group": "B"})
        
        h.add_hyperedge([1, 2], start=0, end=1)
        h.add_hyperedge([2, 3], start=1, end=2)
        
        scores = temporal_rwhs(h, s=1, tid=0, num_walks=10, walk_length=5, method='jump')
        
        self.assertIsInstance(scores, dict)
    
    def test_temporal_rwhs_invalid_method_raises(self):
        """Test that temporal RWHS with invalid method raises ValueError."""
        h = ASH()
        h.add_node(1, start=0, attr_dict={"group": "A"})
        h.add_hyperedge([1], start=0)
        
        with self.assertRaises(ValueError):
            temporal_rwhs(h, s=1, tid=0, method='invalid')
    
    def test_rwhs_empty_walks(self):
        """Test RWHS handles case with insufficient nodes gracefully."""
        h = ASH()
        h.add_node(1, start=0, attr_dict={"group": "A"})
        h.add_node(2, start=0, attr_dict={"group": "A"})
        # Two isolated nodes - need hyperedge for walks
        h.add_hyperedge([1, 2], start=0)
        
        scores = rwhs(h, tid=0, num_walks=10, walk_length=5, method='meet')
        
        # Should return valid dict even if walks are trivial
        self.assertIsInstance(scores, dict)
    
    def test_rwhs_start_from_specific_node(self):
        """Test RWHS starting from specific node(s)."""
        scores = rwhs(
            self.h, 
            tid=0, 
            start_from=1,  # Start only from node 1
            num_walks=10, 
            walk_length=5, 
            method='meet'
        )
        
        self.assertIsInstance(scores, dict)
        # Should have results starting from node 1
        if 1 in scores:
            self.assertIsInstance(scores[1], dict)
    
    def test_rwhs_edge_mode(self):
        """Test RWHS on hyperedge line graph."""
        # Need at least 2 hyperedges that share nodes for edge walks
        h = ASH()
        h.add_node(1, start=0, attr_dict={"group": "A"})
        h.add_node(2, start=0, attr_dict={"group": "A"})
        h.add_node(3, start=0, attr_dict={"group": "B"})
        
        h.add_hyperedge([1, 2], start=0)
        h.add_hyperedge([2, 3], start=0)  # Shares node 2 with first edge
        
        scores = rwhs(
            h,
            tid=0,
            num_walks=10,
            walk_length=5,
            edge=True,  # Walk on hyperedge line graph
            method='meet'
        )
        
        self.assertIsInstance(scores, dict)


if __name__ == '__main__':
    unittest.main()
