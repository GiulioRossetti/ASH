"""Tests for the Multi-Ego Network module."""

import unittest
from ash_model.classes import ASH
from ash_model.multiego import (
    get_multiego, 
    get_fractured_multiego, 
    get_core_multiego,
    jaccard_similarity,
    minimum_overlapping_similarity,
    delta_similarity
)


class TestMultiEgo(unittest.TestCase):
    """Test Multi-Ego Network extraction functions."""
    
    def setUp(self):
        """Create a simple temporal hypergraph for testing."""
        self.h = ASH()
        
        # Time 0: Add some hyperedges
        self.h.add_hyperedge([1, 2, 3], start=0)
        self.h.add_hyperedge([2, 3, 4], start=0)
        self.h.add_hyperedge([1, 4, 5], start=0)
        self.h.add_hyperedge([5, 6], start=0)
        
        # Time 1: Add different hyperedges
        self.h.add_hyperedge([1, 2], start=1)
        self.h.add_hyperedge([3, 4, 5], start=1)
        self.h.add_hyperedge([6, 7], start=1)
    
    def test_get_mren_basic(self):
        """Test basic MultiEgo extraction with single ego."""
        U = {1}
        multiego = get_multiego(self.h, U, start=0)
        
        # Node 1 appears in e1 and e3
        self.assertEqual(len(multiego), 2)
        
        # Check that the right hyperedges are included
        multiego_edges = [frozenset(edge) for edge in multiego]
        self.assertIn(frozenset({1, 2, 3}), multiego_edges)
        self.assertIn(frozenset({1, 4, 5}), multiego_edges)
    
    def test_get_mren_multiple_egos(self):
        """Test MultiEgo extraction with multiple ego nodes."""
        U = {1, 2}
        multiego = get_multiego(self.h, U, start=0)
        
        # Nodes 1 and 2 appear in e1, e2, e3
        self.assertEqual(len(multiego), 3)
        
        multiego_edges = [frozenset(edge) for edge in multiego]
        self.assertIn(frozenset({1, 2, 3}), multiego_edges)
        self.assertIn(frozenset({2, 3, 4}), multiego_edges)
        self.assertIn(frozenset({1, 4, 5}), multiego_edges)
    
    def test_get_mren_empty_ego_set(self):
        """Test MultiEgo with empty ego set."""
        U = set()
        multiego = get_multiego(self.h, U, start=0)
        
        self.assertEqual(len(multiego), 0)
    
    def test_get_mren_no_matches(self):
        """Test MultiEgo when ego nodes don't appear in any hyperedge."""
        U = {100, 200}
        multiego = get_multiego(self.h, U, start=0)
        
        self.assertEqual(len(multiego), 0)
    
    def test_get_mren_invalid_tid(self):
        """Test MultiEgo with invalid time ID."""
        U = {1}
        multiego = get_multiego(self.h, U, start=999)
        
        self.assertEqual(len(multiego), 0)
    
    def test_get_mren_temporal(self):
        """Test MultiEgo across different time slices."""
        U = {1}
        
        multiego_t0 = get_multiego(self.h, U, start=0)
        multiego_t1 = get_multiego(self.h, U, start=1)
        
        # Node 1 appears in 2 edges at t=0 and 1 edge at t=1
        self.assertEqual(len(multiego_t0), 2)
        self.assertEqual(len(multiego_t1), 1)
    
    def test_get_mren_time_window(self):
        """Test MultiEgo extraction over a time window."""
        U = {1, 2}
        
        # Single snapshot
        multiego_t0 = get_multiego(self.h, U, start=0)
        
        # Time window [0, 1]
        multiego_window = get_multiego(self.h, U, start=0, end=1)
        
        # Window should have unique hyperedges from both time points
        # t=0: {1,2,3}, {2,3,4}, {1,4,5}
        # t=1: {1,2}, {3,4,5}
        # Union (unique): should aggregate without duplicates
        self.assertGreaterEqual(len(multiego_window), len(multiego_t0))
        
        # Verify all edges contain at least one node from U
        for edge in multiego_window:
            self.assertTrue(U.intersection(edge))
    
    def test_get_fractured_mren_alpha_half(self):
        """Test Fractured MultiEgo with alpha=0.5."""
        U = {1, 2, 3, 4}
        alpha = 0.5  # Need at least 2 nodes from U
        multiego = get_fractured_multiego(self.h, U, start=0, alpha=alpha)
        
        # e1 has {1,2,3} (3 from U), e2 has {2,3,4} (3 from U)
        # e3 has {1,4} (2 from U), e4 has none
        self.assertEqual(len(multiego), 3)
    
    def test_get_fractured_mren_alpha_one(self):
        """Test Fractured MultiEgo with alpha=1.0 (all egos must be present)."""
        U = {1, 2}
        alpha = 1.0
        multiego = get_fractured_multiego(self.h, U, start=0, alpha=alpha)
        
        # Only e1 has both 1 and 2
        self.assertEqual(len(multiego), 1)
        self.assertEqual(frozenset(multiego[0]), frozenset({1, 2, 3}))
    
    def test_get_fractured_mren_invalid_alpha(self):
        """Test Fractured MultiEgo with invalid alpha values."""
        U = {1, 2}
        
        # Alpha = 0 should return empty
        multiego = get_fractured_multiego(self.h, U, start=0, alpha=0)
        self.assertEqual(len(multiego), 0)
        
        # Alpha > 1 should return empty
        multiego = get_fractured_multiego(self.h, U, start=0, alpha=1.5)
        self.assertEqual(len(multiego), 0)
    
    def test_get_core_mren_beta_half(self):
        """Test Core MultiEgo with beta=0.5."""
        U = {1, 2, 3}
        beta = 0.5  # U nodes must be at least 50% of hyperedge
        multiego = get_core_multiego(self.h, U, start=0, beta=beta)
        
        # e1: {1,2,3} -> 3/3 = 100% >= 50% ✓
        # e2: {2,3,4} -> 2/3 = 66% >= 50% ✓
        # e3: {1,4,5} -> 1/3 = 33% < 50% ✗
        # e4: {5,6} -> 0/2 = 0% < 50% ✗
        self.assertEqual(len(multiego), 2)
    
    def test_get_core_mren_beta_one(self):
        """Test Core MultiEgo with beta=1.0 (U must be the entire hyperedge)."""
        U = {1, 2, 3}
        beta = 1.0
        multiego = get_core_multiego(self.h, U, start=0, beta=beta)
        
        # Only e1 has exactly {1,2,3}
        self.assertEqual(len(multiego), 1)
        self.assertEqual(frozenset(multiego[0]), frozenset({1, 2, 3}))
    
    def test_get_core_mren_invalid_beta(self):
        """Test Core MultiEgo with invalid beta values."""
        U = {1, 2}
        
        # Beta = 0 should return empty
        multiego = get_core_multiego(self.h, U, start=0, beta=0)
        self.assertEqual(len(multiego), 0)
        
        # Beta > 1 should return empty
        multiego = get_core_multiego(self.h, U, start=0, beta=1.5)
        self.assertEqual(len(multiego), 0)


class TestMultiEgoSimilarity(unittest.TestCase):
    """Test MultiEgo similarity measures."""
    
    def test_jaccard_identical_mrens(self):
        """Test Jaccard similarity of identical MultiEgos."""
        multiego1 = [{1, 2}, {2, 3}, {3, 4}]
        multiego2 = [{1, 2}, {2, 3}, {3, 4}]
        
        sim = jaccard_similarity(multiego1, multiego2)
        self.assertEqual(sim, 1.0)
    
    def test_jaccard_disjoint_mrens(self):
        """Test Jaccard similarity of disjoint MultiEgos."""
        multiego1 = [{1, 2}, {2, 3}]
        multiego2 = [{4, 5}, {5, 6}]
        
        sim = jaccard_similarity(multiego1, multiego2)
        self.assertEqual(sim, 0.0)
    
    def test_jaccard_partial_overlap(self):
        """Test Jaccard similarity with partial overlap."""
        multiego1 = [{1, 2}, {2, 3}, {3, 4}]
        multiego2 = [{1, 2}, {4, 5}]
        
        sim = jaccard_similarity(multiego1, multiego2)
        # Intersection: 1, Union: 4 -> 1/4 = 0.25
        self.assertAlmostEqual(sim, 0.25, places=5)
    
    def test_jaccard_empty_mrens(self):
        """Test Jaccard similarity with empty MultiEgos."""
        multiego1 = []
        multiego2 = []
        
        sim = jaccard_similarity(multiego1, multiego2)
        self.assertEqual(sim, 0.0)
    
    def test_jaccard_one_empty(self):
        """Test Jaccard similarity when one MultiEgo is empty."""
        multiego1 = [{1, 2}]
        multiego2 = []
        
        sim = jaccard_similarity(multiego1, multiego2)
        self.assertEqual(sim, 0.0)
    
    def test_minimum_overlapping_identical(self):
        """Test minimum overlapping similarity of identical MultiEgos."""
        multiego1 = [{1, 2}, {2, 3}]
        multiego2 = [{1, 2}, {2, 3}]
        
        sim = minimum_overlapping_similarity(multiego1, multiego2)
        self.assertEqual(sim, 1.0)
    
    def test_minimum_overlapping_partial(self):
        """Test minimum overlapping similarity with partial overlap."""
        multiego1 = [{1, 2}, {2, 3}, {3, 4}]
        multiego2 = [{1, 2}, {4, 5}]
        
        sim = minimum_overlapping_similarity(multiego1, multiego2)
        # Intersection: 1, Min size: 2 -> 1/2 = 0.5
        self.assertAlmostEqual(sim, 0.5, places=5)
    
    def test_minimum_overlapping_empty(self):
        """Test minimum overlapping similarity with empty MultiEgo."""
        multiego1 = [{1, 2}]
        multiego2 = []
        
        sim = minimum_overlapping_similarity(multiego1, multiego2)
        self.assertEqual(sim, 0.0)
    
    def test_delta_identical(self):
        """Test delta similarity of identical MultiEgos."""
        multiego1 = [{1, 2, 3}, {4, 5}]
        multiego2 = [{1, 2, 3}, {4, 5}]
        
        sim = delta_similarity(multiego1, multiego2)
        self.assertEqual(sim, 1.0)
    
    def test_delta_disjoint(self):
        """Test delta similarity of disjoint MultiEgos."""
        multiego1 = [{1, 2}]
        multiego2 = [{3, 4}]
        
        sim = delta_similarity(multiego1, multiego2)
        self.assertEqual(sim, 0.0)
    
    def test_delta_partial_overlap(self):
        """Test delta similarity with partial node overlap."""
        multiego1 = [{1, 2, 3}]
        multiego2 = [{1, 2, 4}, {5, 6}]
        
        sim = delta_similarity(multiego1, multiego2)
        # Best match for {1,2,3} is {1,2,4}: Jaccard = 2/4 = 0.5
        # Normalized by larger MultiEgo size: 0.5/2 = 0.25
        self.assertAlmostEqual(sim, 0.25, places=5)
    
    def test_delta_empty(self):
        """Test delta similarity with empty MultiEgos."""
        multiego1 = []
        multiego2 = [{1, 2}]
        
        sim = delta_similarity(multiego1, multiego2)
        self.assertEqual(sim, 0.0)
    
    def test_delta_asymmetric(self):
        """Test that delta similarity handles size differences."""
        multiego1 = [{1, 2}]
        multiego2 = [{1, 2}, {3, 4}, {5, 6}]
        
        # Should swap so multiego1 is smaller
        sim = delta_similarity(multiego1, multiego2)
        # Perfect match for {1,2}, normalized by 3
        self.assertAlmostEqual(sim, 1.0/3.0, places=5)


class TestMultiEgoIntegration(unittest.TestCase):
    """Integration tests for MultiEgo functionality."""
    
    def setUp(self):
        """Create a more complex hypergraph."""
        self.h = ASH()
        
        # Create a richer structure
        self.h.add_hyperedge([1, 2, 3, 4], start=0)
        self.h.add_hyperedge([2, 3, 5], start=0)
        self.h.add_hyperedge([1, 4, 6], start=0)
        self.h.add_hyperedge([5, 6, 7], start=0)
        self.h.add_hyperedge([7, 8], start=0)
    
    def test_mren_extraction_and_similarity(self):
        """Test extracting MultiEgos and computing similarity."""
        U1 = {1, 2}
        U2 = {2, 3}
        
        multiego1 = get_multiego(self.h, U1, start=0)
        multiego2 = get_multiego(self.h, U2, start=0)
        
        # Both should have non-empty MultiEgos
        self.assertGreater(len(multiego1), 0)
        self.assertGreater(len(multiego2), 0)
        
        # Compute similarities
        jaccard = jaccard_similarity(multiego1, multiego2)
        min_overlap = minimum_overlapping_similarity(multiego1, multiego2)
        delta = delta_similarity(multiego1, multiego2)
        
        # All similarities should be in [0, 1]
        self.assertGreaterEqual(jaccard, 0.0)
        self.assertLessEqual(jaccard, 1.0)
        self.assertGreaterEqual(min_overlap, 0.0)
        self.assertLessEqual(min_overlap, 1.0)
        self.assertGreaterEqual(delta, 0.0)
        self.assertLessEqual(delta, 1.0)
    
    def test_compare_mren_variants(self):
        """Test that MultiEgo variants produce nested results."""
        U = {1, 2, 3}
        
        multiego_standard = get_multiego(self.h, U, start=0)
        multiego_fractured = get_fractured_multiego(self.h, U, start=0, alpha=0.67)
        multiego_core = get_core_multiego(self.h, U, start=0, beta=0.5)
        
        # Standard MultiEgo should be largest (most inclusive)
        self.assertGreaterEqual(len(multiego_standard), len(multiego_fractured))
        
        # All should have at least some edges
        self.assertGreater(len(multiego_standard), 0)


if __name__ == '__main__':
    unittest.main()
