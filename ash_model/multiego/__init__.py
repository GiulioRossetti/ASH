"""
Multi-Ego Networks for temporal hypergraphs.

This module provides functionality to extract and analyze Multi-Ego Networks,
which are hypergraph ego networks with multiple root nodes (egos).
This generalizes the single-node ego network (accessible via ASH.star()) to sets of root nodes.
"""

from .core import get_multiego, get_fractured_multiego, get_core_multiego
from .similarity import (
    jaccard_similarity,
    minimum_overlapping_similarity,
    delta_similarity,
)

__all__ = [
    "get_multiego",
    "get_fractured_multiego",
    "get_core_multiego",
    "jaccard_similarity",
    "minimum_overlapping_similarity",
    "delta_similarity",
]
