"""
Similarity measures for comparing Multi-Ego Networks.
"""

from typing import List, Set


def jaccard_similarity(multiego1: List[Set[int]], multiego2: List[Set[int]]) -> float:
    """
    Compute the Jaccard similarity between two Multi-Ego Networks.

    The Jaccard similarity is defined as the size of the intersection divided by
    the size of the union of the two Multi-Ego Network hyperedge sets.

    :param multiego1: first Multi-Ego Network (list of hyperedges as sets of node IDs)
    :param multiego2: second Multi-Ego Network (list of hyperedges as sets of node IDs)

    :return: Jaccard similarity score in [0, 1]. Returns 0.0 if both Multi-Ego Networks are empty.

    Examples
    --------
    >>> multiego1 = [{1, 2}, {2, 3}, {1, 3}]
    >>> multiego2 = [{1, 2}, {3, 4}]
    >>> sim = jaccard_similarity(multiego1, multiego2)
    >>> print(f"Jaccard similarity: {sim:.3f}")
    """
    if not multiego1 and not multiego2:
        return 0.0

    # Convert hyperedges to frozensets for set operations
    set1 = {frozenset(edge) for edge in multiego1}
    set2 = {frozenset(edge) for edge in multiego2}

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0:
        return 0.0

    return intersection / union


def minimum_overlapping_similarity(
    multiego1: List[Set[int]], multiego2: List[Set[int]]
) -> float:
    """
    Compute the minimum overlapping similarity between two Multi-Ego Networks.

    This similarity is defined as the size of the intersection divided by
    the minimum size of the two Multi-Ego Network hyperedge sets.

    :param multiego1: first Multi-Ego Network (list of hyperedges as sets of node IDs)
    :param multiego2: second Multi-Ego Network (list of hyperedges as sets of node IDs)

    :return: minimum overlapping similarity score in [0, 1]. Returns 0.0 if either Multi-Ego Network is empty.

    Examples
    --------
    >>> multiego1 = [{1, 2}, {2, 3}, {1, 3}]
    >>> multiego2 = [{1, 2}, {3, 4}]
    >>> sim = minimum_overlapping_similarity(multiego1, multiego2)
    >>> print(f"Min overlapping similarity: {sim:.3f}")
    """
    if not multiego1 or not multiego2:
        return 0.0

    # Convert hyperedges to frozensets for set operations
    set1 = {frozenset(edge) for edge in multiego1}
    set2 = {frozenset(edge) for edge in multiego2}

    intersection = len(set1.intersection(set2))
    min_size = min(len(set1), len(set2))

    if min_size == 0:
        return 0.0

    return intersection / min_size


def delta_similarity(multiego1: List[Set[int]], multiego2: List[Set[int]]) -> float:
    """
    Compute the delta similarity between two Multi-Ego Networks.

    This is a weighted Jaccard similarity that considers the best matching
    between hyperedges based on their node overlap. For each hyperedge in the
    smaller Multi-Ego Network, it finds the best match in the larger one based on Jaccard
    similarity at the node level.

    :param multiego1: first Multi-Ego Network (list of hyperedges as sets of node IDs)
    :param multiego2: second Multi-Ego Network (list of hyperedges as sets of node IDs)

    :return: delta similarity score in [0, 1]. Returns 0.0 if either Multi-Ego Network is empty.

    Examples
    --------
    >>> multiego1 = [{1, 2, 3}, {2, 3, 4}]
    >>> multiego2 = [{1, 2}, {3, 4, 5}]
    >>> sim = delta_similarity(multiego1, multiego2)
    >>> print(f"Delta similarity: {sim:.3f}")
    """
    if not multiego1 or not multiego2:
        return 0.0

    # Ensure multiego1 is the smaller one
    if len(multiego1) > len(multiego2):
        multiego1, multiego2 = multiego2, multiego1

    total_similarity = 0.0
    used_edges = set()

    # For each edge in the smaller Multi-Ego Network, find the best match in the larger one
    for edge1 in multiego1:
        set1 = set(edge1)
        best_match_sim = 0.0
        best_match_idx = -1

        for idx, edge2 in enumerate(multiego2):
            if idx in used_edges:
                continue

            set2 = set(edge2)

            # Compute Jaccard similarity at the node level
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))

            if union > 0:
                node_jaccard = intersection / union
                if node_jaccard > best_match_sim:
                    best_match_sim = node_jaccard
                    best_match_idx = idx

        if best_match_idx >= 0:
            used_edges.add(best_match_idx)
            total_similarity += best_match_sim

    # Normalize by the size of the larger Multi-Ego Network
    if len(multiego2) == 0:
        return 0.0

    return total_similarity / len(multiego2)
