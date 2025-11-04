"""
Core Multi-Ego Network extraction functions for temporal hypergraphs.
"""

from typing import Set, List, Optional
from ..classes import ASH


def get_multiego(h: ASH, U: Set[int], start: int, end: Optional[int] = None) -> List[Set[int]]:
    """
    Extract the Multi-Ego Network for a set of ego nodes U within a time window.
    
    A Multi-Ego Network contains all hyperedges that include at least one node from the ego set U.
    This generalizes the single-node ego network (ASH.star()) to multiple root nodes.
    
    Parameters
    ----------
    h : ASH
        The temporal hypergraph instance.
    U : Set[int]
        Set of ego nodes (root nodes).
    start : int
        Start time of the query window.
    end : Optional[int]
        End time of the query window (inclusive). If None, only the start time is considered.
        
    Returns
    -------
    List[Set[int]]
        List of hyperedges (as sets of node IDs) forming the Multi-Ego Network.
        
    Examples
    --------
    >>> h = ASH()
    >>> # ... add hyperedges ...
    >>> U = {1, 2}  # Two ego nodes
    >>> multiego = get_multiego(h, U, start=0)  # Single snapshot
    >>> multiego = get_multiego(h, U, start=0, end=5)  # Time window [0,5]
    >>> print(f"Multi-Ego Network contains {len(multiego)} hyperedges")
    """
    if not U:
        return []
    
    multiego = []
    seen_edges = set()  # Track unique hyperedges across time window
    
    # Get all hyperedges in the time window
    hyperedge_ids = h.hyperedges(start=start, end=end, as_ids=True)
    
    # Iterate through all hyperedges in the time window
    for edge_id in hyperedge_ids:
        edge_nodes = frozenset(h.get_hyperedge_nodes(edge_id))
        
        # Check if the hyperedge contains at least one node from U
        # and hasn't been added yet (to avoid duplicates in time window)
        if U.intersection(edge_nodes) and edge_nodes not in seen_edges:
            multiego.append(set(edge_nodes))
            seen_edges.add(edge_nodes)
    
    return multiego


def get_fractured_multiego(h: ASH, U: Set[int], start: int, end: Optional[int] = None, alpha: float = 0.5) -> List[Set[int]]:
    """
    Extract the Fractured Multi-Ego Network for a set of ego nodes U within a time window.
    
    A Fractured Multi-Ego Network contains all hyperedges where at least alpha*|U| nodes
    from the ego set U are present.
    
    Parameters
    ----------
    h : ASH
        The temporal hypergraph instance.
    U : Set[int]
        Set of ego nodes (root nodes).
    start : int
        Start time of the query window.
    end : Optional[int]
        End time of the query window (inclusive). If None, only the start time is considered.
    alpha : float
        Fraction threshold (0 < alpha <= 1). A hyperedge is included if it
        contains at least alpha*|U| nodes from U.
        
    Returns
    -------
    List[Set[int]]
        List of hyperedges (as sets of node IDs) forming the Fractured Multi-Ego Network.
        
    Examples
    --------
    >>> h = ASH()
    >>> # ... add hyperedges ...
    >>> U = {1, 2, 3, 4}  # Four ego nodes
    >>> # Include hyperedges with at least 50% of U (2 nodes)
    >>> multiego = get_fractured_multiego(h, U, start=0, alpha=0.5)
    >>> multiego = get_fractured_multiego(h, U, start=0, end=5, alpha=0.75)
    """
    if not U or alpha <= 0 or alpha > 1:
        return []
    
    multiego = []
    seen_edges = set()  # Track unique hyperedges across time window
    threshold = alpha * len(U)
    
    # Get all hyperedges in the time window
    hyperedge_ids = h.hyperedges(start=start, end=end, as_ids=True)
    
    # Iterate through all hyperedges in the time window
    for edge_id in hyperedge_ids:
        edge_nodes = frozenset(h.get_hyperedge_nodes(edge_id))
        
        # Count how many nodes from U are in the hyperedge
        intersection_size = len(U.intersection(edge_nodes))
        
        # Check if the hyperedge contains at least alpha*|U| nodes from U
        # and hasn't been added yet
        if intersection_size >= threshold and edge_nodes not in seen_edges:
            multiego.append(set(edge_nodes))
            seen_edges.add(edge_nodes)
    
    return multiego


def get_core_multiego(h: ASH, U: Set[int], start: int, end: Optional[int] = None, beta: float = 0.5) -> List[Set[int]]:
    """
    Extract the Core Multi-Ego Network for a set of ego nodes U within a time window.
    
    A Core Multi-Ego Network contains all hyperedges where the nodes from U represent
    at least beta fraction of the hyperedge size.
    
    Parameters
    ----------
    h : ASH
        The temporal hypergraph instance.
    U : Set[int]
        Set of ego nodes (root nodes).
    start : int
        Start time of the query window.
    end : Optional[int]
        End time of the query window (inclusive). If None, only the start time is considered.
    beta : float
        Fraction threshold (0 < beta <= 1). A hyperedge is included if
        nodes from U represent at least beta*|hyperedge| of its nodes.
        
    Returns
    -------
    List[Set[int]]
        List of hyperedges (as sets of node IDs) forming the Core Multi-Ego Network.
        
    Examples
    --------
    >>> h = ASH()
    >>> # ... add hyperedges ...
    >>> U = {1, 2, 3}  # Three ego nodes
    >>> # Include hyperedges where U nodes are at least 60% of the hyperedge
    >>> multiego = get_core_multiego(h, U, start=0, beta=0.6)
    >>> multiego = get_core_multiego(h, U, start=0, end=5, beta=0.5)
    """
    if not U or beta <= 0 or beta > 1:
        return []
    
    multiego = []
    seen_edges = set()  # Track unique hyperedges across time window
    
    # Get all hyperedges in the time window
    hyperedge_ids = h.hyperedges(start=start, end=end, as_ids=True)
    
    # Iterate through all hyperedges in the time window
    for edge_id in hyperedge_ids:
        edge_nodes = frozenset(h.get_hyperedge_nodes(edge_id))
        
        if len(edge_nodes) == 0:
            continue
        
        # Count how many nodes from U are in the hyperedge
        intersection_size = len(U.intersection(edge_nodes))
        
        # Check if U nodes represent at least beta fraction of the hyperedge
        # and hasn't been added yet
        threshold = beta * len(edge_nodes)
        if intersection_size >= threshold and edge_nodes not in seen_edges:
            multiego.append(set(edge_nodes))
            seen_edges.add(edge_nodes)
    
    return multiego
