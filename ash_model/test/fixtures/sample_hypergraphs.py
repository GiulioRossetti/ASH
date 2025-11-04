"""Factory functions for creating sample hypergraphs used across tests."""

from ash_model import ASH, NProfile


def small_hypergraph():
    """Create a small hypergraph with 4 nodes and 2 hyperedges for basic testing."""
    h = ASH()
    h.add_hyperedge([1, 2, 3], start=0)
    h.add_hyperedge([2, 4], start=0)
    return h


def temporal_hypergraph():
    """Create a hypergraph with temporal dynamics for time-based testing."""
    h = ASH()
    h.add_hyperedge([1, 2, 3], start=0, end=4)  # e1: t in [0,4]
    h.add_hyperedge([1, 4], start=0, end=1)  # e2: t in [0,1]
    h.add_hyperedge([1, 2, 3, 4], start=2, end=3)  # e3: t in [2,3]
    h.add_hyperedge([1, 3, 4], start=2, end=3)  # e4: t in [2,3]
    h.add_hyperedge([3, 4], start=3, end=4)  # e5: t in [3,4]
    return h


def attributed_hypergraph():
    """Create a hypergraph with node attributes for testing attribute-based measures."""
    h = ASH()

    # Add nodes with attributes
    h.add_node(1, start=0, end=2, attr_dict={"group": "A", "age": 25})
    h.add_node(2, start=0, end=2, attr_dict={"group": "A", "age": 30})
    h.add_node(3, start=0, end=2, attr_dict={"group": "B", "age": 35})
    h.add_node(4, start=0, end=2, attr_dict={"group": "B", "age": 40})

    # Add hyperedges
    h.add_hyperedge([1, 2], start=0)
    h.add_hyperedge([2, 3], start=1)
    h.add_hyperedge([3, 4], start=2)
    h.add_hyperedge([1, 2, 3], start=1)

    return h


def dense_temporal_hypergraph():
    """Create a denser hypergraph for performance and stress testing."""
    h = ASH()
    for t in range(10):
        for i in range(1, 6):
            nodes = list(range(i, min(i + 3, 11)))
            h.add_hyperedge(nodes, start=t, end=t)
    return h


def hypergraph_with_profiles():
    """Create a hypergraph with rich node profiles for profile testing."""
    h = ASH()

    profile1 = NProfile(node_id=1, name="Alice", department="CS", publications=10)
    profile2 = NProfile(node_id=2, name="Bob", department="Math", publications=15)
    profile3 = NProfile(node_id=3, name="Charlie", department="CS", publications=8)

    h.add_node(1, start=0, end=2, attr_dict=profile1)
    h.add_node(2, start=0, end=2, attr_dict=profile2)
    h.add_node(3, start=0, end=2, attr_dict=profile3)

    h.add_hyperedge([1, 2, 3], start=0)
    h.add_hyperedge([1, 3], start=1)
    h.add_hyperedge([2, 3], start=2)

    return h


def bipartite_hypergraph():
    """Create a bipartite hypergraph for projection testing."""
    h = ASH()
    # Set A: nodes 1, 2, 3
    # Set B: nodes 4, 5, 6
    h.add_hyperedge([1, 4], start=0)
    h.add_hyperedge([1, 5], start=0)
    h.add_hyperedge([2, 5], start=0)
    h.add_hyperedge([2, 6], start=0)
    h.add_hyperedge([3, 6], start=0)
    return h
