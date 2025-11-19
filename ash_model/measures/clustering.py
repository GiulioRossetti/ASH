from math import comb
from typing import Optional

from ash_model.paths import ASH, is_s_path


def s_local_clustering_coefficient(
    h: ASH,
    s: int,
    hyperedge_id: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> float:
    """
    Compute the local clustering coefficient of a hyperedge within the s-overlap line graph of a hypergraph.

    The local clustering coefficient is defined as the ratio of the number of edges
    actually present among the neighbors of the given node to the maximum possible
    number of edges among those neighbors.

    :param h:            an ASH instance
    :param s:            minimum hyperedge overlap size for projection
    :param hyperedge_id: identifier of the hyperedge in the line graph
    :param start:        optional start time (inclusive)
    :param end:          optional end time (inclusive)

    :return:             local clustering coefficient in [0,1]

    Examples
    --------
    >>> import numpy as np, networkx as nx
    >>> from ash_model.utils.networkx import from_networkx_maximal_cliques_list
    >>> Gs = [nx.barabasi_albert_graph(100, 3, seed=i) for i in range(10)]
    >>> rng = np.random.default_rng(42)
    >>> for G in Gs:
    ...     for n in G.nodes():
    ...         G.nodes[n]['color'] = 'red' if rng.integers(0, 2) == 0 else 'blue'
    >>> h = from_networkx_maximal_cliques_list(Gs)
    >>> tid = 0
    >>> he0 = next(iter(h.hyperedges(start=tid, end=tid)))
    >>> round(s_local_clustering_coefficient(h, 1, he0, start=tid, end=tid), 12)
    0.801169590643
    """
    # Build the s-overlap line graph once
    lg = h.s_line_graph(s, start, end)
    if hyperedge_id not in lg:
        return 0.0

    # Neighbors in the line graph correspond to overlapping hyperedges
    neighbors = list(lg.neighbors(hyperedge_id))
    k = len(neighbors)
    if k < 2:
        return 0.0

    # Induced subgraph on those neighbors
    subgraph = lg.subgraph(neighbors)
    actual_edges = subgraph.number_of_edges()
    possible_edges = comb(k, 2)

    return actual_edges / possible_edges


def average_s_local_clustering_coefficient(
    h: ASH, s: int, start: Optional[int] = None, end: Optional[int] = None
) -> float:
    """
    Compute the average local clustering coefficient across all hyperedges
    in the s-overlap line graph.

    :param h:     an ASH instance
    :param s:     minimum hyperedge overlap size for projection
    :param start: optional start time (inclusive)
    :param end:   optional end time (inclusive)

    :return:      average local clustering coefficient in [0,1], or 0 if no nodes

    Examples
    --------
    >>> import numpy as np, networkx as nx
    >>> from ash_model.utils.networkx import from_networkx_maximal_cliques_list
    >>> Gs = [nx.barabasi_albert_graph(100, 3, seed=i) for i in range(10)]
    >>> rng = np.random.default_rng(42)
    >>> for G in Gs:
    ...     for n in G.nodes():
    ...         G.nodes[n]['color'] = 'red' if rng.integers(0, 2) == 0 else 'blue'
    >>> h = from_networkx_maximal_cliques_list(Gs)
    >>> round(average_s_local_clustering_coefficient(h, 1, start=0, end=0), 12)
    0.581891366366
    """
    lg = h.s_line_graph(s, start, end)
    n = lg.number_of_nodes()
    if n == 0:
        return 0.0

    total = 0.0
    for node in lg.nodes:
        total += s_local_clustering_coefficient(h, s, node, start, end)
    return total / n


def s_intersections(
    h: ASH, s: int, start: Optional[int] = None, end: Optional[int] = None
) -> int:
    """
    Count the number of s-overlap intersections (edges) in the hypergraph's
    s-overlap line graph.

    :param h:     an ASH instance
    :param s:     minimum hyperedge overlap size
    :param start: optional start time (inclusive)
    :param end:   optional end time (inclusive)

    :return:      number of intersections of size >= s

    Examples
    --------
    >>> import numpy as np, networkx as nx
    >>> from ash_model.utils.networkx import from_networkx_maximal_cliques_list
    >>> Gs = [nx.barabasi_albert_graph(100, 3, seed=i) for i in range(10)]
    >>> rng = np.random.default_rng(42)
    >>> for G in Gs:
    ...     for n in G.nodes():
    ...         G.nodes[n]['color'] = 'red' if rng.integers(0, 2) == 0 else 'blue'
    >>> h = from_networkx_maximal_cliques_list(Gs)
    >>> s_intersections(h, 1, start=0, end=0)
    2091
    """
    return h.s_line_graph(s, start, end).number_of_edges()


def inclusiveness(
    h: ASH, start: Optional[int] = None, end: Optional[int] = None
) -> float:
    """
    Computes the inclusiveness of the hypergraph over [start, end], defined as:

        (# of non-facet hyperedges) / (total # of hyperedges)

    A *facet* hyperedge is one that is *not* a subset of any other hyperedge.
    Non-facet hyperedges are those that *are* contained in at least one strictly larger hyperedge.

    :param h:      an ASH instance
    :param start:  optional start time (inclusive)
    :param end:    optional end time (inclusive)

    :return:       a float in [0,1], or 0.0 if there are no hyperedges

    Examples
    --------
    >>> import numpy as np, networkx as nx
    >>> from ash_model.utils.networkx import from_networkx_maximal_cliques_list
    >>> Gs = [nx.barabasi_albert_graph(100, 3, seed=i) for i in range(10)]
    >>> rng = np.random.default_rng(42)
    >>> for G in Gs:
    ...     for n in G.nodes():
    ...         G.nodes[n]['color'] = 'red' if rng.integers(0, 2) == 0 else 'blue'
    >>> h = from_networkx_maximal_cliques_list(Gs)
    >>> inclusiveness(h, start=0, end=0)
    0.0
    """
    node_sets = [set(he) for he in h.hyperedges(start, end, as_ids=False)]
    total = len(node_sets)
    if total == 0:
        return 0.0

    # Sort indices by descending set size to speed up subset checks
    idxs = sorted(range(total), key=lambda i: len(node_sets[i]), reverse=True)
    is_non_facet = [False] * total

    for i in range(total):
        for j in idxs:
            if i == j:
                continue
            if not is_non_facet[i] and node_sets[i].issubset(node_sets[j]):
                is_non_facet[i] = True
                break

    non_facet_count = sum(is_non_facet)
    return non_facet_count / total
