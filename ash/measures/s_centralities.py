import networkx as nx

from ash import ASH


def __s_linegraph(
    h: ASH, s: int, start: int = None, end: int = None, edges: bool = True
) -> nx.Graph:
    """

    :param h:
    :param s:
    :param start:
    :param end:
    :param edges:
    :return:
    """
    if edges:
        lg = h.s_line_graph(s=s, start=start, end=end)
        node_to_eid = None
    else:
        d, node_to_eid = h.dual_hypergraph(start=start, end=end)
        lg = d.s_line_graph(s=s, start=start, end=end)
    return lg, node_to_eid


def s_betweenness_centrality(
    h: ASH,
    s: int,
    start: int = None,
    end: int = None,
    edges: bool = True,
    normalized: bool = True,
    weight: bool = False,
) -> dict:
    """
    The s_betweenness_centrality function computes the s-betweenness centrality for each node in a hypergraph. The
    betweenness centrality of a node is defined as the number of shortest s-paths from all vertices to all others
    that pass through that node.

    :param h: ASH instance
    :param s: minimum intersection between hyperedges to form a path
    :param start: Specify the start of the time window
    :param end: Specify the end of the time window
    :param edges: Determine whether to use the edges or nodes of the hypergraph
    :param normalized: Normalize the s-betweenness centrality values
    :param weight: Determine if the weight of the edges should be considered
    :return: A dictionary with the s-betweenness centrality of each node (or edge) in the hypergraph
    """

    lg, node_to_eid = __s_linegraph(h, s, start, end, edges)

    if weight:
        weight = "w"
    else:
        weight = None

    res = nx.betweenness_centrality(lg, normalized=normalized, weight=weight)
    if node_to_eid is None:
        return res
    else:
        eid_to_node = {v: k for k, v in node_to_eid.items()}
        return {eid_to_node[k]: v for k, v in res.items()}


def s_closeness_centrality(
    h: ASH, s: int, start: int = None, end: int = None, edges: bool = True
) -> dict:
    """
    The s_closeness_centrality function computes the s-closeness centrality of each node in a hypergraph.
    The closeness centrality is defined as the inverse of the sum of s-distances from a given node to all other nodes.

    :param h: ASH instance
    :param s: minimum intersection between hyperedges to form a path
    :param start: Specify the start of the time window
    :param end: Specify the end of the time window
    :param edges: Determine whether to use the edges or nodes of the hypergraph
    :return: A dictionary with the nodes/edges as keys and their s-closeness centrality as values
    """

    lg, node_to_eid = __s_linegraph(h, s, start, end, edges)

    res = nx.closeness_centrality(lg)
    if node_to_eid is None:
        return res
    else:
        eid_to_node = {v: k for k, v in node_to_eid.items()}
        return {eid_to_node[k]: v for k, v in res.items()}


def s_eccentricity(
    h: ASH, s: int, start: int = None, end: int = None, edges: bool = True
) -> dict:
    """
    The s_eccentricity function returns the s-eccentricity of each node in a given hypergraph.
    The s-eccentricity of a node is the maximum s-distance between that node and any other node.

    :param h: ASH instance
    :param s: minimum intersection between hyperedges to form a path
    :param start: Specify the start of the time window
    :param end: Specify the end of the time window
    :param edges: Determine whether to use the edges or nodes of the hypergraph
    :return: A dictionary with the nodes/edges as keys and their s-eccentricity as values
    """

    lg, node_to_eid = __s_linegraph(h, s, start, end, edges)

    res = nx.eccentricity(lg)
    if node_to_eid is None:
        return res
    else:
        eid_to_node = {v: k for k, v in node_to_eid.items()}
        return {eid_to_node[k]: v for k, v in res.items()}


def s_harmonic_centrality(
    h: ASH, s: int, start: int = None, end: int = None, edges: bool = True
) -> dict:
    """
    The s_harmonic_centrality function computes the s-harmonic centrality of each node in a hypergraph.
    The harmonic centrality is defined as the sum of the inverse of the s-distances between all nodes.

    :param h: ASH instance
    :param s: minimum intersection between hyperedges to form a path
    :param start: Specify the start of the time window
    :param end: Specify the end of the time window
    :param edges: Determine whether to use the edges or nodes of the hypergraph
    :return: A dictionary with the nodes/edges as keys and their s-harmonic centrality as values
    """

    lg, node_to_eid = __s_linegraph(h, s, start, end, edges)

    res = nx.harmonic_centrality(lg)
    if node_to_eid is None:
        return res
    else:
        eid_to_node = {v: k for k, v in node_to_eid.items()}
        return {eid_to_node[k]: v for k, v in res.items()}


def s_katz(
    h: ASH,
    s: int,
    start: int = None,
    end: int = None,
    edges: bool = True,
    normalized: bool = True,
    alpha: float = 0.1,
    beta: float = 1.0,
    weight: bool = False,
) -> dict:
    """
    The s_katz function computes the Katz s-centrality of all nodes in a hypergraph.


    :param h: ASH instance
    :param s: minimum intersection between hyperedges to form a path
    :param start: Specify the start of the time window
    :param end: Specify the end of the time window
    :param edges: Determine whether to use the edges or nodes of the hypergraph
    :param normalized: Normalize the centrality scores
    :param alpha: Control the rate of convergence
    :param beta: Control the influence of the number of s-paths on katz centrality
    :param weight: Determine whether or not the weight of each edge is used in the calculation
    :return: A dictionary with the katz centrality of each node/edge
    """

    lg, node_to_eid = __s_linegraph(h, s, start, end, edges)

    if weight:
        weight = "w"
    else:
        weight = None

    res = nx.katz_centrality_numpy(
        lg, normalized=normalized, alpha=alpha, beta=beta, weight=weight
    )
    if node_to_eid is None:
        return res
    else:
        eid_to_node = {v: k for k, v in node_to_eid.items()}
        return {eid_to_node[k]: v for k, v in res.items()}


def s_load_centrality(
    h: ASH,
    s: int,
    start: int = None,
    end: int = None,
    edges: bool = True,
    normalized: bool = True,
    weight: bool = False,
) -> dict:
    """
    The s_load_centrality function calculates the s-load centrality of all nodes in a hypergraph.
    The load centrality is defined as the fraction of all shortest s-paths that pass through a given node.

    :param h: ASH instance
    :param s: minimum intersection between hyperedges to form a path
    :param start: Specify the start of the time window
    :param end: Specify the end of the time window
    :param edges: Determine whether to use the edges or nodes of the hypergraph
    :param normalized: Normalize the centrality scores
    :param weight: Determine whether or not the weight of each edge is used in the calculation
    :return: A dictionary with the s-load centrality of each node/edge
    """

    lg, node_to_eid = __s_linegraph(h, s, start, end, edges)

    if weight:
        weight = "w"
    else:
        weight = None

    res = nx.load_centrality(lg, normalized=normalized, weight=weight)
    if node_to_eid is None:
        return res
    else:
        eid_to_node = {v: k for k, v in node_to_eid.items()}
        return {eid_to_node[k]: v for k, v in res.items()}


def s_eigenvector_centrality(
    h: ASH,
    s: int,
    start: int = None,
    end: int = None,
    edges: bool = True,
    weight: bool = False,
    max_iter: int = 50,
    tol: float = 0,
) -> dict:
    """
    The s_eigenvector_centrality function computes the eigenvector centrality for each node in a hypergraph.

    :param h: ASH instance
    :param s: minimum intersection between hyperedges to form a path
    :param start: Specify the start of the time window
    :param end: Specify the end of the time window
    :param edges: Determine whether to use the edges or nodes of the hypergraph
    :param normalized: Normalize the centrality scores
    :param weight: Determine whether the weight of each edge is used in the calculation
    :param max_iter: Set the maximum number of iterations in power method eigenvalue solver
    :param tol: Set the tolerance for convergence,
    :return: A dictionary with the eigenvector s-centrality of each node
    """

    lg, node_to_eid = __s_linegraph(h, s, start, end, edges)

    if weight:
        weight = "w"
    else:
        weight = None

    res = nx.eigenvector_centrality_numpy(lg, weight=weight, max_iter=max_iter, tol=tol)
    if node_to_eid is None:
        return res
    else:
        eid_to_node = {v: k for k, v in node_to_eid.items()}
        return {eid_to_node[k]: v for k, v in res.items()}


def s_information_centrality(
    h: ASH, s: int, start: int = None, end: int = None, edges: bool = True, weight=None
) -> dict:
    """
    The s_information_centrality function computes the information centrality of all nodes in a hypergraph The
    s-information centrality is defined as the entropy of the distribution over all s-paths from a to b, where a and
    b are nodes or hyperedges in the hypergraph.

    :param h: ASH instance
    :param s: minimum intersection between hyperedges to form a path
    :param start: Specify the start of the time window
    :param end: Specify the end of the time window
    :param edges: Determine whether to use the edges or nodes of the hypergraph
    :param weight: Determine whether the weight of each edge is used in the calculation
    :return: A dictionary with the nodes/edges as keys and their s-information centrality as values
    """

    lg, node_to_eid = __s_linegraph(h, s, start, end, edges)

    if weight:
        weight = "w"
    else:
        weight = None

    res = nx.information_centrality(lg, weight=weight)
    if node_to_eid is None:
        return res
    else:
        eid_to_node = {v: k for k, v in node_to_eid.items()}
        return {eid_to_node[k]: v for k, v in res.items()}


def s_second_order_centrality(
    h: ASH, s: int, start: int = None, end: int = None, edges: bool = True
) -> dict:
    """
    The s_second_order_centrality function computes the s-second order centrality of all nodes in a hypergraph The
    s-second order centrality for a node/edge is the standard deviation of the return times to that node(edge of a
    perpetual random s-walk on the hypergraph:


    :param h: Specify the graph that is being used
    :param s: Specify the source node
    :param start: Specify the start of a time interval,
    :param end: Specify the end of a time interval
    :param edges: Specify whether the edges of the line graph should be included in the
    :return: A dictionary of the second order centrality for each node
    """
    lg, node_to_eid = __s_linegraph(h, s, start, end, edges)

    res = nx.second_order_centrality(lg)
    if node_to_eid is None:
        return res
    else:
        eid_to_node = {v: k for k, v in node_to_eid.items()}
        return {eid_to_node[k]: v for k, v in res.items()}
