import networkx as nx

from ash_model import ASH


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
    Returns the betweenness centrality of the nodes in the line graph of the hypergraph.
    If `edges` is True, the function computes the betweenness centrality for hyperedges (the nodes of the line graph).
    If `edges` is False, it computes the betweenness centrality for nodes by first converting the hypergraph to its dual.

    :param h: the ASH instance
    :param s: minimum hyperedge overlap size for paths
    :param start: start time of the interval
    :param end: end time of the interval
    :param edges: if True, compute for hyperedges; if False, compute for nodes
    :param normalized: if True, normalize the betweenness centrality values
    :param weight: if True, use edge weights for the betweenness centrality calculation

    :return: a dictionary mapping node IDs (or edge IDs if `edges` is True) to their betweenness centrality values

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
    >>> head3 = sorted(list(s_betweenness_centrality(h, 1, start=0, end=0).items()))[:3]
    >>> head3
    [('e1', 0.0), ('e10', 0.021109461571662495), ('e100', 0.0035032472389105243)]
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
    Returns the closeness centrality of the nodes in the line graph of the hypergraph.
    If `edges` is True, the function computes the closeness centrality for hyperedges (the nodes of the line graph).
    If `edges` is False, it computes the closeness centrality for nodes by first converting the hypergraph to its dual.

    :param h: the ASH instance
    :param s: minimum hyperedge overlap size for paths
    :param start: start time of the interval
    :param end: end time of the interval
    :param edges: if True, compute for hyperedges; if False, compute for nodes

    :return: a dictionary mapping node IDs (or edge IDs if `edges` is True) to their closeness centrality values

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
    >>> head3 = sorted(list(s_closeness_centrality(h, 1, start=0, end=0).items()))[:3]
    >>> head3
    [('e1', 0.4563758389261745), ('e10', 0.5483870967741935), ('e100', 0.40556660039761433)]
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
    Returns the eccentricity of the nodes in the line graph of the hypergraph.
    If `edges` is True, the function computes the eccentricity for hyperedges (the nodes of the line graph).
    If `edges` is False, it computes the eccentricity for nodes by first converting the hypergraph to its dual.

    :param h: the ASH instance
    :param s: minimum hyperedge overlap size for paths
    :param start: start time of the interval
    :param end: end time of the interval
    :param edges: if True, compute for hyperedges; if False, compute for nodes

    :return: a dictionary mapping node IDs (or edge IDs if `edges` is True) to their eccentricity values

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
    >>> head3 = sorted(list(s_eccentricity(h, 1, start=0, end=0).items()))[:3]
    >>> head3
    [('e1', 4), ('e10', 3), ('e100', 4)]
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
    Returns the harmonic centrality of the nodes in the line graph of the hypergraph.
    If `edges` is True, the function computes the harmonic centrality for hyperedges (
    the nodes of the line graph).
    If `edges` is False, it computes the harmonic centrality for nodes by first converting
    the hypergraph to its dual.

    :param h: the ASH instance
    :param s: minimum hyperedge overlap size for paths
    :param start: start time of the interval
    :param end: end time of the interval
    :param edges: if True, compute for hyperedges; if False, compute for nodes

    :return: a dictionary mapping node IDs (or edge IDs if `edges` is True)
        to their harmonic centrality values

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
    >>> head3 = sorted(list(s_harmonic_centrality(h, 1, start=0, end=0).items()))[:3]
    >>> head3
    [('e1', 105.58333333333324), ('e10', 127.66666666666663), ('e100', 89.99999999999993)]

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
    Returns the Katz centrality of the nodes in the line graph of the hypergraph.
    If `edges` is True, the function computes the Katz centrality for hyperedges (
    the nodes of the line graph).
    If `edges` is False, it computes the Katz centrality for nodes by first converting
    the hypergraph to its dual.

    :param h: the ASH instance
    :param s: minimum hyperedge overlap size for paths
    :param start: start time of the interval
    :param end: end time of the interval
    :param edges: if True, compute for hyperedges; if False, compute for nodes
    :param normalized: if True, normalize the Katz centrality values
    :param alpha: attenuation factor for the Katz centrality
    :param beta: scaling factor for the Katz centrality
    :param weight: if True, use edge weights for the Katz centrality calculation

    :return: a dictionary mapping node IDs (or edge IDs if `edges` is True) to their Katz centrality values

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
    >>> head3 = sorted(list(s_katz(h, 1, start=0, end=0).items()))[:3]
    >>> [(k, float(v)) for k,v in head3]
    [('e1', 0.015921892363685495), ('e10', -0.01107345978999999), ('e100', 0.03526804656510612)]

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
    Returns the load centrality of the nodes in the line graph of the hypergraph.
    If `edges` is True, the function computes the load centrality for hyperedges (
    the nodes of the line graph).
    If `edges` is False, it computes the load centrality for nodes by first converting
    the hypergraph to its dual.

    :param h: the ASH instance
    :param s: minimum hyperedge overlap size for paths
    :param start: start time of the interval
    :param end: end time of the interval
    :param edges: if True, compute for hyperedges; if False, compute for nodes
    :param normalized: if True, normalize the load centrality values
    :param weight: if True, use edge weights for the load centrality calculation

    :return: a dictionary mapping node IDs (or edge IDs if `edges` is True) to their load centrality values

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
    >>> head3 = sorted(list(s_load_centrality(h, 1, start=0, end=0).items()))[:3]
    >>> head3
    [('e1', 0.0), ('e10', 0.017716537175626074), ('e100', 0.004101958933395599)]
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
    Returns the eigenvector centrality of the nodes in the line graph of the hypergraph.
    If `edges` is True, the function computes the eigenvector centrality for hyperedges
    (the nodes of the line graph).
    If `edges` is False, it computes the eigenvector centrality for nodes by first
    converting the hypergraph to its dual.

    :param h: the ASH instance
    :param s: minimum hyperedge overlap size for paths
    :param start: start time of the interval
    :param end: end time of the interval
    :param edges: if True, compute for hyperedges; if False, compute for nodes
    :param weight: if True, use edge weights for the eigenvector centrality calculation
    :param max_iter: maximum number of iterations for the eigenvector centrality calculation
    :param tol: tolerance for convergence in the eigenvector centrality calculation

    :return: a dictionary mapping node IDs (or edge IDs if `edges` is True) to their eigenvector centrality values

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
    >>> head3 = sorted(list(s_eigenvector_centrality(h, 1, start=0, end=0).items()))[:3]
    >>> [(k, float(v)) for k,v in head3]
    [('e1', 0.12621433668406887), ('e10', 0.1777592407633311), ('e100', 0.010063200782658498)]
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
    Returns the information centrality of the nodes in the line graph of the hypergraph.
    If `edges` is True, the function computes the information centrality for hyperedges
    (the nodes of the line graph).
    If `edges` is False, it computes the information centrality for nodes by first
    converting the hypergraph to its dual.

    :param h: the ASH instance
    :param s: minimum hyperedge overlap size for paths
    :param start: start time of the interval
    :param end: end time of the interval
    :param edges: if True, compute for hyperedges; if False, compute for nodes
    :param weight: if True, use edge weights for the information centrality calculation

    :return: a dictionary mapping node IDs (or edge IDs if `edges` is True) to their information centrality values

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
    >>> head3 = sorted(list(s_information_centrality(h, 1, start=0, end=0).items()))[:3]
    >>> head3
    [('e1', 0.036151507093556814), ('e10', 0.04152134988072226), ('e100', 0.024272445362294077)]
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
    Returns the second-order centrality of the nodes in the line graph of the hypergraph.
    If `edges` is True, the function computes the second-order centrality for hyperedges
    (the nodes of the line graph).
    If `edges` is False, it computes the second-order centrality for nodes by first
    converting the hypergraph to its dual.

    :param h: the ASH instance
    :param s: minimum hyperedge overlap size for paths
    :param start: start time of the interval
    :param end: end time of the interval
    :param edges: if True, compute for hyperedges; if False, compute for nodes

    :return: a dictionary mapping node IDs (or edge IDs if `edges` is True) to their second-order centrality values

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
    >>> head3 = sorted(list(s_second_order_centrality(h, 1, start=0, end=0).items()))[:3]
    >>> head3
    [('e1', 373.1451853808109), ('e10', 226.34942005127067), ('e100', 687.213714885934)]
    """
    lg, node_to_eid = __s_linegraph(h, s, start, end, edges)

    res = nx.second_order_centrality(lg)
    if node_to_eid is None:
        return res
    else:
        eid_to_node = {v: k for k, v in node_to_eid.items()}
        return {eid_to_node[k]: v for k, v in res.items()}
