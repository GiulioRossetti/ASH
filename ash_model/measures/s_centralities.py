from ash_model import ASH
import networkx as nx


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

    :param h:
    :param s:
    :param start:
    :param end:
    :param edges:
    :param normalized:
    :param weight:
    :return:
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

    :param h:
    :param s:
    :param start:
    :param end:
    :param edges:
    :return:
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

    :param h:
    :param s:
    :param start:
    :param end:
    :param edges:
    :return:
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

    :param h:
    :param s:
    :param start:
    :param end:
    :param edges:
    :return:
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

    :param h:
    :param s:
    :param start:
    :param end:
    :param edges:
    :param normalized:
    :param alpha:
    :param beta:
    :param weight:
    :return:
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

    :param h:
    :param s:
    :param start:
    :param end:
    :param edges:
    :param normalized:
    :param weight:
    :return:
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

    :param h:
    :param s:
    :param start:
    :param end:
    :param edges:
    :param weight:
    :param max_iter:
    :param tol:
    :return:
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

    :param h:
    :param s:
    :param start:
    :param end:
    :param edges:
    :param weight:
    :return:
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

    :param h:
    :param s:
    :param start:
    :param end:
    :param edges:
    :return:
    """
    lg, node_to_eid = __s_linegraph(h, s, start, end, edges)

    res = nx.second_order_centrality(lg)
    if node_to_eid is None:
        return res
    else:
        eid_to_node = {v: k for k, v in node_to_eid.items()}
        return {eid_to_node[k]: v for k, v in res.items()}
