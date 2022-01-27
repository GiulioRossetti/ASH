from ash import ASH
import networkx as nx
from collections import defaultdict, Counter
from itertools import combinations


def shortest_s_walk(
    h: ASH,
    s: int,
    hyperedge_a: str = None,
    hyperedge_b: str = None,
    start: int = None,
    end: int = None,
    weight: bool = False,
) -> object:
    """

    All returned paths include both the source and target in the path.

    If the source and target are both specified, return a single list of nodes in a shortest path from the source to the target.

    If only the source is specified, return a dictionary keyed by targets with a list of nodes in a shortest path from the source to one of the targets.

    If only the target is specified, return a dictionary keyed by sources with a list of nodes in a shortest path from one of the sources to the target.

    If neither the source nor target are specified return a dictionary of dictionaries with path[source][target]=[list of nodes in path].

    :param h:
    :param s:
    :param hyperedge_a:
    :param hyperedge_b:
    :param start:
    :param end:
    :param weight:
    :return:
    """

    g = h.s_line_graph(s, start, end)

    if hyperedge_a is not None and hyperedge_b is not None:
        if not g.has_node(hyperedge_a) or not g.has_node(hyperedge_b):
            return []
    elif hyperedge_a is not None and not g.has_node(hyperedge_a):
        return []

    if weight:
        return nx.shortest_path(g, source=hyperedge_a, target=hyperedge_b, weight="w")
    return nx.shortest_path(g, source=hyperedge_a, target=hyperedge_b)


def closed_s_walk(
    h: ASH, s: int, hyperedge_a: str = None, start: int = None, end: int = None
) -> object:
    """

    :param h:
    :param s:
    :param hyperedge_a:
    :param start:
    :param end:
    :return:
    """
    g = h.s_line_graph(s, start, end)

    if not g.has_node(hyperedge_a):
        return []
    return nx.cycle_basis(g, hyperedge_a)


def shortest_node_s_walk(
    h: ASH,
    s: int,
    node_a: int = None,
    node_b: int = None,
    start: int = None,
    end: int = None,
    weight: bool = False,
) -> object:
    """

    :param h:
    :param s:
    :param node_a:
    :param node_b:
    :param start:
    :param end:
    :param weight:
    :return:
    """
    h1, node_to_eid = h.dual_hypergraph(start=start, end=end)
    eid_to_node = {v: k for k, v in node_to_eid.items()}
    eid1, eid2 = node_a, node_b
    if node_a is not None:
        eid1 = node_to_eid[node_a]
    if node_b is not None:
        eid2 = node_to_eid[node_b]

    res = shortest_s_walk(h1, s, eid1, eid2, weight=weight)
    out = {}
    if isinstance(res, dict):
        for k, v in res.items():
            out[eid_to_node[k]] = [eid_to_node[i] for i in v]
        return out
    else:
        return [eid_to_node[i] for i in res]


def s_distance(
    h: ASH,
    s: int,
    hyperedge_a: str = None,
    hyperedge_b: str = None,
    start: int = None,
    end: int = None,
    weight: bool = False,
) -> object:
    """

    :param h:
    :param s:
    :param hyperedge_a:
    :param hyperedge_b:
    :param start:
    :param end:
    :param weight:
    :return:
    """
    g = h.s_line_graph(s, start, end)

    if hyperedge_a is not None and hyperedge_b is not None:
        if not g.has_node(hyperedge_a) or not g.has_node(hyperedge_b):
            return None
    elif hyperedge_a is not None and not g.has_node(hyperedge_a):
        return None

    if weight:
        return nx.shortest_path_length(
            g, source=hyperedge_a, target=hyperedge_b, weight="w"
        )
    return nx.shortest_path_length(g, source=hyperedge_a, target=hyperedge_b)


def node_s_distance(
    h: ASH,
    s: int,
    node_a: int = None,
    node_b: int = None,
    start: int = None,
    end: int = None,
    weight: bool = False,
) -> object:
    """

    :param h:
    :param s:
    :param node_a:
    :param node_b:
    :param start:
    :param end:
    :param weight:
    :return:
    """
    h1, node_to_eid = h.dual_hypergraph(start=start, end=end)
    eid_to_node = {v: k for k, v in node_to_eid.items()}
    eid1, eid2 = node_a, node_b
    if node_a is not None:
        eid1 = node_to_eid[node_a]
    if node_b is not None:
        eid2 = node_to_eid[node_b]

    res = s_distance(h1, s, eid1, eid2, weight=weight)
    if isinstance(res, int):
        return res
    elif isinstance(res, dict):
        return {eid_to_node[k]: v for k, v in res.items()}
    elif res is None:
        return None
    else:
        res = list(res)
        out = []
        for n, d in res:
            out.append((eid_to_node[n], {eid_to_node[k]: v for k, v in d.items()}))
        return out


def average_s_distance(
    h: ASH, s: int, start: int = None, end: int = None, weight: bool = False
) -> float:
    """

    :param h:
    :param s:
    :param start:
    :param end:
    :param weight:
    :return:
    """
    g = h.s_line_graph(s, start, end)

    if weight:
        return nx.average_shortest_path_length(g, weight="w")
    return nx.average_shortest_path_length(g)


def average_node_s_distance(
    h: ASH, s: int, start: int = None, end: int = None, weight: bool = False
) -> float:
    """

    :param h:
    :param s:
    :param start:
    :param end:
    :param weight:
    :return:
    """
    h1, _ = h.dual_hypergraph(start=start, end=end)
    return average_s_distance(h1, s, weight=weight)


def has_s_walk(
    h: ASH,
    s: int,
    hyperedge_a: str = None,
    hyperedge_b: str = None,
    start: int = None,
    end: int = None,
) -> bool:
    """

    :param h:
    :param s:
    :param hyperedge_a:
    :param hyperedge_b:
    :param start:
    :param end:
    :return:
    """

    g = h.s_line_graph(s, start, end)

    if hyperedge_a is not None and hyperedge_b is not None:
        if not g.has_node(hyperedge_a) or not g.has_node(hyperedge_b):
            return False
    elif hyperedge_a is not None and not g.has_node(hyperedge_a):
        return False

    return nx.has_path(g, hyperedge_a, hyperedge_b)


def has_node_s_walk(
    h: ASH,
    s: int,
    node_a: int = None,
    node_b: int = None,
    start: int = None,
    end: int = None,
) -> bool:
    """

    :param h:
    :param s:
    :param node_a:
    :param node_b:
    :param start:
    :param end:
    :return:
    """

    h1, node_to_eid = h.dual_hypergraph(start=start, end=end)

    eid1, eid2 = node_a, node_b
    if node_a is not None:
        eid1 = node_to_eid[node_a]
    if node_b is not None:
        eid2 = node_to_eid[node_b]

    return has_s_walk(h1, s, eid1, eid2)


def s_diameter(
    h: ASH, s: int, start: int = None, end: int = None, weight: bool = False
) -> int:
    """

    :param h:
    :param s:
    :param start:
    :param end:
    :param weight:
    :return:
    """

    distances = s_distance(h, s, start=start, end=end, weight=weight)
    diameter = 0
    for _, data in dict(distances).items():
        for _, dist in data.items():
            if diameter < dist:
                diameter = dist

    return diameter


def node_s_diameter(
    h: ASH, s: int, start: int = None, end: int = None, weight: bool = False
) -> int:
    """

    :param h:
    :param s:
    :param start:
    :param end:
    :param weight:
    :return:
    """
    h1, _ = h.dual_hypergraph(start=start, end=end)
    return s_diameter(h1, s, weight=weight)


def s_components(h: ASH, s: int, start: int = None, end: int = None) -> list:
    """

    :param h:
    :param s:
    :param start:
    :param end:
    :return:
    """
    g = h.s_line_graph(s, start, end)
    for comp in nx.connected_components(g):
        yield comp


def node_s_components(h: ASH, s: int, start: int = None, end: int = None) -> list:
    """

    :param h:
    :param s:
    :param start:
    :param end:
    :return:
    """

    h1, node_to_eid = h.dual_hypergraph(start=start, end=end)
    eid_to_node = {v: k for k, v in node_to_eid.items()}

    for comp in s_components(h1, s):
        yield {eid_to_node[i] for i in comp}


def is_s_path(h: ASH, walk: list) -> bool:
    """

    :param h:
    :param walk:
    :return:
    """
    p = Counter(walk)
    if max(p.values()) > 1:
        return False

    res = []
    for u, v in combinations(walk, 2):
        intersect = set(h.get_hyperedge_nodes(u)) & set(h.get_hyperedge_nodes(v))
        res.extend(list(intersect))

    p = Counter(res)
    if max(p.values()) > 1:
        return False
    else:
        return True
