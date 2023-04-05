import networkx as nx
from collections import defaultdict, Counter
from itertools import combinations

from ash_model import ASH


def all_simple_paths(
    h: ASH,
    s: int,
    hyperedge_a: str = None,
    hyperedge_b: str = None,
    start: int = None,
    end: int = None,
    cutoff: int = None,
) -> object:
    """

    :param h:
    :param s:
    :param hyperedge_a:
    :param hyperedge_b:
    :param start:
    :param end:
    :param cutoff:
    :return:
    """

    g = h.s_line_graph(s, start, end)

    if hyperedge_a is not None and hyperedge_b is not None:
        if not g.has_node(hyperedge_a) or not g.has_node(hyperedge_b):
            return []
    elif hyperedge_a is not None and not g.has_node(hyperedge_a):
        return []

    for p in nx.all_simple_paths(g, hyperedge_a, hyperedge_b, cutoff):
        if is_s_path(h, p):
            yield p


def shortest_s_path(
    h: ASH,
    s: int,
    hyperedge_a: str = None,
    hyperedge_b: str = None,
    start: int = None,
    end: int = None,
) -> object:
    """

    :param h:
    :param s:
    :param hyperedge_a:
    :param hyperedge_b:
    :param start:
    :param end:
    :return:
    """

    paths = list(all_simple_paths(h, s, hyperedge_a, hyperedge_b, start, end))
    min_len = min([len(p) for p in paths])
    res = [p for p in paths if len(p) == min_len]
    return res


def all_shortest_s_path(
    h: ASH, s: int, hyperedge_a: str = None, start: int = None, end: int = None
) -> dict:
    """

    :param h:
    :param s:
    :param hyperedge_a:
    :param start:
    :param end:
    :return:
    """

    res = defaultdict(list)
    for he in h.hyperedge_id_iterator(start, end):
        if hyperedge_a is not None:
            if he != hyperedge_a:
                paths = list(all_simple_paths(h, s, hyperedge_a, he, start, end))
                min_len = min([len(p) for p in paths])
                res[(hyperedge_a, he)] = [p for p in paths if len(p) == min_len]
        else:
            for he1 in h.hyperedge_id_iterator(start, end):
                if he != he1:
                    paths = list(all_simple_paths(h, s, he, he1, start, end))
                    min_len = min([len(p) for p in paths])
                    res[he, he1] = [p for p in paths if len(p) == min_len]
    return res


def all_shortest_s_path_length(
    h: ASH, s: int, hyperedge_a: str = None, start: int = None, end: int = None
) -> dict:
    """

    :param h:
    :param s:
    :param hyperedge_a:
    :param start:
    :param end:
    :return:
    """

    res = defaultdict(lambda: defaultdict(int))
    for he in h.hyperedge_id_iterator(start, end):
        if hyperedge_a is not None:
            if he != hyperedge_a:
                paths = list(all_simple_paths(h, s, hyperedge_a, he, start, end))
                res[hyperedge_a][he] = min([len(p) for p in paths]) - 1
                res[he][hyperedge_a] = min([len(p) for p in paths]) - 1
            else:
                res[hyperedge_a][he] = 0
                res[he][hyperedge_a] = 0
        else:
            for he1 in h.hyperedge_id_iterator(start, end):
                if he != he1:
                    paths = list(all_simple_paths(h, s, he, he1, start, end))
                    res[he][he1] = min([len(p) for p in paths]) - 1
                    res[he1][he] = min([len(p) for p in paths]) - 1
                else:
                    res[he1][he] = 0
                    res[he][he1] = 0
    return res


def all_shortest_s_walk(
    h: ASH, s: int, hyperedge_a: str = None, start: int = None, end: int = None
) -> dict:
    """

    :param h:
    :param s:
    :param hyperedge_a:
    :param start:
    :param end:
    :return:
    """
    return shortest_s_walk(h, s, hyperedge_a, start, end)


def all_shortest_s_walk_length(
    h: ASH, s: int, hyperedge_a: str = None, start: int = None, end: int = None
) -> dict:
    """

    :param h:
    :param s:
    :param hyperedge_a:
    :param start:
    :param end:
    :return:
    """
    res = {}
    walks = all_shortest_s_walk(h, s, hyperedge_a, start, end)
    for k, v in walks.items():
        if hyperedge_a not in res:
            res[hyperedge_a] = {k: len(v)}
        else:
            res[hyperedge_a][k] = len(v)
        res[k] = {hyperedge_a: len(v)}
    return res


def shortest_s_walk(
    h: ASH,
    s: int,
    fr: str = None,
    to: str = None,
    start: int = None,
    end: int = None,
    weight: bool = False,
    edge: bool = True,
) -> object:
    """
    All returned paths include both the source and target in the path.

    If the source and target are both specified, return a single list of nodes in
    a shortest path from the source to the target.
    If only the source is specified, return a dictionary keyed by targets with a
    list of nodes in a shortest path from the source to one of the targets.
    If only the target is specified, return a dictionary keyed by sources with a
    list of nodes in a shortest path from one of the sources to the target.
    If neither the source nor target are specified return a dictionary of dictionaries
    with path[source][target]=[list of nodes in path].


    :param h:
    :param s:
    :param fr:
    :param to:
    :param start:
    :param end:
    :param weight:
    :param edge:
    :return:
    """

    if edge:
        g = h.s_line_graph(s, start, end)

        if fr is not None and to is not None:
            if not g.has_node(fr) or not g.has_node(to):
                return []
        elif fr is not None and not g.has_node(fr):
            return []

        if weight:
            return nx.shortest_path(g, source=fr, target=to, weight="w")
        return nx.shortest_path(g, source=fr, target=to)

    else:
        h1, node_to_eid = h.dual_hypergraph(start=start, end=end)
        eid_to_node = {v: k for k, v in node_to_eid.items()}

        eid1, eid2 = fr, to
        if fr is not None:
            eid1 = node_to_eid[fr]
        if to is not None:
            eid2 = node_to_eid[to]

        res = shortest_s_walk(h1, s, eid1, eid2, weight=weight)
        out = {}
        if isinstance(res, dict):
            for k, v in res.items():
                out[eid_to_node[k]] = [eid_to_node[i] for i in v]
            return out
        else:
            return [eid_to_node[i] for i in res]


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


def s_distance(
    h: ASH,
    s: int,
    fr: str = None,
    to: str = None,
    start: int = None,
    end: int = None,
    weight: bool = False,
    edge: bool = True,
) -> object:
    """

    :param h:
    :param s:
    :param fr:
    :param to:
    :param start:
    :param end:
    :param weight:
    :param edge:
    :return:
    """

    if edge:
        g = h.s_line_graph(s, start, end)

        if fr is not None and to is not None:
            if not g.has_node(fr) or not g.has_node(to):
                return None
        elif fr is not None and not g.has_node(fr):
            return None

        if weight:
            return nx.shortest_path_length(g, source=fr, target=to, weight="w")
        return nx.shortest_path_length(g, source=fr, target=to)

    else:
        h1, node_to_eid = h.dual_hypergraph(start=start, end=end)
        eid_to_node = {v: k for k, v in node_to_eid.items()}
        eid1, eid2 = fr, to
        if fr is not None:
            eid1 = node_to_eid[fr]
        if to is not None:
            eid2 = node_to_eid[to]

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
    h: ASH,
    s: int,
    start: int = None,
    end: int = None,
    weight: bool = False,
    edge: bool = False,
) -> float:
    """

    :param h:
    :param s:
    :param start:
    :param end:
    :param weight:
    :param edge:
    :return:
    """

    if not edge:
        h, _ = h.dual_hypergraph(start=start, end=end)

    g = h.s_line_graph(s, start, end)

    if weight:
        return nx.average_shortest_path_length(g, weight="w")
    return nx.average_shortest_path_length(g)


def has_s_walk(
    h: ASH,
    s: int,
    fr: str = None,
    to: str = None,
    start: int = None,
    end: int = None,
    edge: bool = True,
) -> bool:
    """

    :param h:
    :param s:
    :param fr:
    :param to:
    :param start:
    :param end:
    :param edge:
    :return:
    """

    if not edge:
        h, node_to_eid = h.dual_hypergraph(start=start, end=end)

        if fr is not None:
            fr = node_to_eid[fr]
        if to is not None:
            to = node_to_eid[to]

    g = h.s_line_graph(s, start, end)

    if fr is not None and to is not None:
        if not g.has_node(fr) or not g.has_node(to):
            return False
    elif fr is not None and not g.has_node(fr):
        return False

    return nx.has_path(g, fr, to)


def s_diameter(
    h: ASH,
    s: int,
    start: int = None,
    end: int = None,
    weight: bool = False,
    edge: bool = False,
) -> int:
    """

    :param h:
    :param s:
    :param start:
    :param end:
    :param weight:
    :param edge:
    :return:
    """
    if not edge:
        h, _ = h.dual_hypergraph(start=start, end=end)

    distances = s_distance(h, s, start=start, end=end, weight=weight)
    diameter = 0
    for _, data in dict(distances).items():
        for _, dist in data.items():
            if diameter < dist:
                diameter = dist

    return diameter


def s_components(
    h: ASH, s: int, start: int = None, end: int = None, edge: bool = True
) -> list:
    """

    :param h:
    :param s:
    :param start:
    :param end:
    :param edge:
    :return:
    """
    if edge:
        g = h.s_line_graph(s, start, end)
        for comp in nx.connected_components(g):
            yield comp

    else:
        h1, node_to_eid = h.dual_hypergraph(start=start, end=end)
        eid_to_node = {v: k for k, v in node_to_eid.items()}

        for comp in s_components(h1, s):
            yield {eid_to_node[i] for i in comp}


def is_s_path(h: ASH, walk: list) -> bool:
    """
    The is_s_path function takes in an ASH instance and a list of nodes, which represents
    a walk through the hypergraph. The function returns True if the walk is an s-path,
    and False otherwise.

    :param h: ASH instance
    :param walk: The sequence of nodes
    :return: True if the given walk is an s-path, and false otherwise

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
