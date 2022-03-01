from ash import ASH
from ash.paths import *
import networkx as nx
from math import comb


def s_local_clustering_coefficient(
    h: ASH, s: int, hyperedge_id: str, start: int = None, end: int = None
) -> float:
    """

    :param h:
    :param s:
    :param hyperedge_id:
    :param start:
    :param end:
    :return:
    """

    lg = h.s_line_graph(s, start, end)
    if not lg.has_node(hyperedge_id):
        return 1

    ego = nx.ego_graph(lg, hyperedge_id)
    ego.remove_node(hyperedge_id)

    if ego.number_of_nodes() == 0:
        return 0

    triangle = 0
    components = nx.connected_components(ego)
    for c in components:
        res = [hyperedge_id]
        res.extend(c)
        if len(res) > 2 and is_s_path(h, res):
            triangle += ego.number_of_edges()

    denom = comb(2, ego.number_of_nodes())
    if denom == 0:
        return 0
    LCC = triangle / denom
    return LCC


def average_s_local_clustering_coefficient(
    h: ASH, s: int, start: int = None, end: int = None
) -> float:
    """

    :param h:
    :param s:
    :param start:
    :param end:
    :return:
    """

    LCCs = []
    count = 0
    for n in h.hyperedge_id_iterator(start, end):
        count += 1
        LCCs.append(s_local_clustering_coefficient(h, s, n, start, end))

    if count > 0:
        return sum(LCCs) / count

    return 0


def s_intersections(h: ASH, s: int, tid: int = None) -> float:
    """

    :param h:
    :param s:
    :param tid:
    :return:
    """

    s_intersections = 0
    hedge_nodesets = []

    for hyperedge_id in h.hyperedge_id_iterator(start=tid):
        nodes = h.get_hyperedge_nodes(hyperedge_id)
        hedge_nodesets.append(set(nodes))

    for he1 in hedge_nodesets:
        for he2 in hedge_nodesets:
            if len(he1.intersection(he2)) >= s and he1 != he2:
                s_intersections += 1

    return s_intersections // 2


# def inclusion(h: ASH, tid: int = None) -> float:
#     """

#     :param h:
#     :param tid:
#     :return:
#     """

#     # create graph
#     edges = [
#         (node_a, node_b)
#         for hyperedge_id in h.hyperedge_id_iterator(start=tid)
#         for node_a in h.get_hyperedge_nodes(hyperedge_id)
#         for node_b in h.get_hyperedge_nodes(hyperedge_id)
#         if node_a != node_b
#     ]
#     g = nx.Graph()
#     g.add_edges_from(edges)

#     # extract cliques and compute metric
#     toplexes = list(nx.clique.find_cliques(g))
#     hyperedges = h.get_hyperedge_id_set()
#     res = 1 - (len(toplexes) / len(hyperedges))

#     return res

