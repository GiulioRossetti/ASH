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


def k_intersections(h: ASH, k: int, tid: int) -> int:
    """

    :param h:
    :param k:
    :param tid:
    :return:
    """

    k_intersections = 0
    hedge_nodesets = []

    for hyperedge_id in h.hyperedge_id_iterator(start=tid):
        nodes = h.get_hyperedge_nodes(hyperedge_id)
        hedge_nodesets.append(set(nodes))

    for he1 in hedge_nodesets:
        for he2 in hedge_nodesets:
            if len(he1.intersection(he2)) >= k and he1 != he2:
                k_intersections += 1

    return k_intersections // 2


def inclusiveness(h: ASH) -> float:
    """
    Inclusiveness of an hypergraph is the ratio between the number of non-external 
    hyperdeges and the hypergraph size.

    :param h: an ASH object
    :return: inclusiveness value
    """
    he_nodesets = [set(h.get_hyperedge_nodes(he)) for he in h.get_hyperedge_id_set()]

    non_facets = set()
    for nset in he_nodesets:
        for nset2 in he_nodesets:
            if nset.issubset(nset2) and nset != nset2:
                non_facets.add(he_nodesets.index(nset))

    
    # 1 - (toplexes/hyperedges)
    return len(non_facets) / len(he_nodesets)
