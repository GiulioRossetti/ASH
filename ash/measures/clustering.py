from math import comb

from ash.paths import *


def s_local_clustering_coefficient(
    h: ASH, s: int, hyperedge_id: str, start: int = None, end: int = None
) -> float:
    """
    The s_local_clustering_coefficient function computes the local clustering coefficient of a hyperedge in a
    hypergraph. The start and end parameters are optional arguments that can be used to specify which interval should
    be considered when computing this metric. If no interval is specified then all time points will be used.

    :param h: ASH instance
    :param s: Specify the number of steps to take in the line graph
    :param hyperedge_id: Specify the hyperedge for which to compute the local clustering coefficient
    :param start: Specify the start of a time window
    :param end: Specify the end of a time window
    :return: The local clustering coefficient of the hyperedge with id `hyperedge_id`
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
    The average_s_local_clustering_coefficient function computes the average of the s-local clustering coefficient
    for all hyperedges.

    :param h: ASH instance
    :param s: Specify the number of nodes that are to be considered for each hyperedge
    :param start: Specify the start of the time window
    :param end: Specify the end of the time window
    :return: The average s-local clustering coefficient


    """

    LCCs = []
    count = 0
    for n in h.hyperedge_id_iterator(start, end):
        count += 1
        LCCs.append(s_local_clustering_coefficient(h, s, n, start, end))

    if count > 0:
        return sum(LCCs) / count

    return 0


def s_intersections(h: ASH, s: int, tid: int = None) -> int:
    """
    The s_intersections function counts the number of intersections of at least size s
    between hyperedges. Parameter tid optionally specifies the temporal snapshot to analyze.

    :param h: ASH instance
    :param s: Specify the minimum size of the intersection
    :param tid: Specify the current time step
    :return: The number of s-intersections in the hypergraph
    """

    intersections = 0
    hedge_nodesets = []

    for hyperedge_id in h.get_hyperedge_id_set(tid=tid):
        nodes = h.get_hyperedge_nodes(hyperedge_id)
        hedge_nodesets.append(set(nodes))

    for he1 in hedge_nodesets:
        for he2 in hedge_nodesets:
            if len(he1.intersection(he2)) >= s and he1 != he2:
                intersections += 1

    return intersections // 2


def inclusiveness(h: ASH, tid: int = None) -> float:
    """
    The inclusiveness function is a measure of the number of non-external hyperedges in an ASH.
    It is defined as the ratio between the number of non-external hyperedges and
    the hypergraph size. The higher this value, the more inclusive (or less exclusive)
    the ASH is.

    :param h: ASH instance
    :param tid: Specify the temporal snapshot id
    :return: The ratio between the number of non-external hyperedges and the hypergraph size

    """

    he_nodesets = [
        set(h.get_hyperedge_nodes(he)) for he in h.get_hyperedge_id_set(tid=tid)
    ]

    non_facets = set()
    for nset in he_nodesets:
        for nset2 in he_nodesets:
            if nset.issubset(nset2) and nset != nset2:
                non_facets.add(he_nodesets.index(nset))

    # 1 - (toplexes/hyperedges)
    return len(non_facets) / len(he_nodesets)
