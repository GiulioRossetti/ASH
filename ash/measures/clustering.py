from math import comb

from ash.paths import *


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
            triangle += ego.number_of_hyperedges()

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
    hes = set()
    for t in range(start, end + 1):
        hes.update(set(h.hyperedges(t)))
    for n in hes:
        count += 1
        LCCs.append(s_local_clustering_coefficient(h, s, n, start, end))

    if count > 0:
        return sum(LCCs) / count

    return 0


def s_intersections(h: ASH, s: int, tid: int) -> int:
    """

    :param h:
    :param s:
    :param tid:
    :return:
    """

    g = h.s_line_graph(s, tid)
    return g.number_of_edges()


def inclusiveness(h: ASH, tid: int = None) -> float:
    """
    The inclusiveness function is a measure of the number of non-external hyperedges in an ASH.
    It is defined as the ratio between the number of non-external hyperedges and
    the hypergraph size. The higher this value, the more inclusive (or less exclusive)
    the ASH is.

    :param h: an ASH object
    :param tid: temporal snapshot id
    :return: inclusiveness value
    """

    he_nodesets = [set(he) for he in h.hyperedges(tid=tid, as_ids=False)]

    non_facets = set()
    for nset in he_nodesets:
        for nset2 in he_nodesets:
            if nset.issubset(nset2) and nset != nset2:
                non_facets.add(he_nodesets.index(nset))

    # 1 - (toplexes/hyperedges)
    return len(non_facets) / len(he_nodesets)
