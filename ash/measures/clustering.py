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

    LCC = triangle / comb(2, ego.number_of_nodes())
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
