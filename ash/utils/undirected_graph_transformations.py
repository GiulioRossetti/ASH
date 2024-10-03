import copy
from itertools import combinations
from typing import Dict

import networkx as nx

from ash import ASH, NProfile


def to_graph_decomposition(h: ASH, tid: int = None) -> Dict[int, ASH]:

    if tid is not None:
        tids = [tid]
    else:
        tids = h.temporal_snapshots_ids()

    res = {}

    for tid in tids:
        G = ASH()

        for node in h.nodes(tid=tid):
            profile = h.get_node_profile(node, tid=tid)
            G.add_node(node, start=tid, attr_dict=profile)

        edges = []
        for hyperedge_nodes in h.hyperedges(tid=tid, as_ids=False):
            edges.extend(list(combinations(hyperedge_nodes, 2)))

        G.add_hyperedges(edges, start=tid)
        res[tid] = G

    return res


def to_networkx_graph(h: ASH, tid: int = None) -> Dict[int, nx.Graph]:
    """
    Returns a dictionary of NetworkX Graph objects that are the graph decomposition of the given ASH object.

    """

    if tid is not None:
        tids = [tid]
    else:
        tids = h.temporal_snapshots_ids()

    res = {}

    for tid in tids:
        G = to_graph_decomposition(h, tid)[tid]

        nx_graph = nx.Graph()

        for node in G.nodes():
            nx_graph.add_node(
                node, **G.get_node_profile(node, tid=tid).get_attributes()
            )

        for hyperedge_id in G.hyperedges():
            edge_nodes = G.get_hyperedge_nodes(hyperedge_id)
            edge_attributes = G.get_hyperedge_attributes(hyperedge_id)
            nx_graph.add_edge(edge_nodes[0], edge_nodes[1], **edge_attributes)

        res[tid] = nx_graph

    return res


def from_networkx_graph(nx_graph: nx.Graph, start: int, end: int = None) -> ASH:
    """Returns an ASH object that is the graph decomposition of the given
    NetworkX Graph object.
    """

    if not isinstance(nx_graph, nx.Graph):
        raise TypeError("Transformation only applicable to undirected NetworkX graphs")

    G = ASH(hedge_removal=True)

    for node in nx_graph.nodes():
        G.add_node(
            node,
            start=start,
            end=end,
            attr_dict=NProfile(node, **copy.copy(nx_graph.nodes[node])),
        )

    for edge in nx_graph.edges():
        G.add_hyperedge([edge[0], edge[1]], start=start, end=end)

    return G
