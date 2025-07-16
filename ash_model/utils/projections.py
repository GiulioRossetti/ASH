import copy
from itertools import combinations
from typing import Dict
from collections import defaultdict

import networkx as nx

from ash_model import ASH, NProfile


def clique_projection(
    h: ASH, start: int = None, end: int = None, keep_attrs: bool = False
) -> nx.Graph:
    """
    Returns a NetworkX Graph object that is the clique projection of the
    given ASH object.
    The clique projection of a hypergraph is a graph constructed by placing
    an edge between every pair of nodes that belong to the same hyperedge.

    :param h: The ASH object to be projected.
    :param start: The start time of the projection (optional).
    :param end: The end time of the projection (optional).
    :param keep_attrs: If True, the attributes of the nodes in the ASH will
                        be preserved in the resulting NetworkX graph.
    :return: A NetworkX Graph object representing the clique projection of the ASH.
    """

    res = nx.Graph()
    for edge in h.hyperedges(start=start, end=end, as_ids=False):
        if len(edge) > 1:
            for u, v in combinations(edge, 2):
                if res.has_edge(u, v):
                    res[u][v]["weight"] += 1
                else:
                    res.add_edge(u, v, weight=1)
        if keep_attrs:
            for node in res.nodes():
                if node in h.nodes():
                    # Copy attributes from ASH node to NetworkX node
                    attrs = h.get_node_attributes(node)

                    res.nodes[node].update(attrs)

    return res


def bipartite_projection(
    h: ASH, start: int = None, end: int = None, keep_attrs: bool = False
) -> nx.Graph:
    """
    Returns a NetworkX Graph object that is the bipartite projection of the
    given ASH object.
    The bipartite projection of a hypergraph is a graph constructed by placing
    an edge between every node and every hyperedge that contains it.

    :param h: The ASH object to be projected.
    :param start: The start time of the projection (optional).
    :param end: The end time of the projection (optional).
    :param keep_attrs: If True, the attributes of the nodes in the ASH will
                        be preserved in the resulting NetworkX graph.
    :return: A NetworkX Graph object representing the bipartite projection of the ASH.
    """

    hes = h.hyperedges(start, end)

    g = nx.Graph()
    for he in hes:
        g.add_node(he, bipartite=1)
        nodes = h.get_hyperedge_nodes(he)
        for node in nodes:
            if not g.has_node(node):
                g.add_node(node, bipartite=0)
            g.add_edge(node, he)
    if keep_attrs:
        for node in g.nodes():
            if node in h.nodes():
                # Copy attributes from ASH node to NetworkX node
                attrs = h.get_node_attributes(node)
                g.nodes[node].update(attrs)
    return g


def line_graph_projection(
    h: ASH, s: int = 1, start: int = None, end: int = None, keep_attrs: bool = False
) -> nx.Graph:
    """
    Returns a NetworkX Graph object that is the line graph projection of the
    given ASH object.
    The line graph projection of a hypergraph is a graph constructed by placing
    an edge between every pair of hyperedges that share at least one node.

    :param h: The ASH object to be projected.
    :param s: The size of the edge overlap.
    :param end: The end time of the projection (optional).
    :param keep_attrs: If True, the attributes of the nodes in the ASH will
                        be preserved in the resulting NetworkX graph.
    :return: A NetworkX Graph object representing the line graph projection of the ASH.
    """

    g = nx.Graph()

    # Get hyperedges in the specified time window that have at least s nodes
    hyperedges = [
        he for he in h.hyperedges(start, end) if len(h.get_hyperedge_nodes(he)) >= s
    ]

    if keep_attrs:
        for he in hyperedges:
            g.add_node(he, attr_dict=h.hyperedges[he].attr_dict.to_dict())
    else:
        for he in hyperedges:
            g.add_node(he)

    # Add edges between hyperedges that intersect in at least s nodes
    for i, he1 in enumerate(hyperedges):
        for he2 in hyperedges[i + 1 :]:
            nodes1 = set(h.get_hyperedge_nodes(he1))
            nodes2 = set(h.get_hyperedge_nodes(he2))
            intersection_size = len(nodes1 & nodes2)

            if intersection_size >= s:
                g.add_edge(he1, he2, w=intersection_size)

    return g


def dual_hypergraph_projection(h: ASH, start: int = None, end: int = None) -> tuple:
    """
    The dual_hypergraph function takes a hypergraph and returns the dual of that hypergraph.
    The dual of a hypergraph is a hypergraph where each hyperedge becomes a node, and each
    node is connected to every other node in its corresponding hyperedge. The function also
    returns a node-to-edge mapping dictionary that maps each node in the original hypergraph
    to the new hyperedge ids in the dual hypergraph.

    :param h: The ASH object to be transformed into a dual hypergraph.
    :param start: Specify the start of a time window
    :param end: SpSpecify the end of a time window

    :return: the dual ASH and a node-to-edge mapping dictionary
    """

    b = ASH()
    node_to_edges = defaultdict(list)
    for he in h.hyperedges(start, end):
        nodes = h.get_hyperedge_nodes(he)
        for node in nodes:
            node_to_edges[node].append(he)

    old_nodes_to_new_edges = {}
    for node, edges in node_to_edges.items():
        b.add_hyperedge(edges, start=0, end=None)
        eid = b.get_hyperedge_id(edges)
        old_nodes_to_new_edges[node] = eid

    return b, old_nodes_to_new_edges


def clique_projection_by_time(h: ASH, keep_attrs: bool = False) -> Dict[int, nx.Graph]:
    """
    Returns a dictionary of NetworkX Graph objects that are the clique projections
    of the given ASH object for each time step.

    :param h: The ASH object to be projected.
    :param keep_attrs: If True, the attributes of the nodes in the ASH will
                        be preserved in the resulting NetworkX graphs.
    :return: A dictionary where keys are time steps and values are NetworkX Graph objects
             representing the clique projection of the ASH at that time step.
    """

    clique_projections = {}
    for t in h.temporal_snapshots_ids():
        clique_projections[t] = clique_projection(
            h, start=t, end=t, keep_attrs=keep_attrs
        )

    return clique_projections


def bipartite_projection_by_time(
    h: ASH, keep_attrs: bool = False
) -> Dict[int, nx.Graph]:
    """
    Returns a dictionary of NetworkX Graph objects that are the bipartite projections
    of the given ASH object for each time step.

    :param h: The ASH object to be projected.
    :param keep_attrs: If True, the attributes of the nodes in the ASH will
                        be preserved in the resulting NetworkX graphs.
    :return: A dictionary where keys are time steps and values are NetworkX Graph objects
             representing the bipartite projection of the ASH at that time step.
    """

    bipartite_projections = {}
    for t in h.temporal_snapshots_ids():
        bipartite_projections[t] = bipartite_projection(
            h, start=t, end=t, keep_attrs=keep_attrs
        )

    return bipartite_projections


def line_graph_projection_by_time(
    h: ASH, s: int = 1, keep_attrs: bool = False
) -> Dict[int, nx.Graph]:
    """
    Returns a dictionary of NetworkX Graph objects that are the line graph projections
    of the given ASH object for each time step.

    :param h: The ASH object to be projected.
    :param s: The size of the edge overlap.
    :param keep_attrs: If True, the attributes of the nodes in the ASH will
                        be preserved in the resulting NetworkX graphs.
    :return: A dictionary where keys are time steps and values are NetworkX Graph objects
             representing the line graph projection of the ASH at that time step.
    """

    line_graph_projections = {}
    for t in h.temporal_snapshots_ids():
        line_graph_projections[t] = line_graph_projection(
            h, s=s, start=t, end=t, keep_attrs=keep_attrs
        )

    return line_graph_projections


def dual_hypergraph_projection_by_time(h: ASH) -> Dict[int, tuple]:
    """
    Returns a dictionary of dual hypergraphs for each time step in the ASH object.
    Each entry in the dictionary contains the dual ASH and a node-to-edge mapping dictionary.

    :param h: The ASH object to be transformed into dual hypergraphs.
    :return: A dictionary where keys are time steps and values are tuples containing
             the dual ASH and a node-to-edge mapping dictionary.
    """

    dual_hypergraphs = {}
    for t in h.temporal_snapshots_ids():
        dual_hypergraphs[t] = dual_hypergraph_projection(h, start=t, end=t)

    return dual_hypergraphs
