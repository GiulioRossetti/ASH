from ash import ASH, NProfile
import copy
import networkx as nx
import halp.utilities.undirected_graph_transformations as um


def to_graph_decomposition(H: ASH, tid: int = None) -> dict:
    """Returns an ASH object that has the same nodes (and
    corresponding attributes) as the given H, except that for all
    hyperedges in the given H, each node in the hyperedge is pairwise
    connected to every other node also in that hyperedge in the new H.
    Said another way, each of the original hyperedges are decomposed in the
    new H into cliques (aka the "2-section" or "clique graph").
    :param H: the H to decompose into a graph.
    :param tid:
    :returns: ASH -- the decomposed H.
    :raises: TypeError -- Transformation only applicable to
            undirected Hs
    """
    if not isinstance(H, ASH):
        raise TypeError(
            "Transformation only applicable to \
                        undirected Hs"
        )

    if tid is not None:
        tids = [tid]
    else:
        tids = H.snapshots.keys()

    res = {}

    for tid in tids:
        G = ASH()

        for node in H.node_iterator(tid=tid):
            profile = H.get_node_profile(node, tid=tid)
            G.add_node(node, start=tid, attr_dict=profile)

        edges = [
            (node_a, node_b)
            for hyperedge_id in H.hyperedge_id_iterator(start=tid)
            for node_a in H.get_hyperedge_nodes(hyperedge_id)
            for node_b in H.get_hyperedge_nodes(hyperedge_id)
            if node_a != node_b
        ]

        G.add_hyperedges(edges, start=tid)
        res[tid] = G

    return res


def to_networkx_graph(H: ASH, tid: int = None) -> dict:
    """Returns a NetworkX Graph object that is the graph decomposition of
    the given H.
    See "to_graph_decomposition()" for more details.
    :param H: the H to decompose into a graph.
    :param tid:
    :returns: nx.Graph -- NetworkX Graph object representing the
            decomposed H.
    :raises: TypeError -- Transformation only applicable to
            undirected Hs
    """

    if not isinstance(H, ASH):
        raise TypeError(
            "Transformation only applicable to \
                        undirected Hs"
        )

    if tid is not None:
        tids = [tid]
    else:
        tids = H.snapshots.keys()

    res = {}

    for tid in tids:
        G = to_graph_decomposition(H, tid)[tid]

        nx_graph = nx.Graph()

        for node in G.node_iterator():
            nx_graph.add_node(
                node, **G.get_node_profile(node, tid=tid).get_attributes()
            )

        for hyperedge_id in G.hyperedge_id_iterator():
            edge_nodes = G.get_hyperedge_nodes(hyperedge_id)
            edge_attributes = G.get_hyperedge_attributes(hyperedge_id)
            nx_graph.add_edge(edge_nodes[0], edge_nodes[1], **edge_attributes)

        res[tid] = nx_graph

    return res


def from_networkx_graph(nx_graph: nx.Graph, start: int, end: int = None) -> ASH:
    """Returns an UndirectedHypergraph object that is the graph equivalent of
    the given NetworkX Graph object.
    :param nx_graph: the NetworkX undirected graph object to transform.
    :param start:
    :param end:
    :returns: UndirectedHypergraph -- H object equivalent to the
            NetworkX undirected graph.
    :raises: TypeError -- Transformation only applicable to undirected
            NetworkX graphs
    """

    if not isinstance(nx_graph, nx.Graph):
        raise TypeError(
            "Transformation only applicable to undirected \
                        NetworkX graphs"
        )

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
