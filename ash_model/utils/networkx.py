import copy
from typing import Optional, Sequence, Union

import networkx as nx
from ash_model import ASH, NProfile


def _validate_graph(graph: nx.Graph) -> None:
    """
    Ensure the input is an undirected NetworkX Graph.
    :raises TypeError: if not a networkx.Graph instance.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("Transformation only applicable to undirected NetworkX Graphs")


def _add_nodes(
    ash: ASH,
    graph: nx.Graph,
    start: int,
    end: Optional[int],
    keep_attrs: bool,
    bipartite: bool = False,
) -> None:
    """
    Add nodes from a NetworkX Graph to an ASH instance.

    :param ash: Target ASH object.
    :param graph: Source NetworkX Graph.
    :param start: Start time for node presence.
    :param end: Optional end time for node presence.
    :param keep_attrs: If True, preserves original node attributes.
    """
    
    if not bipartite:
        node_data = graph.nodes(data=True) 
    else:
        # For bipartite graphs, we only want the nodes from one partition
        nodes, _ = nx.algorithms.bipartite.sets(graph)
        node_data = ((node, graph.nodes[node]) for node in nodes)
    
    for node, data in node_data:
        if keep_attrs:
            attrs = copy.copy(data)
            profile = NProfile(node, **attrs)
            ash.add_node(node, start=start, end=end, attr_dict=profile)
        else:
            ash.add_node(node, start=start, end=end)


def _add_hyperedges(
    ash: ASH,
    graph: nx.Graph,
    start: int,
    end: Optional[int],
) -> None:
    """
    Add each edge of a NetworkX Graph as a 2-node hyperedge to an ASH instance.

    :param ash: Target ASH object.
    :param graph: Source NetworkX Graph.
    :param start: Start time for hyperedges.
    :param end: Optional end time for hyperedges.
    """
    for u, v in graph.edges():
        ash.add_hyperedge([u, v], start=start, end=end)


def from_networkx_graph(
    graph: nx.Graph,
    start: int,
    end: Optional[int] = None,
    keep_attrs: bool = False,
) -> ASH:
    """
    Convert an undirected NetworkX Graph into an ASH with 2-node hyperedges.

    :param graph: Undirected NetworkX Graph.
    :param start: Start time of hyperedges and nodes.
    :param end: Optional end time of hyperedges and nodes.
    :param keep_attrs: If True, node attributes are preserved.
    :return: Populated ASH object.
    """
    _validate_graph(graph)
    ash = ASH()
    _add_nodes(ash, graph, start, end, keep_attrs)
    _add_hyperedges(ash, graph, start, end)
    return ash


def from_networkx_graph_list(
    graphs: Sequence[nx.Graph],
    keep_attrs: bool = False,
) -> ASH:
    """
    Convert a sequence of NetworkX Graphs into a time­-sliced ASH.

    Each graph corresponds to a snapshot: its index is the start time.

    :param graphs: Sequence of undirected NetworkX Graphs.
    :param keep_attrs: If True, node attributes are preserved.
    :return: Populated ASH with hyperedges from each snapshot.
    """
    ash = ASH()
    for t, graph in enumerate(graphs):
        _validate_graph(graph)
        _add_nodes(ash, graph, t, t, keep_attrs)
        _add_hyperedges(ash, graph, t, t)
    return ash


def from_networkx_maximal_cliques(
    graph: nx.Graph,
    start: int,
    end: Optional[int] = None,
) -> ASH:
    """
    Convert maximal cliques of a NetworkX Graph into an ASH.

    :param graph: Undirected NetworkX Graph.
    :param start: Start time of clique hyperedges.
    :param end: Optional end time of clique hyperedges.
    :return: ASH object with each maximal clique as a hyperedge.
    """
    _validate_graph(graph)
    ash = ASH()
    # add only cliques of size >1
    for clique in nx.find_cliques(graph):
        if len(clique) > 1:
            ash.add_hyperedge(clique, start=start, end=end)
    # add all nodes with profiles
    for node, data in graph.nodes(data=True):
        ash.add_node(node, start=start, end=end, attr_dict=NProfile(node, **copy.copy(data)))
    return ash


def from_networkx_maximal_cliques_list(
    graphs: Sequence[nx.Graph]
) -> ASH:
    """
    Convert maximal cliques across time from a list of NetworkX Graphs into an ASH.

    Each graph index is the start time of its clique hyperedges and node presence.

    :param graphs: Sequence of undirected NetworkX Graphs.
    :return: Time­sliced ASH with cliques as hyperedges.
    """
    ash = ASH()
    for t, graph in enumerate(graphs):
        _validate_graph(graph)
        for clique in nx.find_cliques(graph):
            if len(clique) > 1:
                ash.add_hyperedge(clique, start=t, end=t)
        for node, data in graph.nodes(data=True):
            ash.add_node(node, start=t, end=t, attr_dict=NProfile(node, **copy.copy(data)))
    return ash


def from_networkx_bipartite(
    graph: nx.Graph,
    start: int,
    end: Optional[int] = None,
    keep_attrs: bool = False,
) -> ASH:
    """
    Convert a bipartite NetworkX Graph into an ASH where one node set becomes hyperedges
    and the other set becomes nodes. Edges connect hyperedges to their nodes.

    :param graph: Undirected bipartite NetworkX Graph.
    :param start: Start time for presence.
    :param end: Optional end time for presence.
    :param keep_attrs: If True, preserves node attributes of the original graph.
    :raises ValueError: If the graph is not bipartite.
    :return: ASH with hyperedges corresponding to bipartite partitions.
    """
    _validate_graph(graph)
    if not nx.is_bipartite(graph):
        raise ValueError("Provided Graph is not bipartite")
    ash = ASH()
    # collect nodes by partition
    nodes, edges = nx.algorithms.bipartite.sets(graph)
    # for each hyperedge (a node in one partition), take its neighbors
    for he in edges:
        neighbors = list(graph.neighbors(he))
        if neighbors:  # only add non-empty hyperedges
            print(f"Adding hyperedge {he} with neighbors {neighbors} from graph {graph}")
            ash.add_hyperedge(neighbors, start=start, end=end)
    # add all nodes
    _add_nodes(ash, graph, start, end, keep_attrs, bipartite=True)
    return ash


def from_networkx_bipartite_list(
    graphs: Sequence[nx.Graph],
    keep_attrs: bool = False,
) -> ASH:
    """
    Time­sliced conversion of bipartite graphs into an ASH.

    :param graphs: Sequence of undirected bipartite NetworkX Graphs.
    :param keep_attrs: If True, preserves node attributes.
    :return: ASH with hyperedges per snapshot.
    """
    ash = ASH()
    for t, graph in enumerate(graphs):
        _validate_graph(graph)
        if not nx.is_bipartite(graph):
            raise ValueError(f"Graph at index {t} is not bipartite")
        edges, nodes = nx.algorithms.bipartite.sets(graph)
        # for each hyperedge (a node in one partition), take its neighbors
        for he in edges:
            neighbors = list(graph.neighbors(he))
            if neighbors:
                ash.add_hyperedge(neighbors, start=t, end=t)
                
        node_data = ((node, graph.nodes[node]) for node in nodes)
    
        for node, data in node_data:
            if keep_attrs:
                attrs = copy.copy(data)
                profile = NProfile(node, **attrs)
                ash.add_node(node, start=t, end=t, attr_dict=profile)
            else:
                ash.add_node(node, start=t, end=t)
    return ash
