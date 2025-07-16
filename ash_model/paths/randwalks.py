from ash_model import ASH
import numpy as np
from scipy import sparse
import csrgraph as cg
from ash_model.paths import temporal_s_dag
import networkx as nx

from typing import Tuple, Dict, Union


def random_walk_probabilities(
    h: ASH, start: int = None, end: int = None, edge: bool = False
) -> Tuple[sparse.csr_matrix, Dict[int, int]]:
    """
    Calculate the transition probabilities for random walks on the hypergraph.
    :param h: The ASH object.
    :param start: Lower temporal bound for the walks.
    :param end: Upper temporal bound for the walks.
    :param edge:
    ,mn
    """
    if edge:
        g = h.s_line_graph(start=start, end=end)
        T = nx.to_numpy_array(g, nodelist=sorted(g.nodes()), dtype=float)
        T = T / T.sum(axis=1, keepdims=True)

        n2idx = {node: i for i, node in enumerate(g.nodes())}
        return sparse.csr_matrix(T), n2idx

    nnodes = h.number_of_nodes(start=start, end=end)
    T = np.zeros((nnodes, nnodes), dtype=float)

    n2idx = {node: i for i, node in enumerate(h.nodes(start=start, end=end))}

    for edge in h.hyperedges(start=start, end=end, as_ids=False):
        edge = list(edge)  # Convert frozen set to list for indexing
        for i_ in range(len(edge)):
            i = n2idx[edge[i_]]
            for j_ in range(i_ + 1, len(edge)):
                j = n2idx[edge[j_]]
                T[i, j] += len(edge) - 1
                T[j, i] += len(edge) - 1
    T = np.matrix(T)
    T = T / T.sum(axis=1)

    return sparse.csr_matrix(T), n2idx


def random_walks(
    h: ASH,
    start_from: Union[int, str, list] = None,
    num_walks: int = 100,
    walk_length: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    edge: bool = False,
    start: int = None,
    end: int = None,
    threads: int = -1,
) -> np.ndarray:
    """
    Generate random walks on the hypergraph.

    :param h: The ASH object.
    :param start: Lower temporal bound for the walks.
    :param end: Upper temporal bound for the walks.
    :param num_walks: Number of random walks to generate.
    :param walk_length: Length of each walk.
    :param p: Return parameter (higher means less likely to return to the previous node).
    :param q: In-out parameter (higher means more likely to explore new nodes).
    :return: Sparse matrix representing the transition probabilities of the random walks.
    """

    T, n2idx = random_walk_probabilities(h, start, end, edge=edge)
    idx2n = {v: k for k, v in n2idx.items()}  # idx to node
    G = cg.csrgraph(T, threads=threads)
    if isinstance(start_from, (int, str)):
        start_from = n2idx[start_from]  # Convert node ID to index
    elif isinstance(start_from, list):
        start_from = [
            n2idx[node] for node in start_from
        ]  # Convert list of node IDs to indices
    else:
        start_from = None  # Start from all nodes
    all_walks = G.random_walks(
        walklen=walk_length,  # length of the walks
        epochs=num_walks,  # how many times to start a walk from each node
        start_nodes=start_from,  # starting nodes
        return_weight=1 / p,
        neighbor_weight=1 / q,
    )
    # Remap node indices back to original node IDs
    remapped_walks = np.array([[idx2n[idx] for idx in walk] for walk in all_walks])
    return remapped_walks


def time_respecting_random_walks(
    h: ASH,
    s: int,
    hyperedge_from: str = None,  # or list[str],
    hyperedge_to: str = None,
    start: int = None,
    end: int = None,
    num_walks: int = 100,
    walk_length: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    threads: int = -1,
) -> np.ndarray:
    """
    Generate time-respecting random walks on the hypergraph.
    :param h: The ASH object.
    :param s: the minimum hyperedge overlap size for subsequent steps.
    :param hyperedge_from: The hyperedge from which to start the walks.
    :param hyperedge_to: The hyperedge to which the walks should go.
    :param start: Lower temporal bound for the walks.
    :param end: Upper temporal bound for the walks.
    :param num_walks: Number of random walks to generate.
    :param walk_length: Length of each walk.
    :param p: Return parameter (higher means less likely to return to the previous node).
    :param q: In-out parameter (higher means more likely to explore new nodes).
    :param threads: Number of threads to use for parallel processing.
    :return: Sparse matrix representing the transition probabilities of the random walks.
    """

    if isinstance(hyperedge_from, str):
        hyperedge_from = [hyperedge_from]  # Convert to list if a single edge
    elif hyperedge_from is None:
        hyperedge_from = h.hyperedges(start=start, end=end, as_ids=True)

    G_nx = nx.DiGraph()
    dags = []
    for he in hyperedge_from:
        dag, _, _ = temporal_s_dag(h, s, he, hyperedge_to, start=start, end=end)
        dags.append(dag)
    G_nx = nx.compose_all(dags)

    # relabel nodes
    node_mapping = {old: new for new, old in enumerate(G_nx.nodes())}
    inv_mapping = {new: old for old, new in node_mapping.items()}
    G_nx = nx.relabel_nodes(G_nx, node_mapping)

    # Build sparse adjacency matrix
    adj = nx.to_numpy_array(G_nx, nodelist=sorted(G_nx.nodes()), dtype=float)
    adj = sparse.csr_matrix(adj)

    # Initialize csrgraph
    G_cg = cg.csrgraph(adj, threads=threads)

    # Prepare start_nodes: either None (→ all nodes) or an int64 numpy array
    raw_start = [node_mapping[he] for he in hyperedge_from if he in node_mapping]
    if not raw_start:
        start_nodes = None
    else:
        start_nodes = np.array(raw_start, dtype=np.int64)

    # Run the walks
    allwalks = G_cg.random_walks(
        walklen=walk_length,
        epochs=num_walks,
        start_nodes=start_nodes,
        return_weight=1.0 / p,
        neighbor_weight=1.0 / q,
    )

    # Map back to original node IDs
    remapped = np.array(
        [[inv_mapping[idx] for idx in walk] for walk in allwalks], dtype=object
    )

    return remapped
