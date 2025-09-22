from typing import Any, Dict, List, Optional, Tuple, Union
from collections import namedtuple

import numpy as np
from scipy import sparse
import networkx as nx
import csrgraph as cg

from ash_model import ASH
from ash_model.paths import temporal_s_dag

TemporalEdge = namedtuple("TemporalEdge", "fr to weight tid")
TemporalEdge.__new__.__defaults__ = (None,) * len(TemporalEdge._fields)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize each row of a numpy matrix so that rows sum to 1.
    """
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def _map_to_indices(items: List[Any]) -> Tuple[Dict[Any, int], Dict[int, Any]]:
    """
    Create forward and reverse mappings between items and integer indices.
    """
    fwd = {item: idx for idx, item in enumerate(items)}
    rev = {idx: item for item, idx in fwd.items()}
    return fwd, rev


def _build_node_transition_matrix(
    h: ASH, start: Optional[int], end: Optional[int]
) -> Tuple[sparse.csr_matrix, Dict[Any, int]]:
    """
    Construct transition probability matrix between nodes in hyperedges.
    """
    nodes = list(h.nodes(start=start, end=end))
    n2idx, _ = _map_to_indices(nodes)
    n = len(nodes)
    T = np.zeros((n, n), dtype=float)

    for edge in h.hyperedges(start=start, end=end, as_ids=False):
        vertices = list(edge)
        weight = len(vertices) - 1
        for u in vertices:
            for v in vertices:
                if u != v:
                    T[n2idx[u], n2idx[v]] += weight

    T = _normalize_rows(T)
    return sparse.csr_matrix(T), n2idx


def _build_edge_transition_matrix(
    h: ASH, start: Optional[int], end: Optional[int]
) -> Tuple[sparse.csr_matrix, Dict[Any, int]]:
    """
    Construct transition probability matrix on the line graph of hyperedges.
    """
    G = h.s_line_graph(start=start, end=end)
    nodes = sorted(G.nodes())
    n2idx, _ = _map_to_indices(nodes)

    A = nx.to_numpy_array(G, nodelist=nodes, dtype=float)
    A = _normalize_rows(A)
    return sparse.csr_matrix(A), n2idx


def random_walk_probabilities(
    h: ASH,
    start: Optional[int] = None,
    end: Optional[int] = None,
    edge: bool = False,
) -> Tuple[sparse.csr_matrix, Dict[Any, int]]:
    """
    Compute CSR transition matrix and index mapping for nodes or hyperedges.
    """
    if edge:
        return _build_edge_transition_matrix(h, start, end)
    return _build_node_transition_matrix(h, start, end)


def random_walks(
    h: ASH,
    start_from: Union[int, str, List[Union[int, str]], None] = None,
    num_walks: int = 100,
    walk_length: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    edge: bool = False,
    start: Optional[int] = None,
    end: Optional[int] = None,
    threads: int = -1,
) -> np.ndarray:
    """
    Generate biased random walks on ASH hypergraph (node or edge graph).

    :param h: ASH hypergraph object
    :param start_from: Node or list of nodes to start walks from
    :param num_walks: Number of walks per start node
    :param walk_length: Length of each walk
    :param p: Return parameter
    :param q: In-out parameter
    :param edge: If True, walk on hyperedge line graph
    :param start: Lower temporal bound
    :param end: Upper temporal bound
    :param threads: Parallel threads for random walk computation

    :returns: Array of walks (each walk is a list of original node/edge IDs)
    """
    T_csr, n2idx = random_walk_probabilities(h, start, end, edge=edge)
    idx2n = {idx: node for node, idx in n2idx.items()}
    G = cg.csrgraph(T_csr, threads=threads)

    if start_from is None:
        start_nodes = None
    else:
        if not isinstance(start_from, list):
            start_from = [start_from]
        start_nodes = [n2idx[item] for item in start_from]

    raw = G.random_walks(
        walklen=walk_length,
        epochs=num_walks,
        start_nodes=start_nodes,
        return_weight=1.0 / p,
        neighbor_weight=1.0 / q,
    )

    return np.array([[idx2n[idx] for idx in walk] for walk in raw])


def time_respecting_random_walks(
    h: ASH,
    s: int,
    hyperedge_from: Optional[Union[int, str, List[Union[int, str]]]] = None,
    hyperedge_to: Optional[Union[int, str]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    num_walks: int = 100,
    walk_length: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    threads: int = -1,
) -> Dict[Tuple[str, str], List[List[TemporalEdge]]]:
    """
    Perform biased, time-respecting random s-walks on a temporal hypergraph.

    :param h: ASH hypergraph object
    :param s: Minimum s-incidence threshold
    :param hyperedge_from: Hyperedge ID(s) to start walks from
    :param hyperedge_to: Hyperedge ID to stop walks at (optional)
    :param start: Lower temporal bound
    :param end: Upper temporal bound
    :param num_walks: Number of walks per start edge
    :param walk_length: Maximum steps per walk
    :param p: Return parameter
    :param q: In-out parameter
    :param threads: Parallel threads

    :returns: Mapping (start_edge, end_edge) -> list of walks (temporal edge sequences)
    """
    # Build temporal DAG
    DAG, sources, _ = temporal_s_dag(
        h, s, hyperedge_from, hyperedge_to, start=start, end=end
    )

    # Index mapping
    nodes = list(DAG.nodes())
    n2idx, idx2n = _map_to_indices(nodes)

    # CSR adjacency
    rows, cols, data = [], [], []
    for u, v, attrs in DAG.edges(data=True):
        rows.append(n2idx[u])
        cols.append(n2idx[v])
        data.append(attrs.get("weight", 1.0))
    T_csr = sparse.csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))

    G = cg.csrgraph(T_csr, threads=threads)

    # Determine start indices
    if hyperedge_from is None:
        start_indices = [n2idx[n] for n in sources]
    else:
        if not isinstance(hyperedge_from, list):
            hyperedge_from = [hyperedge_from]
        start_indices = [
            idx
            for n, idx in n2idx.items()
            if n.split("_")[0] in map(str, hyperedge_from)
        ]

    raw_walks = G.random_walks(
        walklen=walk_length,
        epochs=num_walks,
        start_nodes=start_indices,
        return_weight=1.0 / p,
        neighbor_weight=1.0 / q,
    )

    # Aggregate walks by (start_edge, end_edge)
    from collections import defaultdict

    res: Dict[Tuple[str, str], List[List[TemporalEdge]]] = defaultdict(list)
    for seq in raw_walks:
        path: List[TemporalEdge] = []
        prev: Optional[int] = None
        for idx in seq:
            if prev is None:
                prev = idx
                continue
            u_node = idx2n[prev]
            v_node = idx2n[idx]
            try:
                fr, ft = u_node.split("_")
                to, tt = v_node.split("_")
            except ValueError:
                """
                report = ""
                report += f"u_node: {u_node}\n"
                report += f"v_node: {v_node}\n"
                report += f"prev: {prev}\n"
                report += f"idx: {idx}\n"
                report += f"seq: {seq}\n"
                report += f"nodes: {DAG.nodes()}\n"
                raise ValueError(f"Unexpected node format in DAG: {report}")
                """
                continue

            weight = DAG[u_node][v_node].get("weight", 1.0)
            path.append(TemporalEdge(fr, to, weight, int(tt)))
            prev = idx
            if hyperedge_to and to == str(hyperedge_to):
                break
        if path:
            key = (path[0].fr, path[-1].to)
            res[key].append(path)

    return dict(res)
