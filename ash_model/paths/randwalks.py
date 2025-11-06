from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import sparse
import networkx as nx
import csrgraph as cg

from ash_model import ASH
from ash_model.paths import temporal_s_dag, TemporalEdge


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
    h: ASH, s: int, start: Optional[int], end: Optional[int]
) -> Tuple[sparse.csr_matrix, Dict[Any, int]]:
    """
    Construct transition probability matrix between nodes in hyperedges.

    :param h: ASH hypergraph object
    :param s: Minimum s-incidence threshold (nodes must co-occur in at least s hyperedges)
    :param start: Lower temporal bound
    :param end: Upper temporal bound
    """
    nodes = list(h.nodes(start=start, end=end))
    n2idx, _ = _map_to_indices(nodes)
    n = len(nodes)

    # Track co-occurrence counts between node pairs
    cooccurrence_counts: Dict[Tuple[Any, Any], int] = {}

    for edge in h.hyperedges(start=start, end=end, as_ids=False):
        vertices = list(edge)
        for u in vertices:
            for v in vertices:
                if u != v:
                    pair = (u, v)
                    cooccurrence_counts[pair] = cooccurrence_counts.get(pair, 0) + 1

    # Build transition matrix only for pairs with >= s co-occurrences
    T = np.zeros((n, n), dtype=float)
    for (u, v), count in cooccurrence_counts.items():
        if count >= s:
            # Weight by number of co-occurrences
            T[n2idx[u], n2idx[v]] += count

    T = _normalize_rows(T)
    return sparse.csr_matrix(T), n2idx


def _build_edge_transition_matrix(
    h: ASH, s: int, start: Optional[int], end: Optional[int]
) -> Tuple[sparse.csr_matrix, Dict[Any, int]]:
    """
    Construct transition probability matrix on the line graph of hyperedges.

    :param h: ASH hypergraph object
    :param s: Minimum size of node intersection between hyperedges
    :param start: Lower temporal bound
    :param end: Upper temporal bound
    """
    G = h.s_line_graph(s=s, start=start, end=end)
    nodes = sorted(G.nodes())
    n2idx, _ = _map_to_indices(nodes)

    A = nx.to_numpy_array(G, nodelist=nodes, dtype=float)
    A = _normalize_rows(A)
    return sparse.csr_matrix(A), n2idx


def random_walk_probabilities(
    h: ASH,
    s: int = 1,
    start: Optional[int] = None,
    end: Optional[int] = None,
    edge: bool = False,
) -> Tuple[sparse.csr_matrix, Dict[Any, int]]:
    """
    Compute CSR transition matrix and index mapping for nodes or hyperedges.

    :param h: ASH hypergraph object
    :param s: Minimum s-incidence threshold (for nodes: co-occurrence count; for edges: intersection size)
    :param start: Lower temporal bound
    :param end: Upper temporal bound
    :param edge: If True, compute for hyperedge line graph
    """
    if edge:
        return _build_edge_transition_matrix(h, s, start, end)
    return _build_node_transition_matrix(h, s, start, end)


def random_walks(
    h: ASH,
    start_from: Union[int, str, List[Union[int, str]], None] = None,
    num_walks: int = 100,
    walk_length: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    s: int = 1,
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
    :param p: Return parameter (higher values make walk less likely to return to previous node)
    :param q: In-out parameter (higher values make walk more local, lower values encourage exploration)
    :param s: Minimum s-incidence threshold. For node walks: nodes must co-occur in at least s hyperedges to be connected. For edge walks: hyperedges must share at least s nodes to be connected.
    :param edge: If True, walk on hyperedge line graph
    :param start: Lower temporal bound
    :param end: Upper temporal bound
    :param threads: Parallel threads for random walk computation

    :returns: Array of walks (each walk is a list of original node/edge IDs)

    Examples
    --------
    .. code-block:: python

        # Node-based random walks with s=1 (any co-occurrence)
        walks = random_walks(h, num_walks=100, walk_length=10, s=1)

        # Node-based random walks with s=2 (nodes must co-occur in at least 2 hyperedges)
        walks = random_walks(h, num_walks=100, walk_length=10, s=2)

        # Hyperedge-based random walks with s=2 (hyperedges must share at least 2 nodes)
        walks = random_walks(h, num_walks=100, walk_length=10, s=2, edge=True)
    """
    T_csr, n2idx = random_walk_probabilities(h, s, start, end, edge=edge)
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
    start_from: Optional[Union[int, str, List[Union[int, str]]]] = None,
    stop_at: Optional[Union[int, str]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    num_walks: int = 100,
    walk_length: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    edge: bool = False,
    threads: int = -1,
) -> Union[np.ndarray, Dict[Tuple[str, str], List[List[TemporalEdge]]]]:
    """
    Generate biased, time-respecting random walks on the temporal hypergraph.

    This function builds a time-respecting transition matrix and uses it to guide
    random walks that respect temporal ordering. The approach uses the temporal DAG
    structure where all edges are forward-in-time transitions (t -> t' where t' > t).

    Semantics:
    - All transitions are forward-in-time, respecting strict temporal ordering
    - Each step moves to a strictly later timestamp
    - Walks terminate when no forward neighbors exist (reached a temporal sink)

    :param h: ASH hypergraph object
    :param s: Minimum s-incidence threshold
    :param start_from: Node or edge (or list) to start walks from
    :param stop_at: Node or edge to stop walks at (optional)
    :param start: Lower temporal bound
    :param end: Upper temporal bound
    :param num_walks: Number of walks per start node/edge
    :param walk_length: Length of each walk (number of transitions)
    :param p: Return parameter (higher values discourage returning to previous node)
    :param q: In-out parameter (higher values favor local exploration)
    :param edge: If True, walk on hyperedge line graph and return TemporalEdge dict
    :param threads: Parallel threads for random walk computation (currently unused in custom logic)

    :returns: If edge=False, ndarray of node ID sequences. If edge=True, dict mapping (start, end) to lists of TemporalEdge walks.

    Examples
    --------
    .. code-block:: python

        # Time-respecting node walks
        walks = time_respecting_random_walks(h, s=1, num_walks=100, walk_length=10)

        # Time-respecting hyperedge walks
        walks_dict = time_respecting_random_walks(h, s=2, num_walks=100, walk_length=10, edge=True)

        # Start from specific nodes
        walks = time_respecting_random_walks(h, s=1, start_from=[1, 2], num_walks=50)
    """
    # Build temporal DAG
    DAG, sources, _ = temporal_s_dag(
        h, s, start_from, stop_at, start=start, end=end, edge=edge
    )

    # Build neighbor maps: all edges are now forward-in-time (time-respecting)
    # There are NO same-timestamp transitions in the corrected implementation
    nodes = [n for n in DAG.nodes() if isinstance(n, str) and "_" in n]
    neighbors: Dict[str, List[Tuple[str, float]]] = {}

    for u, v, attrs in DAG.edges(data=True):
        if "_" not in str(u) or "_" not in str(v):
            continue
        # All edges are forward-in-time transitions (t -> t' where t' > t)
        neighbors.setdefault(str(u), []).append(
            (str(v), float(attrs.get("weight", 1.0)))
        )

    # Helper to choose a neighbor by weights
    def pick_weighted(neis: List[Tuple[str, float]]) -> Optional[str]:
        if not neis:
            return None
        vs, ws = zip(*neis)
        ws = np.array(ws, dtype=float)
        s_sum = ws.sum()
        if s_sum <= 0:
            ws = np.ones_like(ws) / len(ws)
        else:
            ws = ws / s_sum
        idx = np.random.choice(len(vs), p=ws)
        return vs[idx]

    # Determine start nodes
    if start_from is None:
        start_nodes = [n for n in sources if n in neighbors or n in nodes]
    else:
        if not isinstance(start_from, list):
            start_from = [start_from]
        bases = set(str(sf) for sf in start_from)
        start_nodes = [n for n in nodes if n.split("_")[0] in bases]

    if edge:
        from collections import defaultdict

        res: Dict[Tuple[str, str], List[List[TemporalEdge]]] = defaultdict(list)

        for s_node in start_nodes:
            for _ in range(num_walks):
                path: List[TemporalEdge] = []
                cur = s_node
                steps = 0

                while steps < walk_length:
                    neis = neighbors.get(cur, [])
                    if not neis:
                        # No outgoing edges - walk reached a temporal sink
                        break

                    nxt = pick_weighted(neis)
                    if nxt is None:
                        break

                    # Append step (time-respecting: cur_t -> nxt_t' where t' > t)
                    fr, ft = cur.split("_")
                    to, tt = nxt.split("_")
                    w = next((w for v, w in neis if v == nxt), 1.0)
                    path.append(TemporalEdge(fr, to, float(w), int(tt)))
                    steps += 1
                    cur = nxt

                    if stop_at and (to == str(stop_at) or nxt == str(stop_at)):
                        break

                if path:
                    key = (path[0].fr, path[-1].to)
                    res[key].append(path)

        return dict(res)
    else:
        walks: List[List[Union[int, str]]] = []

        for s_node in start_nodes:
            for _ in range(num_walks):
                seq: List[Union[int, str]] = []
                cur = s_node
                steps = 0

                while steps < walk_length:
                    neis = neighbors.get(cur, [])
                    if not neis:
                        # No outgoing edges - walk reached a temporal sink
                        break

                    nxt = pick_weighted(neis)
                    if nxt is None:
                        break

                    base = nxt.split("_")[0]
                    seq.append(base)
                    steps += 1
                    cur = nxt

                    if stop_at is not None and str(base) == str(stop_at):
                        break

                if seq:
                    walks.append(seq)

        return np.array(walks, dtype=object)
