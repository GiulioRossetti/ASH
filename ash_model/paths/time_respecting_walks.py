import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union

import networkx as nx
import numpy as np

from ash_model import ASH


# A temporal edge in an s-walk: from-hyperedge id, to-hyperedge id, weight, and timestamp.
@dataclass(frozen=True)
class TemporalEdge:
    fr: str
    to: str
    weight: float
    tid: int


def temporal_s_dag(
    h: ASH,
    s: int,
    start_from: Optional[Union[int, str, List[Union[int, str]]]] = None,
    stop_at: Optional[Union[int, str]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    edge: bool = False,
) -> Tuple[nx.DiGraph, List[str], List[str]]:
    """
    Build a time-respecting DAG over [start, end] for either hyperedges (edge=True) or nodes (edge=False).
    Nodes are labeled as "<id>_<tid>".
    Edges connect items from different timestamps following chronological order, ensuring time-respecting properties.
    Only forward-in-time edges are created: from timestamp t to timestamps t' where t' > t.

    :param h: The source hypergraph.
    :param s: Minimum s-incidence threshold.
    :param start_from: Node or hyperedge id(s) to start from. If None, starts from all items present at the first snapshot in range.
    :param stop_at: Node or hyperedge id to stop at (optional).
    :param start: First snapshot ID to include. Defaults to earliest.
    :param end: Last snapshot ID to include. Defaults to latest.
    :param edge: If True, operate on hyperedges. If False, operate on nodes.
    :returns: (DAG, sources, targets) with labels "<id>_<tid>".
    :raises ValueError: If the [start, end] interval is not a valid subset of the hypergraph's snapshot IDs.
    """
    ids = h.temporal_snapshots_ids()
    if len(ids) == 0:
        return nx.DiGraph(), [], []

    if end is None:
        end = ids[-1]
    if start is None:
        start = ids[0]

    if start < min(ids) or start > end or end > max(ids) or start > max(ids):
        raise ValueError(
            f"The specified interval {[start, end]} is not a proper subset of the network timestamps "
            f"{[min(ids), max(ids)]}."
        )

    start_idx = next(i for i, t in enumerate(ids) if t >= start)
    end_idx = max(i for i, t in enumerate(ids) if t <= end)
    ids = ids[start_idx : end_idx + 1]

    # Normalize seeds to list[str]
    if start_from is None:
        seeds: List[str] = []
    elif isinstance(start_from, (list, tuple, set)):
        seeds = [str(x) for x in start_from]
    else:
        seeds = [str(start_from)]

    DG = nx.DiGraph()
    sources, targets = {}, {}

    # First pass: build all edges and collect reachable items
    all_edges = []
    item_at_time = defaultdict(set)  # Track which items exist at which times

    for i, tid in enumerate(ids):
        # Track items present at this timestamp
        if edge:
            for he in h.hyperedges(start=tid, end=tid):
                item_at_time[tid].add(str(he))
        else:
            for n in h.nodes(start=tid, end=tid):
                item_at_time[tid].add(str(n))

        # Build connections to all future timestamps
        for future_idx in range(i + 1, len(ids)):
            future_tid = ids[future_idx]

            if edge:
                # For edge mode: find s-incident hyperedges at future timestamp
                for he in h.hyperedges(start=tid, end=tid):
                    he_id = str(he)
                    raw_neighbors = h.get_s_incident(
                        he_id, s=s, start=future_tid, end=future_tid
                    )
                    for n_id, w in raw_neighbors:
                        if n_id != he_id:
                            all_edges.append(
                                (f"{he_id}_{tid}", f"{n_id}_{future_tid}", w)
                            )
            else:
                # For node mode: find co-members at future timestamp
                node_neighbors_future: dict[str, dict[str, int]] = defaultdict(
                    lambda: defaultdict(int)
                )
                for he in h.hyperedges(start=future_tid, end=future_tid):
                    he_nodes = list(h.get_hyperedge_nodes(he))
                    for ii in range(len(he_nodes)):
                        u = str(he_nodes[ii])
                        for j in range(len(he_nodes)):
                            if ii == j:
                                continue
                            v = str(he_nodes[j])
                            node_neighbors_future[u][v] += 1

                # Process nodes active at current tid
                for n in h.nodes(start=tid, end=tid):
                    n_id = str(n)
                    counts = node_neighbors_future.get(n_id, {})
                    for v, c in counts.items():
                        if c >= s and v != n_id:
                            all_edges.append((f"{n_id}_{tid}", f"{v}_{future_tid}", c))

    # Add all edges to graph
    for u, v, w in all_edges:
        DG.add_edge(u, v, weight=w)

    # Determine sources based on start_from
    if seeds:
        # If seeds specified, sources are only at timestamps where seed items have outgoing edges
        for seed_id in seeds:
            for tid in ids:
                node_label = f"{seed_id}_{tid}"
                # Check if this seed exists at this time and has outgoing edges
                if (
                    seed_id in item_at_time[tid]
                    and DG.has_node(node_label)
                    and DG.out_degree[node_label] > 0
                ):
                    sources[node_label] = None
    else:
        # If no seeds, sources are all items at the first timestamp with outgoing edges
        first_tid = ids[0]
        for item_id in item_at_time[first_tid]:
            node_label = f"{item_id}_{first_tid}"
            if DG.has_node(node_label) and DG.out_degree[node_label] > 0:
                sources[node_label] = None

    # Determine targets
    if stop_at is not None:
        # Only nodes matching stop_at are targets
        stop_id = str(stop_at)
        for tid in ids:
            node_label = f"{stop_id}_{tid}"
            if DG.has_node(node_label) and DG.in_degree[node_label] > 0:
                targets[node_label] = None
    else:
        # All reachable nodes (except sources) are potential targets
        for node in DG.nodes():
            if node not in sources and DG.in_degree[node] > 0:
                targets[node] = None

    excluded_ids = set(str(s).split("_")[0] for s in seeds) if seeds else set()
    final_targets = [t for t in targets if t.split("_")[0] not in excluded_ids]
    return DG, list(sources), final_targets


def time_respecting_s_walks(
    h: ASH,
    s: int,
    start_from: Union[str, List[str]],
    stop_at: Optional[str] = None,
    start: int = None,
    end: int = None,
    sample: float = 1,
) -> dict:
    """
    Enumerate all time-respecting s-walks between a given source and optionally a target hyperedge.

    :param h: The source hypergraph.
    :param s: Minimum number of shared nodes for s-incidence.
    :param start_from: ID o lista di iperarchi da cui partire.
    :param stop_at: Se fornito, considera solo cammini che terminano a questo iperarco.
    :param start: First snapshot to include.
    :param end: Last snapshot to include.
    :param sample: Fraction of source-target pairs to sample (0 < sample <= 1).

    :returns: Mapping (start_edge, end_edge) -> list of walks (TemporalEdge lists).
    """
    DAG, sources, targets = temporal_s_dag(
        h, s, start_from=start_from, stop_at=stop_at, start=start, end=end, edge=True
    )

    pairs = [(x, y) for x in sources for y in targets]
    if sample < 1:
        to_sample = int(len(pairs) * sample)
        idxs = np.random.choice(len(pairs), size=to_sample, replace=False)
        pairs = [pairs[i] for i in idxs]

    paths = []
    for src, dst in pairs:
        for path_nodes in nx.all_simple_paths(DAG, src, dst):
            seq = []
            for u, v in zip(path_nodes, path_nodes[1:]):
                t_from = int(u.split("_")[-1])
                t_to = int(v.split("_")[-1])
                w = DAG[u][v]["weight"]
                seq.append(TemporalEdge(u.split("_")[0], v.split("_")[0], w, t_to))

            # Validate time-respecting property: each step must happen at a strictly later time
            if len(seq) > 0:
                valid = True
                prev_edge = seq[0]
                for edge in seq[1:]:
                    # Each edge must occur at a strictly later timestamp
                    if edge.tid <= prev_edge.tid:
                        valid = False
                        break
                    # Also reject immediate back-and-forth between same pair of nodes
                    if edge.fr == prev_edge.to and edge.to == prev_edge.fr:
                        valid = False
                        break
                    prev_edge = edge

                if valid:
                    paths.append(seq)

    unique = list({tuple(w): w for w in paths}.values())
    res = defaultdict(list)
    for w in unique:
        key = (w[0].fr, w[-1].to)
        res[key].append(w)

    return res


def all_time_respecting_s_walks(
    h: ASH,
    s: int,
    start: int = None,
    end: int = None,
    sample: float = 1,
) -> dict:
    """
    Compute time-respecting s-walks originating from every hyperedge in the graph.

    :param h: The hypergraph.
    :param s: Minimum s-incidence threshold.
    :param start: Earliest snapshot to include.
    :param end: Latest snapshot to include.
    :param sample: Fraction of source-target samples per origin.

    :returns: Mapping (origin_edge, destination_edge) -> list of walks.
    """
    res = {}
    for he in h.hyperedges(start=start, end=end):
        subpaths = time_respecting_s_walks(
            h,
            s,
            start_from=he,
            stop_at=None,
            start=start,
            end=end,
            sample=sample,
        )
        for key, walks in subpaths.items():
            if walks:
                res[(he, key[1])] = walks
    return res


def annotate_walks(paths: list) -> dict:
    """
    Annotate a list of s-walks with standard path metrics.

    :param paths: The walks to classify.

    :returns: Dictionary of metric names to lists of walks.
    """
    metrics = []
    for p in paths:
        length = len(p)
        duration = p[-1].tid - p[0].tid
        weight = sum(e.weight for e in p)
        reach = p[-1].tid
        metrics.append(
            {
                "path": p,
                "length": length,
                "duration": duration,
                "weight": weight,
                "reach": reach,
            }
        )
    shortest = min(metrics, key=lambda m: m["length"])["length"]
    fastest = min(metrics, key=lambda m: m["duration"])["duration"]
    heaviest = max(metrics, key=lambda m: m["weight"])["weight"]
    foremost = min(metrics, key=lambda m: m["reach"])["reach"]

    def by(key, op, val):
        return [m["path"] for m in metrics if op(m[key], val)]

    return {
        "shortest": by("length", lambda x, y: x == y, shortest),
        "fastest": by("duration", lambda x, y: x == y, fastest),
        "heaviest": by("weight", lambda x, y: x == y, heaviest),
        "foremost": by("reach", lambda x, y: x == y, foremost),
        "shortest_fastest": by(
            "duration",
            lambda x, y: x == y,
            min(m["duration"] for m in metrics if m["length"] == shortest),
        ),
        "shortest_heaviest": by(
            "weight",
            lambda x, y: x == y,
            max(m["weight"] for m in metrics if m["length"] == shortest),
        ),
        "fastest_shortest": by(
            "length",
            lambda x, y: x == y,
            min(m["length"] for m in metrics if m["duration"] == fastest),
        ),
        "fastest_heaviest": by(
            "weight",
            lambda x, y: x == y,
            max(m["weight"] for m in metrics if m["duration"] == fastest),
        ),
        "heaviest_fastest": by(
            "duration",
            lambda x, y: x == y,
            max(m["duration"] for m in metrics if m["weight"] == heaviest),
        ),
        "heaviest_shortest": by(
            "length",
            lambda x, y: x == y,
            max(m["length"] for m in metrics if m["weight"] == heaviest),
        ),
    }


def walk_length(path: list) -> int:
    """
    Compute the number of edges in a temporal walk.

    :param path: The walk to measure.

    :returns: Number of steps in the walk.
    """
    return len(path)


def walk_duration(path: list) -> int:
    """
    Compute the duration of a temporal walk.

    :param path: The walk to measure.

    :returns: Time difference between first and last edge.
    """
    return int(path[-1].tid) - int(path[0].tid)


def walk_weight(path: list) -> int:
    """
    Compute the total weight of a temporal walk.

    :param path: The walk to measure.

    :returns: Cumulative weight of the walk.
    """
    return sum(p.weight for p in path)
