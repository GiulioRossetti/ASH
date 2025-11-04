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
    Intra-timestamp edges connect different items active at the same timestamp when s-incidence (edges) or co-membership (nodes) is satisfied.
    Forward-in-time edges x_t -> x_{t+1} exist only if the same item is active at both timestamps and are intended for waiting; consumers should not count them as steps.

    :param h: The source hypergraph.
    :param s: Minimum s-incidence threshold.
    :param start_from: Node or hyperedge id(s) to start from. If None, starts from all items present at the first snapshot in range.
    :param stop_at: Node or hyperedge id to stop at (optional).
    :param start: First snapshot ID to include. Defaults to earliest.
    :param end: Last snapshot ID to include. Defaults to latest.
    :param edge: If True, operate on hyperedges. If False, operate on nodes.
    :returns: (DAG, sources, targets) with labels "<id>_<tid>".
    :raises ValueError: If the [start, end] interval is not a valid subset of the graph's snapshot IDs.
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
    active = {s_id: None for s_id in seeds} if seeds else {}
    sources, targets = {}, {}

    for i, tid in enumerate(ids):
        # Precompute neighbors per time for node-mode to avoid repeated scans
        node_neighbors: dict[str, dict[str, int]] = {}
        if not edge:
            node_neighbors = defaultdict(lambda: defaultdict(int))
            for he in h.hyperedges(start=tid, end=tid):
                he_nodes = list(h.get_hyperedge_nodes(he))
                # count co-memberships for all pairs within this hyperedge
                for ii in range(len(he_nodes)):
                    u = str(he_nodes[ii])
                    for j in range(len(he_nodes)):
                        if ii == j:
                            continue
                        v = str(he_nodes[j])
                        node_neighbors[u][v] += 1

        to_remove: List[str] = []
        to_add: List[str] = []

        for an in list(active) if active else [None]:
            # If no explicit seeds, we start from all items present at current tid
            if an is None:
                # derive all possible current items
                if edge:
                    current_items = [str(he) for he in h.hyperedges(start=tid, end=tid)]
                else:
                    current_items = [str(n) for n in h.nodes(start=tid, end=tid)]
                # mark these as (implicit) sources for this tid
                for item in current_items:
                    sources[f"{item}_{tid}"] = None
                # consider them as 'an' for neighbor expansion below
                iter_items = current_items
            else:
                iter_items = [str(an)]

            for an_item in iter_items:
                base_id = str(an_item).split("_")[0]
                # Build neighbor set depending on mode
                if edge:
                    if not h.has_hyperedge(base_id, tid):
                        continue
                    raw_neighbors = h.get_s_incident(base_id, s=s, start=tid, end=tid)
                    neighbors = {
                        f"{n_id}_{tid}": w
                        for n_id, w in raw_neighbors
                        if n_id != base_id
                    }
                else:
                    # For nodes, rely on precomputed node_neighbors at this tid; if base_id is not active,
                    # counts will be empty and no edges will be added.
                    counts = node_neighbors.get(str(base_id), {})
                    neighbors = {
                        f"{v}_{tid}": c
                        for v, c in counts.items()
                        if c >= s and v != base_id
                    }

                # Update targets set for neighbor transitions
                if stop_at is not None:
                    key = f"{stop_at}_{tid}"
                    if key in neighbors:
                        targets[key] = None
                else:
                    for k in neighbors:
                        targets[k] = None

                # If no neighbors and not the original seed, remove from active
                if not neighbors and seeds and an_item not in seeds:
                    to_remove.append(an_item)

                # Add edges: ensure source is labeled with CURRENT tid to keep same-time transitions
                for n_label, w in neighbors.items():
                    an_node = f"{base_id}_{tid}"
                    # Mark as source only if it originates from a seed when seeds are provided
                    if seeds and base_id in seeds:
                        sources[an_node] = None
                    DG.add_edge(an_node, n_label, weight=w)
                    to_add.append(n_label)

        for n_label in to_add:
            active[n_label] = None
        for rm in to_remove:
            active.pop(rm, None)

        # Add forward-in-time edges (stay on the same entity if still active at next timestamp)
        if i < len(ids) - 1:
            next_tid = ids[i + 1]
            if edge:
                current_items = [str(he) for he in h.hyperedges(start=tid, end=tid)]
                next_items = set(
                    str(he) for he in h.hyperedges(start=next_tid, end=next_tid)
                )
                for item in current_items:
                    if item in next_items:
                        DG.add_edge(f"{item}_{tid}", f"{item}_{next_tid}", weight=1.0)
                        # Only mark forward targets along the seed chain
                        if item in seeds:
                            targets[f"{item}_{next_tid}"] = None
            else:
                current_items = [str(n) for n in h.nodes(start=tid, end=tid)]
                next_items = set(str(n) for n in h.nodes(start=next_tid, end=next_tid))
                for item in current_items:
                    if item in next_items:
                        DG.add_edge(f"{item}_{tid}", f"{item}_{next_tid}", weight=1.0)
                        if item in seeds:
                            targets[f"{item}_{next_tid}"] = None

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
                t_to = v.split("_")[-1]
                w = DAG[u][v]["weight"]
                seq.append(TemporalEdge(u.split("_")[0], v.split("_")[0], w, int(t_to)))
            if len(seq) <= 1:
                paths.append(seq)
            else:
                valid = True
                first = seq[0]
                for nxt in seq[1:]:
                    if (nxt.fr == first.to and nxt.to == first.fr) or (
                        nxt.tid == first.tid
                    ):
                        valid = False
                        break
                    first = nxt
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
