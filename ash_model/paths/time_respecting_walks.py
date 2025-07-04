import copy
from collections import defaultdict, namedtuple
from typing import Tuple

import networkx as nx
import numpy as np

from ash_model import ASH

# A temporal edge in an s-walk: from-hyperedge id, to-hyperedge id, weight, and timestamp.
TemporalEdge = namedtuple("TemporalEdge", "fr to weight tid")
TemporalEdge.__new__.__defaults__ = (None,) * len(TemporalEdge._fields)


def temporal_s_dag(
    h: ASH,
    s: int,
    hyperedge_from: str,
    hyperedge_to: str = None,
    start: int = None,
    end: int = None,
) -> Tuple[nx.DiGraph, list, list]:
    """
    Build a time-respecting directed acyclic graph (DAG) of s-incidence transitions
    among hyperedges over a specified temporal window.

    :param h: The source hypergraph.
    :param s: Minimum number of shared nodes to define an s-incident transition.
    :param hyperedge_from: Identifier of the starting hyperedge.
    :param hyperedge_to: If given, only transitions leading to this hyperedge are considered as targets.
    :param start: First snapshot ID to include. Defaults to earliest.
    :param end: Last snapshot ID to include. Defaults to latest.
    :returns: A tuple (DAG, sources, targets).
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
    end_idx = (
        len(ids) - 1
        if end == ids[-1]
        else next(i for i, t in enumerate(ids) if t >= end)
    )
    ids = ids[start_idx : end_idx + 1]

    DG = nx.DiGraph()
    DG.add_node(hyperedge_from)
    active = {hyperedge_from: None}
    sources, targets = {}, {}

    for tid in ids:
        to_remove = []
        to_add = []

        for an in list(active):
            if not h.has_hyperedge(str(an).split("_")[0], tid):
                continue

            raw_neighbors = h.get_s_incident(
                str(an).split("_")[0], s=s, start=tid, end=tid
            )
            neighbors = {f"{n[0]}_{tid}": n[1] for n in raw_neighbors}

            if hyperedge_to:
                if f"{hyperedge_to}_{tid}" in neighbors:
                    targets[f"{hyperedge_to}_{tid}"] = None
            else:
                for k in neighbors:
                    targets[k] = None

            if not neighbors and an != hyperedge_from:
                to_remove.append(an)

            for n_label, w in neighbors.items():
                if "_" not in an:
                    an_time = f"{an}_{tid}"
                    sources[an_time] = None
                    an_node = an_time
                else:
                    an_node = an
                DG.add_edge(an_node, n_label, weight=w)
                to_add.append(n_label)

        for n_label in to_add:
            active[n_label] = None
        for rm in to_remove:
            active.pop(rm, None)

    final_targets = [t for t in targets if t.split("_")[0] != hyperedge_from]
    return DG, list(sources), final_targets


def time_respecting_s_walks(
    h: ASH,
    s: int,
    hyperedge_from: str,
    hyperedge_to: str = None,
    start: int = None,
    end: int = None,
    sample: float = 1,
) -> dict:
    """
    Enumerate all time-respecting s-walks between a given source and optionally a target hyperedge.

    :param h: The source hypergraph.
    :param s: Minimum number of shared nodes for s-incidence.
    :param hyperedge_from: ID of the starting hyperedge.
    :param hyperedge_to: If provided, only walks ending at this hyperedge are considered.
    :param start: First snapshot to include.
    :param end: Last snapshot to include.
    :param sample: Fraction of source-target pairs to sample (0 < sample <= 1).
    :returns: Mapping (start_edge, end_edge) -> list of walks (TemporalEdge lists).
    """
    DAG, sources, targets = temporal_s_dag(
        h, s, hyperedge_from, hyperedge_to, start=start, end=end
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
            hyperedge_from=he,
            hyperedge_to=None,
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
