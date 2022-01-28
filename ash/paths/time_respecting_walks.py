from ash import ASH
from ash.paths.walks import is_s_path
import networkx as nx
import itertools
from collections import defaultdict, namedtuple
import random
import numpy as np
import copy

TemporalEdge = namedtuple("TemporalEdge", "fr to weight tid")
TemporalEdge.__new__.__defaults__ = (None,) * len(TemporalEdge._fields)


def temporal_s_dag(
    h: ASH,
    s: int,
    hyperedge_from: str,
    hyperedge_to: str = None,
    start: int = None,
    end: int = None,
) -> nx.DiGraph:
    """

    :param h:
    :param s:
    :param hyperedge_from:
    :param hyperedge_to:
    :param start:
    :param end:
    :return:
    """
    ids = h.temporal_snapshots_ids()
    if len(ids) == 0:
        return nx.DiGraph(), [], []

    # correcting missing values
    if end is None:
        end = ids[-1]

    if start is None:
        start = ids[0]

    if start < min(ids) or start > end or end > max(ids) or start > max(ids):
        raise ValueError(
            f"The specified interval {[start, end]} is not a proper subset of the network timestamps "
            f"{[min(ids), max(ids)]}."
        )

    # adjusting temporal window
    start = list([i >= start for i in ids]).index(True)
    end = end if end == ids[-1] else list([i >= end for i in ids]).index(True)
    ids = ids[start : end + 1]

    # creating empty DAG
    DG = nx.DiGraph()
    DG.add_node(hyperedge_from)
    active = {hyperedge_from: None}
    sources, targets = {}, {}

    for tid in ids:
        to_remove = []
        to_add = []

        for an in active:

            if not h.has_hyperedge_id(str(an).split("_")[0], tid=tid):
                continue

            neighbors = {
                f"{n[0]}_{tid}": n[1]
                for n in h.get_s_incident(
                    str(an).split("_")[0], s=s, start=tid, end=tid
                )
            }

            if hyperedge_to is not None:
                if f"{hyperedge_to}_{tid}" in neighbors:
                    targets[f"{hyperedge_to}_{tid}"] = None
            else:
                for k in neighbors:
                    targets[k] = None

            if len(neighbors) == 0 and an != hyperedge_from:
                to_remove.append(an)

            for n in neighbors:
                if "_" not in an:
                    an = f"{an}_{tid}"
                    sources[an] = None

                DG.add_edge(an, n, weight=neighbors[n])
                to_add.append(n)

        for n in to_add:
            active[n] = None

        for rm in to_remove:
            del active[rm]

    targets = [t for t in targets if t.split("_")[0] != hyperedge_from]

    return DG, list(sources), list(targets)


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

    :param h:
    :param s:
    :param hyperedge_from:
    :param hyperedge_to:
    :param start:
    :param end:
    :param sample:
    :return:
    """

    DAG, sources, targets = temporal_s_dag(
        h, s, hyperedge_from, hyperedge_to, start=start, end=end
    )

    pairs = [(x, y) for x in sources for y in targets]

    if sample < 1:
        to_sample = int(len(pairs) * sample)
        pairs_idx = np.random.choice(len(pairs), size=to_sample, replace=False)
        pairs = np.array(pairs)[pairs_idx]

    paths = []
    for pair in pairs:
        path = list(nx.all_simple_paths(DAG, pair[0], pair[1]))

        for p in path:
            pt = []
            for first, second in zip(p, p[1:]):
                hyperedge_from = first.split("_")
                if len(hyperedge_from) == 2:
                    hyperedge_from = hyperedge_from[0]
                else:
                    hyperedge_from = "_".join(hyperedge_from[0:-1])

                hyperedge_to = second.split("_")
                if len(hyperedge_to) == 2:
                    t = hyperedge_to[1]
                    hyperedge_to = hyperedge_to[0]
                else:
                    t = hyperedge_to[-1]
                    hyperedge_to = "_".join(hyperedge_to[0:-1])

                pt.append(
                    TemporalEdge(
                        hyperedge_from,
                        hyperedge_to,
                        DAG[first][second]["weight"],
                        int(t),
                    )
                )
            # check ping pong

            flag = True
            if len(pt) > 1:
                s = pt[0]
                for l in pt[1:]:
                    if l[0] == s[1] and l[1] == s[0] or l[2] == s[2]:
                        flag = False
                        continue
                    s = l

            if flag:
                paths.append(pt)

    pa = list(dict.fromkeys([tuple(x) for x in paths]))

    res = defaultdict(list)
    for p in pa:
        k = (p[0][0], p[-1][1])
        res[k].append(p)

    return res


def all_time_respecting_s_walks(
    h: ASH, s: int, start: int = None, end: int = None, sample: float = 1
) -> dict:
    """

    :param h:
    :param s:
    :param start:
    :param end:
    :param sample:
    :return:
    """
    res = {}
    for he in h.hyperedge_id_iterator():
        paths = time_respecting_s_walks(
            h,
            s=s,
            hyperedge_from=he,
            hyperedge_to=None,
            start=start,
            end=end,
            sample=sample,
        )
        if len(paths) > 0:
            for k, path in paths.items():
                v = k[-1]
                res[(he, v)] = path

    return res


def annotate_walks(paths: list) -> dict:
    """

    :param h:
    :param paths:
    :return:
    """
    annotated = {
        "shortest": [],
        "fastest": [],
        "shortest_fastest": [],
        "shortest_heaviest": [],
        "fastest_shortest": [],
        "fastest_heaviest": [],
        "foremost": [],
        "heaviest": [],
        "heaviest_fastest": [],
        "heaviest_shortest": [],
    }

    min_to_reach = None
    shortest = None
    fastest = None
    p_weight = None

    for path in paths:

        length = walk_length(path)
        duration = walk_duration(path)
        weight = walk_weight(path)
        reach = path[-1][-1]

        if p_weight is None or weight > p_weight:
            p_weight = weight
            annotated["heaviest"] = [copy.copy(path)]
        else:
            annotated["heaviest"].append(copy.copy(path))

        if shortest is None or length < shortest:
            shortest = length
            annotated["shortest"] = [copy.copy(path)]
        elif length == shortest:
            annotated["shortest"].append(copy.copy(path))

        if fastest is None or duration < fastest:
            fastest = duration
            annotated["fastest"] = [copy.copy(path)]
        elif duration == fastest:
            annotated["fastest"].append(copy.copy(path))

        if min_to_reach is None or reach < min_to_reach:
            min_to_reach = reach
            annotated["foremost"] = [copy.copy(path)]
        elif reach == min_to_reach:
            annotated["foremost"].append(copy.copy(path))

    fastest_shortest = {
        tuple(path): walk_duration(path) for path in annotated["shortest"]
    }
    minval = min(fastest_shortest.values())
    fastest_shortest = list(
        [x for x in fastest_shortest if fastest_shortest[x] == minval]
    )

    fastest_heaviest = {
        tuple(path): walk_duration(path) for path in annotated["heaviest"]
    }
    minval = min(fastest_heaviest.values())
    fastest_heaviest = list(
        [x for x in fastest_heaviest if fastest_heaviest[x] == minval]
    )

    shortest_fastest = {tuple(path): walk_length(path) for path in annotated["fastest"]}
    minval = min(shortest_fastest.values())
    shortest_fastest = list(
        [x for x in shortest_fastest if shortest_fastest[x] == minval]
    )

    shortest_heaviest = {
        tuple(path): walk_weight(path) for path in annotated["heaviest"]
    }
    minval = min(shortest_heaviest.values())
    shortest_heaviest = list(
        [x for x in shortest_heaviest if shortest_heaviest[x] == minval]
    )

    heaviest_shortest = {
        tuple(path): walk_weight(path) for path in annotated["shortest"]
    }
    maxval = max(heaviest_shortest.values())
    heaviest_shortest = list(
        [x for x in heaviest_shortest if heaviest_shortest[x] == maxval]
    )

    heaviest_fastest = {tuple(path): walk_weight(path) for path in annotated["fastest"]}
    maxval = max(heaviest_fastest.values())
    heaviest_fastest = list(
        [x for x in heaviest_fastest if heaviest_fastest[x] == maxval]
    )

    annotated["fastest_shortest"] = [list(p) for p in fastest_shortest]
    annotated["fastest_heaviest"] = [list(p) for p in fastest_heaviest]
    annotated["shortest_fastest"] = [list(p) for p in shortest_fastest]
    annotated["shortest_heaviest"] = [list(p) for p in shortest_heaviest]
    annotated["heaviest_shortest"] = [list(p) for p in heaviest_shortest]
    annotated["heaviest_fastest"] = [list(p) for p in heaviest_fastest]

    return annotated


def walk_length(path: list) -> int:
    """

    :param path:
    :return:
    """
    return len(path)


def walk_duration(path: list) -> int:
    """

    :param path:
    :return:
    """

    return int(path[-1][-1]) - int(path[0][-1])


def walk_weight(path: list) -> int:
    """

    :param path:
    :return:
    """
    return sum([p.weight for p in path])
