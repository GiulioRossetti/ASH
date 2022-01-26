from ash import ASH
import networkx as nx
import itertools
from collections import defaultdict
import random
import numpy as np
import copy


def temporal_s_dag(h: ASH, s: int, hyperedge_from: str, hyperedge_to: str = None, start: int = None, end: int = None) -> nx.DiGraph:
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
        raise ValueError(f"The specified interval {[start, end]} is not a proper subset of the network timestamps "
                         f"{[min(ids), max(ids)]}.")

    # adjusting temporal window
    start = list([i >= start for i in ids]).index(True)
    end = end if end == ids[-1] else list([i >= end for i in ids]).index(True)
    ids = ids[start:end+1]

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

            neighbors = {f"{n}_{tid}": None for n in h.get_s_incident(str(an).split("_")[0], s=s, start=tid, end=tid)}

            if hyperedge_to is not None:
                if f"{hyperedge_to}_{tid}" in neighbors:
                    targets[f"{hyperedge_to}_{tid}"] = None
            else:
                for k in neighbors:
                    targets[k] = None

            if len(neighbors) == 0 and an != hyperedge_from:
                to_remove.append(an)

            for n in neighbors:
                if '_' not in an:
                    an = f"{an}_{tid}"
                    sources[an] = None

                DG.add_edge(an, n)
                to_add.append(n)

        for n in to_add:
            active[n] = None

        for rm in to_remove:
            del active[rm]

    targets = [t for t in targets if t.split("_")[0] != hyperedge_from]

    return DG, list(sources), list(targets)
