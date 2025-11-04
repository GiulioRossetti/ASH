from ash_model.paths.randwalks import *
from typing import Union, List, Optional, Dict
import numpy as np

from ash_model.classes import ASH
from collections import defaultdict


def rwhs(
    h: ASH,
    s: int,
    tid: int,
    start_from: Union[int, str, List[Union[int, str]], None] = None,
    num_walks: int = 100,
    walk_length: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    edge: bool = False,
    method: str = "meet",
    threads: int = -1,
) -> Dict:
    """
    Compute Random Walk Hypergraph Similarity (RWHS) scores for nodes in a hypergraph.

    :param h: ASH instance.
    :param s: Minimum s-incidence threshold. For node walks: nodes must co-occur in at least s hyperedges.
              For edge walks: hyperedges must share at least s nodes.
    :param tid: Temporal snapshot id.
    :param start_from: Node(s) or hyperedge id(s) from which to start the random walks. If
                          None, random walks will start from all nodes (or all hyperedges if edge=True).
    :param num_walks: Number of random walks to perform.
    :param walk_length: Length of each random walk.
    :param p: Return parameter for the random walk.
    :param q: In-out parameter for the random walk.
    :param edge: If True, perform random walks on the hyperedge line graph.
    :param method: Method to compute RWHS scores. Supported methods are 'meet' and 'jump'.
    :param threads: Number of threads to use for parallel computation. Default is -1 (use all available threads).
    :return: Dictionary with nodes as keys and their RWHS scores as values for each attribute
    """

    scores = defaultdict(lambda: defaultdict(float))
    walks = random_walks(
        h,
        start_from,
        num_walks,
        walk_length,
        p,
        q,
        s,
        edge,
        start=tid,
        end=tid,
        threads=threads,
    )

    if method == "meet":
        # Meet-wise RWHS is the ratio of nodes with the same attribute value as v that appear in
        # a realization within W t,k, averaged over all realizations
        for walk in walks:
            start_node = walk[0]
            for other in walk:
                if start_node != other:
                    for attr, val in h.get_node_attributes(start_node, tid=tid).items():
                        if h.get_node_attributes(other, tid=tid).get(attr) == val:
                            scores[start_node][attr] += 1.0
        for v in scores:
            for attr in scores[v]:
                scores[v][attr] /= num_walks * (walk_length - 1)

    elif method == "jump":
        # jump-wise RWHS is defined as the ratio of pairs of nodes that have the same attribute value
        # and are sequentially adjacent in a realization within W t,k
        # averaged over all realizations
        for walk in walks:
            start_node = walk[0]
            for i in range(len(walk) - 1):
                node1 = walk[i]
                node2 = walk[i + 1]

                for attr, val in h.get_node_attributes(node1, tid=tid).items():
                    if h.get_node_attributes(node2, tid=tid).get(attr) == val:
                        scores[start_node][attr] += 1.0
        for v in scores:
            for attr in scores[v]:
                scores[v][attr] /= num_walks * (walk_length - 1)
    else:
        raise ValueError(
            f"Unknown method {method}. Supported methods are 'meet' and 'jump'."
        )
    return scores


# uses time-respecting random walks
def temporal_rwhs(
    h: ASH,
    s: int,
    tid: int,
    start_from: Optional[Union[int, str, List[Union[int, str]]]] = None,
    end_at: Optional[Union[int, str]] = None,
    num_walks: int = 100,
    walk_length: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    method: str = "meet",
    threads: int = -1,
) -> Dict:
    """
    Compute Temporal Random Walk Hypergraph Similarity (RWHS) scores for nodes in a hypergraph.

    :param h: ASH instance.
    :param s: Order of the hypergraph (minimum s-incidence threshold).
    :param tid: Temporal snapshot id.
    :param start_from: Hyperedge(s) or hyperedge id(s) from which to start the random walks. If
                          None, random walks will start from all hyperedges.
    :param end_at: Hyperedge or hyperedge id to which to end the random walks. If None, random walks
                         will end at all hyperedges.
    :param num_walks: Number of random walks to perform.
    :param walk_length: Length of each random walk.
    :param p: Return parameter for the random walk.
    :param q: In-out parameter for the random walk.
    :param method: Method to compute RWHS scores. Supported methods are 'meet' and 'jump'.
    :param threads: Number of threads to use for parallel computation. Default is -1 (use all available threads).
    :return: Dictionary with nodes as keys and their RWHS scores as values for each attribute
    """

    scores = defaultdict(lambda: defaultdict(float))
    walks = time_respecting_random_walks(
        h,
        s,
        start_from,
        end_at,
        start=tid,
        end=tid,
        num_walks=num_walks,
        walk_length=walk_length,
        p=p,
        q=q,
        edge=True,
        threads=threads,
    )

    if method == "meet":
        # Meet-wise RWHS is the ratio of nodes with the same attribute value as v that appear in
        # a realization within W t,k, averaged over all realizations
        for walk_list in walks.values():
            for walk in walk_list:
                if len(walk) == 0:
                    continue

                start_hyperedge = walk[0].src.split("_")[0]  # Extract base hyperedge id
                nodes_in_start_hyperedge = h.get_hyperedge_nodes(start_hyperedge)

                # Track all nodes visited in the walk
                visited_nodes = set()
                for temporal_edge in walk:
                    hedge_id = temporal_edge.dst.split("_")[
                        0
                    ]  # Extract base hyperedge id
                    visited_nodes.update(h.get_hyperedge_nodes(hedge_id))

                # Compute scores for each node in the starting hyperedge
                for start_node in nodes_in_start_hyperedge:
                    for other_node in visited_nodes:
                        if start_node != other_node:
                            start_attrs = h.get_node_attributes(start_node, tid=tid)
                            other_attrs = h.get_node_attributes(other_node, tid=tid)
                            for attr, val in start_attrs.items():
                                if other_attrs.get(attr) == val:
                                    scores[start_node][attr] += 1.0

        # Normalize scores
        total_walks = sum(len(walk_list) for walk_list in walks.values())
        if total_walks > 0:
            for v in scores:
                for attr in scores[v]:
                    # Normalize by total number of walks and average walk length
                    scores[v][attr] /= total_walks

    elif method == "jump":
        # Jump-wise RWHS is defined as the ratio of pairs of nodes that have the same attribute value
        # and are sequentially adjacent in a realization within W t,k
        # averaged over all realizations
        for walk_list in walks.values():
            for walk in walk_list:
                if len(walk) < 2:
                    continue

                start_hyperedge = walk[0].src.split("_")[0]
                nodes_in_start_hyperedge = h.get_hyperedge_nodes(start_hyperedge)

                # Iterate through consecutive hyperedge pairs in the walk
                for i in range(len(walk) - 1):
                    hedge1_id = walk[i].dst.split("_")[0]
                    hedge2_id = walk[i + 1].dst.split("_")[0]

                    nodes1 = h.get_hyperedge_nodes(hedge1_id)
                    nodes2 = h.get_hyperedge_nodes(hedge2_id)

                    # Check all pairs of nodes from consecutive hyperedges
                    for node1 in nodes1:
                        for node2 in nodes2:
                            if node1 in nodes_in_start_hyperedge:
                                attrs1 = h.get_node_attributes(node1, tid=tid)
                                attrs2 = h.get_node_attributes(node2, tid=tid)
                                for attr, val in attrs1.items():
                                    if attrs2.get(attr) == val:
                                        scores[node1][attr] += 1.0

        # Normalize scores
        total_walks = sum(len(walk_list) for walk_list in walks.values())
        if total_walks > 0:
            for v in scores:
                for attr in scores[v]:
                    scores[v][attr] /= total_walks
    else:
        raise ValueError(
            f"Unknown method {method}. Supported methods are 'meet' and 'jump'."
        )

    return scores
