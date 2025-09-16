from collections import defaultdict, Counter
from itertools import combinations
from typing import Iterator, List, Dict, Optional, Union, Tuple

import networkx as nx

from ash_model import ASH


def all_simple_paths(
    h: ASH,
    s: int,
    hyperedge_a: Optional[str] = None,
    hyperedge_b: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    cutoff: Optional[int] = None,
) -> Iterator[List[str]]:
    """
    Generate all simple, s-overlap-valid paths in a hypergraph's line graph.

    A simple path is one with no repeated hyperedges, and additionally, the
    sequence must satisfy the s-overlap condition in the original hypergraph.

    Internally, this projects the hypergraph into its s-line-graph and then
    yields each NetworkX simple path that also passes the is_s_path filter.

    :param h:            ASH hypergraph instance.
    :param s:            Minimum overlap size between consecutive hyperedges.
    :param hyperedge_a:  Identifier of the starting hyperedge (optional). If None,
                         paths may start from any node.
    :param hyperedge_b:  Identifier of the ending hyperedge (optional). If None,
                         paths may end at any node.
    :param start:        Lower time bound for hyperedges (inclusive).
    :param end:          Upper time bound for hyperedges (inclusive).
    :param cutoff:       Maximum path length (number of nodes) to explore.
    :return:             Iterator over valid paths, each as a list of hyperedge IDs.
    """
    # Build the s-overlap line graph
    lg = h.s_line_graph(s, start, end)

    # If specific endpoints are requested, ensure they exist
    if hyperedge_a and hyperedge_a not in lg:
        return iter([])
    if hyperedge_b and hyperedge_b not in lg:
        return iter([])

    # Yield only the simple paths that satisfy the s-path condition
    for path in nx.all_simple_paths(
        lg, source=hyperedge_a, target=hyperedge_b, cutoff=cutoff
    ):
        if is_s_path(h, path):
            yield path


def shortest_s_path(
    h: ASH,
    s: int,
    hyperedge_a: str,
    hyperedge_b: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    cutoff: Optional[int] = None,
) -> List[List[str]]:
    """
    Return all shortest simple, s-overlap-valid paths between two hyperedges.

    If hyperedge_b is None, this returns the shortest paths from A to all others.

    :param h:            ASH hypergraph instance.
    :param s:            Minimum overlap size.
    :param hyperedge_a:  Starting hyperedge ID.
    :param hyperedge_b:  Ending hyperedge ID (optional).
    :param start:        Start time bound (inclusive).
    :param end:          End time bound (inclusive).
    :param cutoff:       Maximum path length.
    :return:             List of shortest paths, each as list of IDs.
    """
    paths = list(all_simple_paths(h, s, hyperedge_a, hyperedge_b, start, end, cutoff))
    if not paths:
        return []
    min_len = min(len(p) for p in paths)
    return [p for p in paths if len(p) == min_len]


def all_shortest_s_paths(
    h: ASH,
    s: int,
    hyperedge_a: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    cutoff: Optional[int] = None,
) -> Dict[Tuple[str, str], List[List[str]]]:
    """
    Compute shortest s-paths for all hyperedge pairs (or from a source).

    When hyperedge_a is given, returns shortest paths from that source to all
    other hyperedges. Otherwise, returns shortest paths for every unordered
    pair, keyed both ways for convenience.

    :param h:            ASH hypergraph instance.
    :param s:            Minimum overlap size.
    :param hyperedge_a:  Source hyperedge ID (optional).
    :param start:        Start time bound (inclusive).
    :param end:          End time bound (inclusive).
    :param cutoff:       Maximum path length.
    :return:             Dict mapping (u,v) to list of shortest paths.
    """
    result: Dict[Tuple[str, str], List[List[str]]] = defaultdict(list)
    hyperedges = list(h.hyperedges(start, end))

    if hyperedge_a:
        # From one source to all others
        for he in hyperedges:
            if he == hyperedge_a:
                continue
            paths = list(all_simple_paths(h, s, hyperedge_a, he, start, end, cutoff))
            if not paths:
                continue
            min_len = min(len(p) for p in paths)
            result[(hyperedge_a, he)] = [p for p in paths if len(p) == min_len]
    else:
        # Pairwise between all
        for i, u in enumerate(hyperedges):
            for v in hyperedges[i + 1 :]:
                paths = list(all_simple_paths(h, s, u, v, start, end, cutoff))
                if not paths:
                    continue
                min_len = min(len(p) for p in paths)
                best = [p for p in paths if len(p) == min_len]
                result[(u, v)] = best
                result[(v, u)] = [list(reversed(p)) for p in best]
    return result


def all_shortest_s_path_lengths(
    h: ASH,
    s: int,
    hyperedge_a: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    cutoff: Optional[int] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Compute lengths of shortest s-overlap paths between hyperedges.

    When hyperedge_a is provided, returns dict lengths from source to others;
    otherwise returns full matrix mapping each hyperedge to every other.

    :param h:            ASH hypergraph instance.
    :param s:            Minimum overlap size.
    :param hyperedge_a:  Source hyperedge ID (optional).
    :param start:        Start time bound (inclusive).
    :param end:          End time bound (inclusive).
    :param cutoff:       Maximum path length.
    :return:             Nested dict of distances (number of edges).
    """
    result: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hyperedges = list(h.hyperedges(start, end))

    if hyperedge_a:
        # Distances from source
        for he in hyperedges:
            if he == hyperedge_a:
                result[hyperedge_a][he] = 0
                continue
            paths = list(all_simple_paths(h, s, hyperedge_a, he, start, end, cutoff))
            if paths:
                length = min(len(p) for p in paths) - 1
                result[hyperedge_a][he] = length
                result[he][hyperedge_a] = length
    else:
        # Full pairwise
        for i, u in enumerate(hyperedges):
            result[u][u] = 0
            for v in hyperedges[i + 1 :]:
                paths = list(all_simple_paths(h, s, u, v, start, end, cutoff))
                if paths:
                    length = min(len(p) for p in paths) - 1
                    result[u][v] = length
                    result[v][u] = length
    return result


def all_shortest_s_walks(
    h: ASH,
    s: int,
    hyperedge_a: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    weight: bool = False,
    edge: bool = True,
) -> Union[List[str], Dict[str, List[str]]]:
    """
    Retrieve the shortest s-walk(s) in the hypergraph or its dual.

    Delegates to shortest_s_walk.

    :param h:            ASH instance.
    :param s:            Minimum overlap size.
    :param hyperedge_a:  Source hyperedge ID (optional).
    :param start:        Start time bound.
    :param end:          End time bound.
    :return:             Single path (list) if two endpoints specified, or dict of
                         {target: path} if one endpoint, or nested dict if neither.
    """

    return shortest_s_walk(h, s, hyperedge_a, None, start, end, weight, edge)


def all_shortest_s_walk_lengths(
    h: ASH,
    s: int,
    hyperedge_a: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Compute lengths of shortest s-walks between hyperedges.

    :param h:            ASH instance.
    :param s:            Minimum overlap size.
    :param hyperedge_a:  Source hyperedge ID (optional).
    :param start:        Start time bound.
    :param end:          End time bound.
    :return:             Nested dict mapping hyperedge to target to length.
    """
    walks = all_shortest_s_walks(h, s, hyperedge_a, start, end)
    lengths: Dict[str, Dict[str, int]] = {}
    # walks may be dict or list
    if isinstance(walks, dict):
        for tgt, path in walks.items():
            lengths.setdefault(hyperedge_a, {})[tgt] = len(path)
            lengths.setdefault(tgt, {})[hyperedge_a] = len(path)
    return lengths


def shortest_s_walk(
    h: ASH,
    s: int,
    fr: Optional[str] = None,
    to: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    weight: bool = False,
    edge: bool = True,
) -> Union[List[str], Dict[str, List[str]]]:
    """
    Compute shortest s-walk(s) considering weight or dual hypergraph.

    If edge=True, operates on s-line-graph; otherwise uses dual hypergraph node graph.

    :param h:     ASH instance.
    :param s:     Minimum overlap size.
    :param fr:    Starting hyperedge ID (node or edge depending).
    :param to:    Ending hyperedge ID.
    :param start: Start time bound.
    :param end:   End time bound.
    :param weight:Use edge weight attribute 'w' if True.
    :param edge:  If False, compute on dual hypergraph.
    :return:      Single path list if both fr and to set, else dict mapping.
    """
    # Build appropriate graph
    if edge:
        g = h.s_line_graph(s, start, end)
        if fr and fr not in g or to and to not in g:
            return []
        if weight:
            return nx.shortest_path(g, source=fr, target=to, weight="w")
        return nx.shortest_path(g, source=fr, target=to)
    else:
        # Dual graph handling
        h1, node_to_eid = h.dual_hypergraph(start=start, end=end)
        eid_to_node = {eid: node for node, eid in node_to_eid.items()}
        # map fr/to to dual nodes
        src = node_to_eid.get(fr, fr)
        dst = node_to_eid.get(to, to)
        raw = shortest_s_walk(h1, s, src, dst, start, end, weight, edge=True)
        # remap back
        if isinstance(raw, dict):
            return {
                eid_to_node[k]: [eid_to_node[n] for n in path]
                for k, path in raw.items()
            }
        return [eid_to_node[n] for n in raw]


def closed_s_walk(
    h: ASH,
    s: int,
    hyperedge_a: str,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> List[List[str]]:
    """
    Find all simple cycles (basis) in the s-line-graph containing a given node.

    :param h:            ASH instance.
    :param s:            Minimum overlap size.
    :param hyperedge_a:  Node ID in s-line-graph to find cycles through.
    :param start:        Start time bound.
    :param end:          End time bound.
    :return:             List of cycles, each as list of hyperedge IDs.
    """
    g = h.s_line_graph(s, start, end)
    if hyperedge_a not in g:
        return []
    return nx.cycle_basis(g, hyperedge_a)


def s_distance(
    h: ASH,
    s: int,
    fr: Optional[str] = None,
    to: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    weight: bool = False,
    edge: bool = True,
) -> Union[int, Dict[str, Dict[str, int]], None]:
    """
    Compute shortest-path distances in s-line-graph or dual hypergraph.
    If edge=True, uses s-line-graph; otherwise uses dual hypergraph.

    :param h:    ASH instance.
    :param s:    Minimum overlap size.
    :param fr:   Source ID.
    :param to:   Target ID.
    :param start:Start time bound.
    :param end:  End time bound.
    :param weight:Use edge weight attribute 'w' if True.
    :param edge: If False, use dual hypergraph distances.
    :return:     Single distance, nested dict of distances, or None if unreachable.
    """
    if edge:
        G = h.s_line_graph(s, start, end)
        # guard: if either endpoint is missing, bail
        if (fr and fr not in G) or (to and to not in G):
            return None
        # pick the right algorithm
        if fr and to:
            return nx.shortest_path_length(
                G, source=fr, target=to, weight="w" if weight else None
            )
        elif fr:
            # distances from fr to all reachable nodes
            return nx.single_source_dijkstra_path_length(
                G, fr, weight="w" if weight else None
            )
        else:
            # distances from fr to all reachable nodes
            return {
                fr: nx.single_source_dijkstra_path_length(
                    G, fr, weight="w" if weight else None
                )
                for fr in G.nodes()
            }
    else:
        # build dual and invert its node‐to‐eid map
        H_dual, node_to_eid = h.dual_hypergraph(
            start=start, end=end
        )  # node represent hyperedges
        eid_to_node = {
            eid: node for node, eid in node_to_eid.items()
        }  # maps projected hyperedges to real nodes
        # while node_to_eid maps real nodes to projected hyperedges

        # translate your fr/to into the dual graph’s hyperedges
        src = node_to_eid.get(fr, fr)
        dst = node_to_eid.get(to, to)

        # compute distances on the dual graph
        dual_res = s_distance(
            H_dual, s, fr=src, to=dst, start=start, end=end, edge=True, weight=weight
        )
        # raise ValueError(f"{dual_res}, {src}, {dst}")
        if dual_res is None:
            return None

        # if it’s a single integer, map back and return
        if isinstance(dual_res, int):
            return dual_res

        # if it's a nested dict, map it back
        if isinstance(dual_res, dict) and any(
            isinstance(v, dict) for v in dual_res.values()
        ):
            remapped: Dict[str, Dict[str, int]] = {}
            for dual_src, targets in dual_res.items():
                src = eid_to_node.get(dual_src, dual_src)
                remapped[src] = {}
                for dual_trg, dist in targets.items():
                    trg = eid_to_node.get(dual_trg, dual_trg)
                    remapped[src][trg] = dist
            return remapped

        # else it must be a normal dict
        remapped: Dict[str, int] = {}
        for dual_trg, dist in dual_res.items():
            # remap the target hyperedge ID
            trg = eid_to_node.get(dual_trg, dual_trg)
            # remap the source hyperedge IDs
            remapped[trg] = dist
        print(f"remapped: {remapped}")
        return remapped


def average_s_distance(
    h: ASH,
    s: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    weight: bool = False,
    edge: bool = True,
) -> float:
    """
    Compute the average shortest-path length in the s-line-graph.

    :param h:    ASH instance.
    :param s:    Minimum overlap size.
    :param start:Start time bound.
    :param end:  End time bound.
    :param weight:Use edge weight attribute 'w' if True (unused here).
    :param edge: (Unused) Included for API consistency.
    :return:     Average distance across all connected pairs.
    """
    if not edge:
        h, _ = h.dual_hypergraph(start=start, end=end)

    g = h.s_line_graph(s, start, end)

    if weight:
        return nx.average_shortest_path_length(g, weight="w")
    return nx.average_shortest_path_length(g)


def has_s_walk(
    h: ASH,
    s: int,
    fr: Optional[str] = None,
    to: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    edge: bool = True,
) -> bool:
    """
    Determine if an s-overlap walk exists between two hyperedges.

    :param h:    ASH instance.
    :param s:    Minimum overlap size.
    :param fr:   Source hyperedge ID.
    :param to:   Target hyperedge ID.
    :param start:Start time bound.
    :param end:  End time bound.
    :param edge: If False, evaluate on dual hypergraph.
    :return:     True if a path exists, False otherwise.
    """
    if edge:
        g = h.s_line_graph(s, start, end)
        if fr and fr not in g or to and to not in g:
            return False
        return nx.has_path(g, fr, to)
    else:
        h1, node_to_eid = h.dual_hypergraph(start=start, end=end)
        src = node_to_eid.get(fr, fr)
        dst = node_to_eid.get(to, to)
        return has_s_walk(h1, s, src, dst, start, end, edge=True)


def s_diameter(
    h: ASH,
    s: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    weight: bool = False,
    edge: bool = True,
) -> int:
    """
    Compute the diameter (longest shortest-path) of the s-line-graph.

    :param h:      ASH instance.
    :param s:      Minimum overlap size.
    :param start:  Start time bound.
    :param end:    End time bound.
    :param weight: Use weighted distances if True.
    :param edge:   If False, compute on dual hypergraph.
    :return:       Integer diameter of the graph (0 if empty).
    """
    if not edge:
        h, _ = h.dual_hypergraph(start=start, end=end)
    g = h.s_line_graph(s, start, end)
    # lcc
    comps = list(nx.connected_components(g))
    lcc = max(comps, key=len) if comps else set()
    lcc = g.subgraph(lcc)
    diam = nx.diameter(lcc, weight="w" if weight else None)
    return diam if diam is not None else 0


def s_components(
    h: ASH,
    s: int,
    start: Optional[int] = None,
    end: Optional[int] = None,
    edge: bool = True,
) -> Iterator[set]:
    """
    Yield connected components of the s-line-graph or its dual.

    :param h:    ASH instance.
    :param s:    Minimum overlap size.
    :param start:Start time bound.
    :param end:  End time bound.
    :param edge: If False, yields on dual hypergraph components.
    :return:     Iterator of sets of hyperedge IDs per component.
    """
    if edge:
        g = h.s_line_graph(s, start, end)
        yield from nx.connected_components(g)
    else:
        h1, node_to_eid = h.dual_hypergraph(start=start, end=end)
        translate = {eid: node for node, eid in node_to_eid.items()}
        # iterate over components in the dual graph
        for comp in s_components(h1, s, start, end, edge=True):
            yield {translate[eid] for eid in comp}


def is_s_path(h: ASH, walk: List[str]) -> bool:
    """
    Validate that a hyperedge sequence is a simple s-path:
    1. No hyperedge repeats.
    2. No node in the original hypergraph appears in more than one step beyond
       adjacent pairs (ensures path simplicity in nodes).

    :param h:    ASH instance.
    :param walk: List of hyperedge IDs forming the candidate path.
    :return:     True if valid s-path, False otherwise.
    """
    # Check for repeated hyperedges
    if any(count > 1 for count in Counter(walk).values()):
        return False

    # Collect all node overlaps between any two hyperedges in the walk
    node_counts: Counter = Counter()
    for u, v in combinations(walk, 2):
        nodes_u = set(h.get_hyperedge_nodes(u))
        nodes_v = set(h.get_hyperedge_nodes(v))
        node_counts.update(nodes_u & nodes_v)

    # If any node appears in more than one overlap, path is not simple
    if any(count > 1 for count in node_counts.values()):
        return False
    return True
