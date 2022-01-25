import numpy as np
import networkx as nx
import json
from halp.undirected_hypergraph import UndirectedHypergraph
from collections import defaultdict
from itertools import combinations
from .node_profile import NProfile


class ASH(object):
    def __init__(self, hedge_removal: bool = False) -> None:
        """

        :param hedge_removal:
        """
        self.H = UndirectedHypergraph()
        self.time_to_edge = defaultdict(lambda: defaultdict(str))
        self.snapshots = {}
        self.hedge_removal = hedge_removal

    def __recursive_merge(self, inter: list, start_index: int = 0) -> list:
        """

        :param inter:
        :param start_index:
        :return:
        """
        for i in range(start_index, len(inter) - 1):
            if inter[i][1] > inter[i + 1][0]:
                new_start = inter[i][0]
                new_end = inter[i + 1][1]
                inter[i] = [new_start, new_end]
                del inter[i + 1]
                return self.__recursive_merge(inter.copy(), start_index=i)
        return inter

    def temporal_snapshots_ids(self) -> list:
        """

        :return:
        """
        return sorted(self.snapshots.keys())

    def stream_interactions(self) -> list:
        """

        :return:
        """
        keys = sorted(self.time_to_edge.keys())
        for tid in keys:
            for he, op in self.time_to_edge[tid].items():
                yield tid, he, op

    ## Nodes

    def avg_number_of_nodes(self) -> float:
        """

        :return:
        """
        nodes_snapshots = [
            self.get_number_of_nodes(tid) for tid in self.snapshots.keys()
        ]
        return sum(nodes_snapshots) / len(self.snapshots.keys())

    def add_node(
        self, node: int, start: int, end: int = None, attr_dict: object = None
    ) -> None:
        """

        :param node:
        :param start:
        :param end:
        :param attr_dict:
        :return:
        """
        if end is None:
            start = [start, start]
        else:
            start = [start, end]

        if not self.H.has_node(node):
            old_attrs = {"t": [start]}
        else:
            old_attrs = self.H.get_node_attributes(node)
            if "t" in old_attrs:
                old_attrs["t"].append(start)
            else:
                old_attrs["t"] = [start]

        head = None
        for i in range(start[0], start[1] + 1):
            if attr_dict is None:
                continue
            for key, v in attr_dict.items():
                if key in old_attrs and key != "t":
                    if head is not None:
                        old_attrs[key][i] = head
                    else:
                        old_attrs[key][i] = v
                else:
                    old_attrs[key] = {i: v}
                    head = f"t_{i}"

        # compacting intervals
        intervals = []
        presence = sorted(old_attrs["t"])
        for i, span in enumerate(presence):
            if i < len(presence) - 1:
                if span[1] == presence[i + 1][0] - 1:
                    intervals.append([span[0], presence[i + 1][1]])
                else:
                    intervals.append(span)
            else:
                intervals.append(span)

        merged = self.__recursive_merge(intervals.copy())
        old_attrs["t"] = merged

        self.H.add_node(node, old_attrs)
        if start[0] not in self.snapshots:
            self.snapshots[start[0]] = []
        if end is not None and end not in self.snapshots:
            self.snapshots[end] = []

    def add_nodes(
        self, nodes: int, start: int, end: int = None, node_attr_dict: dict = None
    ) -> None:
        """

        :param nodes:
        :param start:
        :param end:
        :param node_attr_dict:
        :return:
        """
        for node in nodes:
            attr = None if node not in node_attr_dict else node_attr_dict[node]
            self.add_node(node, start, end, attr)

    def get_node_profile(self, node: int, tid: int = None) -> NProfile:
        """

        :param node:
        :param tid:
        :return:
        """
        if tid is None:
            attrs = self.H.get_node_attributes(node)
            for key, l in attrs.items():
                if key != "t":
                    for tid, value in l.items():
                        if isinstance(value, str) and "t_" in value:
                            base_tid = int(value[2:])
                            attrs[key][tid] = l[base_tid]

            return NProfile(node, **attrs)
        else:
            res = {}
            attrs = self.H.get_node_attributes(node)
            for key, l in attrs.items():
                if key != "t":
                    res[key] = l[tid]
                    if isinstance(l[tid], str) and "t_" in l[tid]:
                        base_tid = int(l[tid][2:])
                        res[key] = l[base_tid]
        return NProfile(node, **res)

    def get_node_attribute(
        self, node: int, attribute_name: str, tid: int = None
    ) -> object:
        """

        :param node:
        :param attribute_name:
        :param tid:
        :return:
        """
        if tid is None:
            attrs = self.H.get_node_attribute(node, attribute_name)
            if attribute_name == "t":
                return {"t": attrs}
            for tid, value in attrs.items():
                if "t_" in value:
                    base_tid = int(value[2:])
                    attrs[tid] = attrs[base_tid]
            return attrs

        else:
            attrs = self.H.get_node_attribute(node, attribute_name)
            if attribute_name == "t":
                for spans in attrs:
                    for span in spans:
                        if span[0] <= tid <= span[1] + 1:
                            return {"t": [attrs]}
                return {"t": []}
            value = attrs[tid]
            if "t_" in value:
                return attrs[int(value[2:])]
            return attrs[tid]

    def get_node_set(self, tid: int = None) -> list:
        """

        :param tid:
        :return:
        """
        if tid is None:
            return self.H.get_node_set()
        else:
            return {n for n in self.H.get_node_set() if self.has_node(n, tid)}

    def get_number_of_nodes(self, tid: int = None) -> int:
        if tid is None:
            return len(self.get_node_set())
        else:
            return len(self.get_node_set(tid))

    def get_star(self, node: int, hyperedge_size: int = None, tid: int = None) -> set:
        """

        :param hyperedge_size:
        :param node:
        :param tid:
        :return:
        """
        if tid is None:
            eids = self.H.get_star(node)
            if hyperedge_size is None:
                return eids
            else:
                return {
                    eid
                    for eid in eids
                    if len(self.get_hyperedge_nodes(eid)) == hyperedge_size
                }
        else:
            if hyperedge_size is None:
                return {
                    eid
                    for eid in self.H.get_star(node)
                    if self.has_hyperedge_id(eid, tid)
                }
            else:
                return {
                    eid
                    for eid in self.H.get_star(node)
                    if self.has_hyperedge_id(eid, tid)
                    and len(self.get_hyperedge_nodes(eid)) == hyperedge_size
                }

    def get_number_of_neighbors(
        self, node: int, hyperedge_size: int = None, tid: int = None
    ) -> int:
        """

        :param node:
        :param hyperedge_size:
        :param tid:
        :return:
        """
        neighbors = self.get_neighbors(node, hyperedge_size, tid)
        return len(neighbors)

    def get_neighbors(
        self, node: int, hyperedge_size: int = None, tid: int = None
    ) -> set:
        """

        :param node:
        :param hyperedge_size:
        :param tid:
        :return:
        """
        star = self.get_star(node, tid=tid)
        res = []
        if hyperedge_size is not None:
            for s in star:
                nodes = self.get_hyperedge_nodes(s)
                if len(nodes) == hyperedge_size:
                    res.extend(nodes)
        else:
            for s in star:
                nodes = self.get_hyperedge_nodes(s)
                res.extend(nodes)

        return set(res)

    def get_degree(self, node: int, hyperedge_size: int = None, tid: int = None) -> int:
        """

        :param node:
        :param hyperedge_size:
        :param tid:
        :return:
        """
        star = self.get_star(node, tid=tid)

        if hyperedge_size is not None:
            res = 0
            for s in star:
                nodes = self.get_hyperedge_nodes(s)
                if len(nodes) == hyperedge_size:
                    res += 1
            return res
        else:
            return len(star)

    def get_degree_by_hyperedge_size(self, node: int, tid: int = None) -> dict:
        """

        :param node:
        :param tid:
        :return:
        """

        distr = defaultdict(int)
        star = self.get_star(node, tid=tid)
        for s in star:
            nodes = self.get_hyperedge_nodes(s)
            distr[len(nodes)] += 1
        return distr

    def has_node(self, node: int, tid: int = None) -> bool:
        """

        :param node:
        :param tid:
        :return:
        """
        presence = self.H.has_node(node)
        if presence and tid is not None:
            attrs = self.get_node_profile(node)
            if isinstance(attrs, NProfile):
                attrs = attrs.get_attribute("t")
            else:
                attrs = attrs["t"]
            for span in attrs:
                if span[0] <= tid <= span[1]:
                    return True
            return False
        return presence

    def node_iterator(self, tid: int = None) -> list:
        """

        :param tid:
        :return:
        """
        if tid is None:
            return self.H.node_iterator()
        S = self.hypergraph_temporal_slice(tid)
        return S.H.node_iterator()

    def get_node_presence(self, node: int) -> list:
        """

        :param node:
        :return:
        """
        snaps = []
        tids = self.get_node_attribute(node, "t")
        for spans in tids.values():
            for span in spans:
                snaps.extend(range(span[0], span[1] + 1))
        snaps = sorted(list(set(snaps)))
        return snaps

    def coverage(self) -> float:
        """

        :return:
        """
        T = len(self.snapshots)
        V = self.get_number_of_nodes()
        W = 0
        for tid in self.snapshots:
            W += self.get_number_of_nodes(tid)

        return W / (T * V)

    def node_contribution(self, node) -> float:
        """

        :param node:
        :return:
        """
        ucov = 0
        for tid in self.snapshots:
            ucov += 1 if self.has_node(node, tid) else 0
        return ucov / len(self.snapshots)

    def node_degree_distribution(self, start: int = None, end: int = None) -> dict:
        """

        :param start:
        :param end:
        :return:
        """
        dist = defaultdict(int)

        if start is None:
            for node in self.node_iterator():
                dist[self.get_degree(node)] += 1

        elif start is not None and end is None:
            for node in self.node_iterator(tid=start):
                dist[self.get_degree(node, tid=start)] += 1

        else:
            S = self.hypergraph_temporal_slice(start=start, end=end)
            for node in S.node_iterator():
                dist[S.get_degree(node)] += 1

        return dist

    ## Hyperedges

    def add_hyperedge(self, nodes: list, start: int, end: int = None) -> None:
        """

        :param nodes:
        :param start:
        :param end:
        :return:
        """
        if start is None:
            raise ValueError("The hyperedge appearance time, t, cannot be None")
        if end is not None and end < start:
            raise ValueError(
                "The vanishing time, e, (if present) must be equal or greater than the appearance one."
            )

        for u in nodes:
            if not self.H.has_node(u):
                self.add_node(u, start, end, {})

        if end is None:
            start = [start, start]
        else:
            start = [start, end]

        # add the interaction
        if not self.H.has_hyperedge(nodes):  # new hyperedge

            presence = {"t": [start]}  # : attr_dict}}
            self.H.add_hyperedge(nodes, attr_dict=presence)

        else:  # update existing one
            eid = self.H.get_hyperedge_id(nodes)
            old_attr = self.H.get_hyperedge_attributes(eid)
            presence = old_attr["t"]
            presence.append(start)
            presence = sorted(presence)

            # compacting intervals
            intervals = []
            for i, span in enumerate(presence):
                if i < len(presence) - 1:
                    if span[1] == presence[i + 1][0] - 1:
                        intervals.append([span[0], presence[i + 1][1]])
                    else:
                        intervals.append(span)
                else:
                    intervals.append(span)

            merged = self.__recursive_merge(intervals.copy())
            old_attr["t"] = merged
            old_attr["weight"] = len(merged)
            self.H.add_hyperedge(nodes, old_attr)

        # lookup table (to do)
        eid = self.H.get_hyperedge_id(nodes)
        intervals = self.H.get_hyperedge_attributes(eid)["t"]
        for span in intervals:
            for i in range(span[0], span[1] + 1):
                if eid in self.time_to_edge[i]:
                    del self.time_to_edge[i][eid]
            self.time_to_edge[span[0]][eid] = "+"
            if self.hedge_removal:
                self.time_to_edge[span[1] + 1][eid] = "-"

        for x in range(start[0], start[1] + 1):
            if x not in self.snapshots:
                self.snapshots[x] = [eid]
            else:
                self.snapshots[x].append(eid)
                self.snapshots[x] = list(set(self.snapshots[x]))

    def add_hyperedges(self, hyperedges: list, start: int, end: int = None) -> None:
        """

        :param hyperedges:
        :param start:
        :param end:
        :return:
        """
        for nodes in hyperedges:
            self.add_hyperedge(nodes, start, end)

    def get_hyperedge_attribute(
        self, hyperedge_id: str, attribute_name: str, tid: int = None
    ) -> object:
        """

        :param hyperedge_id:
        :param attribute_name:
        :param tid:
        :return:
        """
        # to check against attributes (not implemented)
        attrs = self.H.get_hyperedge_attribute(hyperedge_id, attribute_name)

        if tid is not None:
            res = {}
            for key, v in attrs.items():
                if key != "t" and key == attribute_name:
                    res[key] = v
                else:
                    for span in v:
                        if span[0] <= tid <= span[1]:
                            res["t"] = [span]
            if "t" in res:
                return res
            else:
                raise ValueError("Hyperedge not present in the selected timeslot")
        else:
            return attrs

    def get_hyperedge_attributes(self, hyperedge_id: str, tid: int = None) -> dict:
        """

        :param hyperedge_id:
        :param tid:
        :return:
        """
        attrs = self.H.get_hyperedge_attributes(hyperedge_id)
        if tid is not None:
            res = {}
            for key, v in attrs.items():
                if key != "t":
                    res[key] = v
                else:
                    for span in v:
                        if span[0] <= tid <= span[1]:
                            res["t"] = [span]
            if "t" in res:
                return res
            else:
                raise ValueError("Hyperedge not present in the selected timeslot")
        else:
            return attrs

    def get_hyperedge_id(self, nodes: list) -> str:
        """

        :param nodes:
        :return:
        """
        return self.H.get_hyperedge_id(nodes)

    def get_hyperedge_id_set(self, hyperedge_size: int = None, tid: int = None) -> list:
        """

        :param hyperedge_size:
        :param tid:
        :return:
        """

        if tid is None:
            if hyperedge_size is None:
                return self.H.get_hyperedge_id_set()
            else:
                return {
                    he
                    for he in self.H.get_hyperedge_id_set()
                    if len(self.get_hyperedge_nodes(he)) == hyperedge_size
                }
        else:
            hedges = {}
            for i in range(min(self.time_to_edge.keys()), tid + 1):
                hest = self.time_to_edge[i]
                for key, v in hest.items():
                    if v == "+" and (
                        hyperedge_size is None
                        or len(self.get_hyperedge_nodes(key)) == hyperedge_size
                    ):
                        hedges[key] = None
                    else:
                        if key in hedges:
                            del hedges[key]
            return set(hedges.keys())

    def get_hyperedge_nodes(self, hyperedge_id: str) -> list:
        """

        :param hyperedge_id:
        :return:
        """
        return self.H.get_hyperedge_nodes(hyperedge_id)

    def get_hyperedge_weight(self, hyperedge_id: str) -> int:
        """

        :param hyperedge_id:
        :return:
        """
        return self.H.get_hyperedge_weight(hyperedge_id)

    def has_hyperedge(self, nodes: list, tid: int = None) -> bool:
        """

        :param nodes:
        :param tid:
        :return:
        """
        presence = self.H.has_hyperedge(nodes)
        if presence and tid is not None:
            eid = self.get_hyperedge_id(nodes)
            return self.has_hyperedge_id(eid, tid)
        return presence

    def has_hyperedge_id(self, hyperedge_id: str, tid: int = None) -> bool:
        """

        :param hyperedge_id:
        :param tid:
        :return:
        """
        presence = self.H.has_hyperedge_id(hyperedge_id)
        if presence and tid is not None:
            attrs = self.get_hyperedge_attributes(hyperedge_id)["t"]
            for span in attrs:
                if span[0] <= tid <= span[1]:
                    return True
            return False
        return presence

    def hyperedge_id_iterator(self, start: int = None, end: int = None) -> list:
        """

        :param start:
        :param end:
        :return:
        """
        if start is None:
            return self.H.hyperedge_id_iterator()
        S = self.hypergraph_temporal_slice(start, end)
        return S.H.hyperedge_id_iterator()

    def get_size(self, tid: int = None) -> int:
        """

        :param tid:
        :return:
        """
        return len(self.get_hyperedge_id_set(tid=tid))

    def get_avg_number_of_hyperedges(self) -> float:
        """

        :return:
        """
        return sum([len(he) for he in self.snapshots.values()]) / len(self.snapshots)

    def hyperedge_contribution(self, hyperedge_id: str) -> float:
        attrs = self.get_hyperedge_attributes(hyperedge_id)["t"]
        count = 0
        for span in attrs:
            count += len(range(span[0], span[1] + 1))
        return count / len(self.snapshots)

    # Slices

    def hypergraph_temporal_slice(self, start: int, end: int = None) -> object:
        """

        :param start:
        :param end:
        :return:
        """
        if end is None and start in self.snapshots:
            edges = self.snapshots[start]
        elif end is None and start not in self.snapshots:
            edges = []
        else:
            edges = [
                e1
                for obs in range(start, end + 1)
                for e1 in self.snapshots.get(obs, [])
            ]

        S = ASH()
        for e1 in edges:
            he = self.get_hyperedge_nodes(e1)
            e_attrs = self.get_hyperedge_attributes(e1)
            t1 = e_attrs["t"]
            for span in t1:
                if end is not None:
                    if span[0] >= start and span[1] <= end:
                        S.add_hyperedge(he, span[0], span[1])
                    elif span[0] >= start and span[1] >= end:
                        S.add_hyperedge(he, span[0], end)
                else:
                    if span[0] <= start <= span[1]:
                        if span[0] != span[1]:
                            S.add_hyperedge(he, span[0], span[1])
                        else:
                            S.add_hyperedge(he, span[0])

        for n in self.get_node_set():
            attrs = self.get_node_profile(n)
            if not isinstance(attrs, NProfile):
                t1 = attrs["t"]
            else:
                t1 = attrs.get_attribute("t")

            for span in t1:
                if end is not None:
                    if span[0] >= start and span[1] <= end:
                        S.add_node(n, span[0], span[1], attr_dict=attrs)
                    if span[0] >= start and span[1] >= end:
                        S.add_node(n, span[0], end, attr_dict=attrs)
                else:
                    if span[0] <= start <= span[1]:
                        S.add_node(n, span[0], span[1], attr_dict=attrs)
        return S

    def uniformity(self) -> float:
        """
            Temporal hypergraph uniformity

            Returns
            -------
            uniformity : float
                Uniformity of the temporal hypergraph in [0, 1]
        """
        nds = self.get_node_set()
        numerator, denominator = 0, 0
        for u, v in combinations(nds, 2):
            for t in self.snapshots:
                if self.has_node(u, t) and self.has_node(v, t):
                    numerator += 1
                if self.has_node(u, t) or self.has_node(v, t):
                    denominator += 1
        return numerator / denominator

    def hyperedge_size_distribution(self, start: int = None, end: int = None) -> dict:
        """

        :param start:
        :param end:
        :return:
        """
        dist = defaultdict(int)

        if start is None:
            for he in self.get_hyperedge_id_set():
                dist[len(self.get_hyperedge_nodes(he))] += 1

        elif start is not None and end is None:
            for he in self.get_hyperedge_id_set(tid=start):
                dist[len(self.get_hyperedge_nodes(he))] += 1

        else:
            for tids in range(start, end + 1):
                for he in self.get_hyperedge_id_set(tid=tids):
                    dist[len(self.get_hyperedge_nodes(he))] += 1

        return dist

    def __str__(self) -> str:
        """

        :return:
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> dict:
        """

        :return:
        """
        descr = {"nodes": {}, "hedges": {}}

        for hedge in self.hyperedge_id_iterator():
            e = self.get_hyperedge_attributes(hedge)
            descr["hedges"][hedge] = e

        for node in self.node_iterator():
            npr = self.get_node_profile(node)
            descr["nodes"][node] = npr.get_attributes()
        return descr

    # Transform

    def line_graph(self, start: int = None, end: int = None) -> object:
        """

        :param start:
        :param end:
        :return:
        """
        node_to_edges = defaultdict(list)
        for he in self.hyperedge_id_iterator(start=start, end=end):
            nodes = self.get_hyperedge_nodes(he)
            for node in nodes:
                node_to_edges[node].append(he)

        g = nx.Graph()
        for edges in node_to_edges.values():
            if len(edges) > 0:
                for e in combinations(edges, 2):
                    if not g.has_edge(e[0], e[1]):
                        g.add_edge(e[0], e[1])

        return g

    def bipartite_projection(self, start: int = None, end: int = None) -> object:
        """

        :param start:
        :param end:
        :return:
        """

        g = nx.Graph()
        for he in self.hyperedge_id_iterator(start=start, end=end):
            g.add_node(he, bipartite=1)
            nodes = self.get_hyperedge_nodes(he)
            for node in nodes:
                if not g.has_node(node):
                    g.add_node(node, bipartite=0)
                g.add_edge(node, he)

        return g

    def dual_hypergraph(self, start: int = None, end: int = None) -> object:
        """

        :param start:
        :param end:
        :return:
        """
        b = ASH(hedge_removal=True)
        node_to_edges = defaultdict(list)
        for he in self.hyperedge_id_iterator(start=start, end=end):
            nodes = self.get_hyperedge_nodes(he)
            for node in nodes:
                node_to_edges[node].append(he)

        for edges in node_to_edges.values():
            b.add_hyperedge(edges, 0)

        return b

    def adjacency(self, node_set: set, start: int = None, end: int = None) -> int:
        """

        :param node_set:
        :param start:
        :param end:
        :return:
        """
        count = 0
        for he in self.hyperedge_id_iterator(start=start, end=end):
            nodes = self.get_hyperedge_nodes(he)
            inc = set(nodes) & set(node_set)
            if len(inc) == len(node_set):
                count += 1
        return count

    def incidence(self, edge_set: set, start: int = None, end: int = None) -> int:
        """

        :param edge_set:
        :param start:
        :param end:
        :return:
        """

        first = True
        if end is None:
            end = start

        res = set()
        for he in edge_set:
            nodes = self.get_hyperedge_nodes(he)
            filtered_nodes = []
            if start is None:
                for node in nodes:
                    if self.has_node(node):
                        filtered_nodes.append(node)
                filtered_nodes = list(set(filtered_nodes))
            else:
                for tid in range(start, end+1):
                    for node in nodes:
                        if self.has_node(node, tid):
                            filtered_nodes.append(node)
                    filtered_nodes = list(set(filtered_nodes))

            if first:
                first = False
                res = set(filtered_nodes)
            else:
                res = set(filtered_nodes) & res

        return len(res)
