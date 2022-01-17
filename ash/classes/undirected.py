from halp.undirected_hypergraph import UndirectedHypergraph
from collections import defaultdict


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

    ## Nodes

    def add_node(
        self, node: int, start: int, end: int = None, attr_dict: dict = None
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

        for i in range(start[0], start[1] + 1):
            for key, v in attr_dict.items():
                if key in old_attrs and key != "t":
                    old_attrs[key][i] = v
                else:
                    old_attrs[key] = {i: v}

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

    def get_node_attributes(self, node: int, tid: int = None) -> dict:
        """

        :param node:
        :param tid:
        :return:
        """
        if tid is None:
            return self.H.get_node_attributes(node)
        else:
            res = {}
            attrs = self.H.get_node_attributes(node)
            for key, l in attrs.items():
                if key != "t":
                    res[key] = l[tid]
        return res

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
            return self.H.get_node_attribute(node, attribute_name)
        else:
            attrs = self.H.get_node_attribute(node, attribute_name)
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

    def get_star(self, node: int, tid: int = None) -> list:
        """

        :param node:
        :param tid:
        :return:
        """
        if tid is None:
            return self.H.get_star(node)
        else:
            return [
                eid for eid in self.H.get_star(node) if self.has_hyperedge_id(eid, tid)
            ]

    def get_number_of_neighbors(
        self, node: int, hyperedge_size: int = None, tid: int = None
    ) -> int:
        """

        :param node:
        :param hyperedge_size:
        :param tid:
        :return:
        """
        star = self.get_star(node, tid)
        res = []
        if hyperedge_size is not None:
            for s in star:
                nodes = self.get_hyperedge_nodes(s)
                if len(nodes) == hyperedge_size:
                    res.extend(nodes)
            return len(set(res))
        else:
            for s in star:
                nodes = self.get_hyperedge_nodes(s)
                res.extend(nodes)
            return len(set(res))

    def get_degree(self, node: int, hyperedge_size: int = None, tid: int = None) -> int:
        """

        :param node:
        :param hyperedge_size:
        :param tid:
        :return:
        """
        star = self.get_star(node, tid)

        if hyperedge_size is not None:
            res = 0
            for s in star:
                nodes = self.get_hyperedge_nodes(s)
                if len(nodes) == hyperedge_size:
                    res += 1
            return res
        else:
            return len(star)

    def has_node(self, node: int, tid: int = None) -> bool:
        """

        :param node:
        :param tid:
        :return:
        """
        presence = self.H.has_node(node)
        if presence and tid is not None:
            attrs = self.get_node_attributes(node)["t"]
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

    ## Edges

    def add_hyperedge(self, nodes: list, start: int, end: int = None) -> None:
        """

        :param nodes:
        :param start:
        :param end:
        :return:
        """
        if start is None:
            raise ValueError("The hyperedge appearance time, t, cannot be None")
        if end is not None and end <= start:
            raise ValueError(
                "The vanishing time, e, (if present) must be higher of the appearance one"
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

    def get_hyperedge_id_set(self, tid: int = None) -> list:
        """

        :param tid:
        :return:
        """

        if tid is None:
            return self.H.get_hyperedge_id_set()

        else:
            hedges = {}
            for i in range(min(self.time_to_edge.keys()), tid + 1):
                hest = self.time_to_edge[i]

                for key, v in hest.items():
                    if v == "+":
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
        return len(self.get_hyperedge_id_set(tid))

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
                        S.add_hyperedge(he, span[0], span[1])

        for n in self.get_node_set():
            attrs = self.get_node_attributes(n)
            t1 = attrs["t"]
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
