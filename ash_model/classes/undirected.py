import json
from collections import defaultdict
from itertools import combinations

import networkx as nx
from halp.undirected_hypergraph import UndirectedHypergraph

from .node_profile import NProfile


class ASH(object):
    """
    Class for representing the Attributed Stream Hypergraph.

    :param hedge_removal: whether to allow for hyperedge removal or not
    """

    def __init__(self, hedge_removal: bool = False) -> None:
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
        Returns the list of temporal snapshots ids for the ASH, i.e.,
        integers representing points in time.

        :return: list of temporal ids
        """
        return sorted(self.snapshots.keys())

    def stream_interactions(self) -> list:
        """
        The stream_interactions function yields a list of tuples,
        where each tuple contains three elements identifying interaction appearance/disappearance.
        The first element is a temporal snapshot id. The second is a hyperedege id. The third
        is either '+' (if the hyperedge appears) or '-' (if the hyperedge disappears).

        :return: A generator that yields the time, hyperedge_id, and appearance/disappearance
        for each interaction in the stream
        """

        keys = sorted(self.time_to_edge.keys())
        for tid in keys:
            for he, op in self.time_to_edge[tid].items():
                yield tid, he, op

    ## Nodes

    def avg_number_of_nodes(self) -> float:
        """
        The avg_number_of_nodes function returns the average number of nodes in an ASH over time, i.e., the
        sum of each snapshot's nodes divided by the number of snapshots.

        :return: The average number of nodes in the ASH over all snapshots
        """

        nodes_snapshots = [
            self.get_number_of_nodes(tid) for tid in self.snapshots.keys()
        ]
        return sum(nodes_snapshots) / len(self.snapshots.keys())

    def add_node(
        self, node: int, start: int, end: int = None, attr_dict: object = None
    ) -> None:
        """
        The add_node function adds a node to the ASh.
        It takes as input:
        -node, an integer representing the node to be added;
        -start, an integer representing the start of the interval in which this node is active;
        -end, an optional argument that represents the end of interval in which this node is active
        (if not specified it defaults to start); and
        -attr_dict, a dictionary containing attributes associated with this particular node.

        :param node: Identify the node
        :param start: Specify the starting snapshot of the node
        :param end: Specify the end of the interval
        :param attr_dict: dictionary of attributes to be added to the node's profile
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

        # contiguity
        cont = [merged[0]]
        pos = 0
        for i in range(1, len(merged)):
            if cont[pos][1] == merged[i][0]:
                cont[pos][1] = merged[i][1]
            else:
                pos += 1
                cont.append(merged[i])

        old_attrs["t"] = cont

        self.H.add_node(node, old_attrs)
        if start[0] not in self.snapshots:
            self.snapshots[start[0]] = []
        if end is not None and end not in self.snapshots:
            self.snapshots[end] = []

    def add_nodes(
        self, nodes: list, start: int, end: int = None, node_attr_dict: dict = None
    ) -> None:
        """
        The add_nodes function adds a list of nodes to the ASH with an optional node-to-attributes
        dictionary. All nodes will be assumed to be active in the same time window, defined by :start: and :end:

        :param nodes: Specify the nodes to be added
        :param start: Specify the appearance time of the nodes
        :param end: Specify the disappearance time of the nodes
        :param node_attr_dict: Dictionary of attributes to be added to the nodes.
        :return: None
        """

        for node in nodes:
            attr = None if node not in node_attr_dict else node_attr_dict[node]
            self.add_node(node, start, end, attr)

    def get_node_profile(self, node: int, tid: int = None) -> NProfile:
        """
        The get_node_profile function returns a copy of the NProfile object associated with the given node.
        The optional parameter tid specifies the temporal snapshot the profile refers to.

        :param node: Specify the node to get the profile of
        :param tid: Get the profile of a node at a specific time
        :return: A NProfile object
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
                if key != "t" and tid in l:
                    res[key] = l[tid]
                    if isinstance(l[tid], str) and "t_" in l[tid]:
                        base_tid = int(l[tid][2:])
                        res[key] = l[base_tid]
        return NProfile(node, **res)

    def get_node_attribute(self, node: int, attr_name: str, tid: int = None) -> object:
        """
        The get_node_attribute function returns the value of a node attribute for a given node.
        If no temporal snapshot id (tid) is specified, it will return a tid-to-attribute-value
        dictionary for that node.
        If a tid is specified, it will return the value of the attribute at that tid.

        :param node: Specify which node to get the attribute of
        :param attr_name: Specify the attribute name
        :param tid: Get the attribute for all time steps, or for a specific time step
        :return: The attribute of the node. A dict if tid is specified, the attribute value otherwise
        """

        if tid is None:
            attrs = self.H.get_node_attribute(node, attr_name)
            if attr_name == "t":
                return {"t": attrs}
            for tid, value in attrs.items():
                if "t_" in value:
                    base_tid = int(value[2:])
                    attrs[tid] = attrs[base_tid]
            return attrs

        else:
            attrs = self.H.get_node_attribute(node, attr_name)
            if attr_name == "t":
                for spans in attrs:
                    for span in spans:
                        if span[0] <= tid <= span[1] + 1:
                            return {"t": [attrs]}
                return {"t": []}
            value = attrs[tid]
            if isinstance(value, str) and "t_" in value:
                return attrs[int(value[2:])]
            return attrs[tid]

    def node_attributes_to_attribute_values(
        self, categorical=False, tid: int = None
    ) -> dict:
        """
        The node_attributes_to_attribute_values function returns a dictionary of the attributes and their values. The
        function takes in two parameters: categorical, which is a boolean that determines whether to include
        numerical attributes, and tid, which is an integer that represents the temporal id. The default value for
        categorical is False and the default value for tid is None.

        :param self: Bind the method to the object
        :param categorical: Specify whether the attributes are categorical or not
        :param tid: Specify the temporal id
        :return: A dictionary of attribute names and the values they can take
        """

        attributes = defaultdict(set)
        for n in self.get_node_set(tid=tid):
            for name, vals in self.get_node_profile(n, tid=tid).items():
                if name != "t":
                    if tid is None:
                        for value in vals.values():
                            attributes[name].add(value)
                    else:
                        attributes[name].add(vals)
        if categorical:
            numerical = [
                attribute
                for attribute in attributes
                if not isinstance(list(attributes[attribute])[0], str)
            ]
            for attribute in numerical:
                del attributes[attribute]

        return attributes

    def get_node_set(self, tid: int = None) -> list:
        """
        The get_node_set function returns the set of all the nodes in the ASH.
        If a snapshot id (tid) is specified, it will return only those nodes that appear in that snapshot.

        :param tid: Optional temporal snapshot id
        :return: The set of nodes in the ASH
        """

        if tid is None:
            return self.H.get_node_set()
        else:
            return {n for n in self.H.get_node_set() if self.has_node(n, tid)}

    def get_number_of_nodes(self, tid: int = None) -> int:
        """
        The get_number_of_nodes function returns the number of nodes in the ASH.
        If no temporal snapshot id (tid) is specified, it will return the total number of nodes.
        Otherwise, it will return the total number of nodes that are active in tid.

        :param tid: Optional temporal snapshot id
        :return: The number of nodes in the ASH
        """

        if tid is None:
            return len(self.get_node_set())
        else:
            return len(self.get_node_set(tid))

    def get_star(self, node: int, hyperedge_size: int = None, tid: int = None) -> set:
        """
        The get_star function returns the set of hyperedge ids that include a given node.
        If no hyperedge size is specified, it returns all of the hyperedges that include a given node.
        Otherwise, it will return only those with exactly that many nodes.
        If temporal snapshot id (tid) is specified, it restricts the star to the hyperedges that are active in tid.

        :param node: Specify the node whose star is to be returned
        :param hyperedge_size: Optionally restrict star to hyperedges with specific size
        :param tid: Optionally restrict star to hyperedges active at a point in time
        :return: The set of hyperedge ids that include a given node
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
        The get_number_of_neighbors function returns the number of nodes a node is connected to via hyperedges.
        If the optional argument hyperedge_size is provided, then only neighbors belonging to hyperedges
        of that size will be counted.
        The optional argument tid restricts the counting to the nodes connected in a specific temporal snapshot.

        :param node: Specify the node whose neighbors are to be counted
        :param hyperedge_size: Specify the exact size of the hyperedges to count the nodes of
        :param tid:: Specify the temporal snapshot
        :return: The number of neighbors for a given node
        """
        neighbors = self.get_neighbors(node, hyperedge_size, tid)
        return len(neighbors)

    def get_neighbors(
        self, node: int, hyperedge_size: int = None, tid: int = None
    ) -> set:
        """
        The get_neighbors function returns the set of all nodes that are connected to a given node via hyperedges.
        If the optional argument hyperedge_size is provided, then only neighbors belonging to hyperedges
        of that size will be returned.
        The optional argument tid restricts the set to the nodes connected in a specific temporal snapshot.

        :param node: Specify the node for which we want to get its neighbors
        :param hyperedge_size: Specify the size of the hyperedge to be returned
        :param tid: Specify a specific transaction id
        :return: The set of neighbors of a given node
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
        res = set(res)
        res.discard(node)
        return res

    def get_degree(self, node: int, hyperedge_size: int = None, tid: int = None) -> int:
        """
        The get_degree function returns the number of hyperedges that a given node is part of.
        If the optional argument hyperedge_size is specified, then it returns only the number
        of hyperedges that have exactly this many nodes.

        :param node: Specify the node id
        :param hyperedge_size: Specify the size of hyperedges to be counted
        :param tid: Get the degree at a specific point in time
        :return: The degree of a node
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
        Given a node, the get_degree_by_hyperedge_size function returns a dictionary where keys are the sizes
        of the hyperedges in the node's star, and values are the raw frequency of that size in the star.
        For example, if there are two hyperedges with three nodes each and one
        hyperedge with four nodes, then the returned dictionary would be: {3: 2, 4: 1}.
        The optional parameter tid restricts the hyperedges to those active in that point in time.

        :param node: Specify the node for which we want to calculate the degree distribution
        :param tid: Specify a temporal snapshot
        :return: A dictionary where the keys are the node's star's hyperedge sizes and the values are their frequencies
        """

        distr = defaultdict(int)
        star = self.get_star(node, tid=tid)
        for s in star:
            nodes = self.get_hyperedge_nodes(s)
            distr[len(nodes)] += 1
        return distr

    def get_s_degree(self, node: int, s: int, tid: int = None) -> int:
        """
        The get_s_degree function returns the s-degree of a specific node.


        :param node: Specify the node for which to compute the degree
        :param s: Specify the minimum number of nodes in each hyperedge to be counted
        :param tid: Temporal id
        :return: The number of hyperedgess in the start that have at least s nodes
        """

        degs = self.get_degree_by_hyperedge_size(node, tid)
        res = 0
        for k, v in degs.items():
            if k >= s:
                res += v
        return res

    def has_node(self, node: int, tid: int = None) -> bool:
        """
        The has_node function checks if a node is present in the ASH.
        Optionally, it is possible to check for presence at a specific point in time.

        :param node: The node whose presence is to be assessed
        :param tid: Check if a node is present in the ASH at a specific time
        :return: True if the node is present and False otherwise
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
        The node_iterator function returns an iterator over the nodes in the ASH.
        If no temporal id (tid) is specified, it will iterate over all nodes.
        Otherwise, it will iterate only over those nodes that exist at that temporal id.

        :param tid: Specify the temporal snapshot id
        :return: An iterator over the ASH's nodes
        """

        if tid is None:
            return self.H.node_iterator()
        S, _ = self.hypergraph_temporal_slice(tid)
        return S.H.node_iterator()

    def get_node_presence(self, node: int) -> list:
        """
        The get_node_presence function returns a list of all the snapshots in which the node is present.
        The function takes as input a node id and returns a list of all the snapshots in which that node is present.

        :param node: Specify the node for which we want to get the presence
        :return: A list of all the snapshots that a node is present in
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
        The coverage function is the fraction of nodes in the ASH that are
        present in at least one snapshot over all possible nodes and snapshots.
        This measure is used to quantify how well the ASH has been preserved over time.

        :return: The fraction of the nodes in the graph that are covered by at least one snapshot
        """

        T = len(self.snapshots)
        V = self.get_number_of_nodes()
        W = 0
        for tid in self.snapshots:
            W += self.get_number_of_nodes(tid)

        return W / (T * V)

    def node_contribution(self, node) -> float:
        """
        The node_contribution function returns the fraction of snapshots where the node is present.
        This is used to determine how important a node is within the ASH.

        :param node: Specify the node for which we want to calculate the contribution
        :return: The contribution of a node to the overall coverage
        """

        ucov = 0
        for tid in self.snapshots:
            ucov += 1 if self.has_node(node, tid) else 0
        return ucov / len(self.snapshots)

    def node_degree_distribution(self, start: int = None, end: int = None) -> dict:
        """
        The node_degree_distribution function returns a dictionary of the number of nodes with each degree.
        The function takes two optional arguments, start and end. If start is not specified, then it will return
        the distribution for all nodes in the ASH. If only start is specified, then it will return the
        distribution for all nodes from that point in time onwards. If both are specified, then it will return
        the distribution for all nodes in that temporal slice range (tid&gt;=start &amp; tid&lt;end).
        The function returns a dictionary where keys are degrees and values are counts.

        :param start: Starting point of optional time window
        :param end: Ending point of optional time window
        :return: A degree-to-frequency dictionary.
        """
        dist = defaultdict(int)

        if start is None:
            for node in self.node_iterator():
                dist[self.get_degree(node)] += 1

        elif start is not None and end is None:
            for node in self.node_iterator(tid=start):
                dist[self.get_degree(node, tid=start)] += 1

        else:
            S, old_to_new = self.hypergraph_temporal_slice(start=start, end=end)
            for node in S.node_iterator():
                dist[S.get_degree(node)] += 1

        return dist

    ## Hyperedges

    def add_hyperedge(self, nodes: list, start: int, end: int = None, **attrs) -> None:
        """
        The add_hyperedge function adds a hyperedge to the ASH, active between :start: and :end:.
        If the :end: parameter is not specified, it defaults to :start:

        :param nodes: Specify the nodes that are part of the hyperedge
        :param start: Indicate the appearance time of the hyperedge
        :param end: Indicate the vanishing time of the hyperedge
        :param **attrs: Pass additional information about the interaction
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
            for k, v in attrs.items():
                presence[k] = v

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

            # contiguity
            cont = [merged[0]]
            pos = 0
            for i in range(1, len(merged)):
                if cont[pos][1] == merged[i][0]:
                    cont[pos][1] = merged[i][1]
                else:
                    pos += 1
                    cont.append(merged[i])

            old_attr["t"] = cont

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
        The add_hyperedges function adds a list of hyperedges to the ASH, all with the same start and end
        snapshot ids.

        :param hyperedges: Specify a list of lists of nodes representing hyperedges
        :param start: Specify the starting temporal snapshot of the hyperedges
        :param end:  Specify the ending temporal snapshot of the hyperedges
        :return:
        """

        for nodes in hyperedges:
            self.add_hyperedge(nodes, start, end)

    def get_hyperedge_attribute(
        self, hyperedge_id: str, attribute_name: str, tid: int = None
    ) -> object:
        """
        The get_hyperedge_attribute function returns the value of a specific attribute of a hyperedge.

        :param hyperedge_id:str: Specify the hyperedge to get the attribute of
        :param attribute_name: Specify the attribute that is to be returned
        :param tid: Specify a time slot
        :return: The attribute of a hyperedge
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
        The get_hyperedge_attributes function returns a dictionary of the attributes associated with a hyperedge.
        If tid is specified, it will only return the attribute values for that timeslot.

        :param hyperedge_id: Specify the hyperedge to get the attributes of
        :param tid: Specify a snapshot
        :return: The attributes of a hyperedge
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
        The get_hyperedge_id function takes a list of nodes and returns the hyperedge ID
        of the hyperedge that contains those nodes.

        :param nodes: Specify the nodes that are in the hyperedge
        :return: The id of the hyperedge that is formed by the nodes in the list
        """

        return self.H.get_hyperedge_id(nodes)

    def get_hyperedge_id_set(self, hyperedge_size: int = None, tid: int = None) -> set:
        """
        The get_hyperedge_id_set function returns the set of hyperedge IDs that satisfy the given constraints. The
        function takes two optional arguments, hyperedge_size and tid. If no arguments are passed in, it will return
        all hyperedges id in the ASH. If only hyperedge_size is passed in, it will return all hyperedges with size
        equal to hyperedge_size. Finally, if both are specified, both constraint are respected.

        :param hyperedge_size: Specify the size of hyperedges to be included
        :param tid: Specify the snapshot of the hyperedges to consider
        :return: A set of hyperedge ids that satisfy the given conditions
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
        The get_hyperedge_nodes function returns a set of the nodes that are connected by the hyperedge with id
        hyperedge_id.

        :param hyperedge_id: Specify which hyperedge to get the nodes for
        :return: The nodes that are connected by the hyperedge_id
        """

        return self.H.get_hyperedge_nodes(hyperedge_id)

    def get_hyperedge_weight(self, hyperedge_id: str) -> int:
        """
        The get_hyperedge_weight function returns the weight of a hyperedge.

        :param hyperedge_id: Identify the hyperedge
        :return: The weight of the hyperedge with id hyperedge_id
        """

        return self.H.get_hyperedge_weight(hyperedge_id)

    def has_hyperedge(self, nodes: list, tid: int = None) -> bool:
        """
        The has_hyperedge function checks to see if the ASH contains an hyperedge
        formed by a specific set of nodes. Optionally, it checks at a specific snapshot (tid).

        :param nodes: Specify the nodes that are part of the hyperedge
        :param tid: To check whether the hyperedge exists in a specific snapshot
        :return: True if the hyperedge with the given nodes exists, False otherwise
        """

        presence = self.H.has_hyperedge(nodes)
        if presence and tid is not None:
            eid = self.get_hyperedge_id(nodes)
            return self.has_hyperedge_id(eid, tid)
        return presence

    def has_hyperedge_id(self, hyperedge_id: str, tid: int = None) -> bool:
        """
        The has_hyperedge_id function checks whether a hyperedge with id hyperedge_id is present in the ASH.
        Optionally, it checks at a specific snapshot id (tid).

        :param hyperedge_id: Specify the hyperedge to be checked
        :param tid: Optionally check if it exists at a specific snapshot
        :return: A boolean indicating whether the hyperedge is present in the given snapshot
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
        The hyperedge_id_iterator function returns a list of hyperedge IDs that are present in the ASH
        between two points in time. If no start or end is specified, it will return all hyperedge IDs.

        :param start: Indicate the start time of the temporal slice
        :param end: Specify the end of the temporal slice
        :return: A list of hyperedge ids that are in the temporal slice specified by start and end
        """

        if start is None:
            return self.H.hyperedge_id_iterator()
        S, eid_to_new_eid = self.hypergraph_temporal_slice(start, end)
        new_eid_to_old_eid = {v: k for k, v in eid_to_new_eid.items()}

        edges = list(S.H.hyperedge_id_iterator())
        return [new_eid_to_old_eid[e] for e in edges]

    def get_size(self, tid: int = None) -> int:
        """
        The get_size function returns the number of hyperedges in an ASH.
        If a temporal snapshot id (tid) is specified, it will count the hyperedges that are active in tid.

        :param tid: Specify the temporal snapshot id
        :return: The number of hyperedges in the ASH
        """

        return len(self.get_hyperedge_id_set(tid=tid))

    def get_avg_number_of_hyperedges(self) -> float:
        """
        The get_avg_number_of_hyperedges function returns the average number of
        hyperedges in each snapshot.

        :return: The average number of hyperedges per snapshot
        """

        return sum([len(he) for he in self.snapshots.values()]) / len(self.snapshots)

    def hyperedge_contribution(self, hyperedge_id: str) -> float:
        """
        The hyperedge_contribution function calculates the fraction of snapshots where
        a given hyperedge is active.

        :param hyperedge_id: Specify which hyperedge to calculate the contribution for
        :return: The contribution of a hyperedge
        """

        attrs = self.get_hyperedge_attributes(hyperedge_id)["t"]
        count = 0
        for span in attrs:
            count += len(range(span[0], span[1] + 1))
        return count / len(self.snapshots)

    # Slices

    def hypergraph_temporal_slice(self, start: int, end: int = None) -> tuple:
        """
        The hypergraph_temporal_slice constructs a new ASH instance that contains hyperedges active in the given
        time window. It returns both the new instance and a dictionary mapping old hyperedge ids to new hyperedge ids.
        If no end time is specified, then all temporal ids greater or equal to start are considered.

        :param start: Specify the start time of the temporal slice
        :param end: Specify the end of the temporal slice
        :return: an ASH instance and a dictionary mapping old hyperedge ids to new hyperedge ids
        """

        if end is None and start in self.snapshots:
            edges = set(
                [
                    e1
                    for obs in range(min(self.snapshots), max(self.snapshots) + 1)
                    for e1 in self.snapshots.get(obs, [])
                ]
            )

        elif end is None and start not in self.snapshots:
            edges = []
        else:
            edges = set(
                [
                    e1
                    for obs in range(min(self.snapshots), end + 1)
                    for e1 in self.snapshots.get(obs, [])
                ]
            )

        S = ASH()
        eid_to_new_eid = {}
        for e1 in edges:
            he = self.get_hyperedge_nodes(e1)
            e_attrs = self.get_hyperedge_attributes(e1)
            t1 = e_attrs["t"]

            for span in t1:
                if end is not None:
                    if span[0] >= start and span[1] <= end:
                        S.add_hyperedge(he, span[0], span[1])
                        new_eid = S.get_hyperedge_id(he)
                        eid_to_new_eid[e1] = new_eid

                    elif end >= span[0] >= start and span[1] >= end:
                        S.add_hyperedge(he, start=span[0], end=end)
                        new_eid = S.get_hyperedge_id(he)
                        eid_to_new_eid[e1] = new_eid

                    elif span[0] < start and span[1] >= end:
                        S.add_hyperedge(he, start, span[1])
                        new_eid = S.get_hyperedge_id(he)
                        eid_to_new_eid[e1] = new_eid

                    # else:
                    #    S.add_hyperedge(he, start, end)
                    #    new_eid = S.get_hyperedge_id(he)
                    #    eid_to_new_eid[e1] = new_eid

                else:
                    if span[0] >= start or start <= span[1]:
                        if span[0] != span[1]:
                            S.add_hyperedge(he, span[0], span[1])
                            new_eid = S.get_hyperedge_id(he)
                            eid_to_new_eid[e1] = new_eid

                        else:
                            S.add_hyperedge(he, span[0])
                            new_eid = S.get_hyperedge_id(he)
                            eid_to_new_eid[e1] = new_eid

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
                    elif end >= span[0] >= start and span[1] >= end:
                        S.add_node(n, span[0], end, attr_dict=attrs)
                    elif span[0] < start and span[1] >= end:
                        S.add_node(n, start, span[1], attr_dict=attrs)

                else:
                    if span[0] <= start <= span[1]:
                        if span[0] != span[1]:
                            S.add_node(n, span[0], span[1], attr_dict=attrs)

                        else:
                            S.add_node(n, span[0], attr_dict=attrs)

        return S, eid_to_new_eid

    def uniformity(self) -> float:
        """
        Temporal hypergraph uniformity

        :return: uniformity value for the hypergraph
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
        The hyperedge_size_distribution function returns a dictionary of the number of hyperedges with a given size.
        The function takes two optional arguments, start and end. If only start is provided, then it will return the
        distribution of hyperedge sizes starting from (and including) that time step. If both start and end are
        provided, then it will return the distribution between those two indices (inclusive). The default behavior is
        to return the distribution over all time steps.

        :param start: Specify the starting time of the temporal slice
        :param end: Specify the ending time of the temporal slice
        :return: A size_to_count dictionary that contains the number of hyperedges with a given size
        """

        dist = defaultdict(int)

        if start is None:
            for he in self.get_hyperedge_id_set():
                dist[len(self.get_hyperedge_nodes(he))] += 1

        elif start is not None and end is None:
            for he in self.get_hyperedge_id_set(tid=start):
                dist[len(self.get_hyperedge_nodes(he))] += 1

        else:
            for tid in range(start, end + 1):
                for he in self.get_hyperedge_id_set(tid=tid):
                    dist[len(self.get_hyperedge_nodes(he))] += 1

        return dist

    def __str__(self) -> str:
        """
        String representation of the ASH

        :return:
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> dict:
        """
        The to_dict function returns a dictionary representation of the ASH.
        The keys are &quot;nodes&quot; and &quot;hedges&quot;. The nodes key points to a dictionary,
        where the keys are node ids and values are dictionaries containing all attributes
        of that node. The hedges key points to another dictionary, where the keys are hyperedge ids
        and values contain dictionaries with all attributes of that hyperedge, including appearence/disappearence temporal ids..

        :return: A dictionary representation of the ASH
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

    def s_line_graph(self, s: int = 1, start: int = None, end: int = None) -> object:
        """
        The s_line_graph function returns the corresponding s-line graph. Each node in the s-line graph represents a
        hyperedge with at least s nodes. Two s-line graph nodes are linked if their corresponding hyperedges
        intersect in at least s nodes in the original hypergraph.

        :param s: Specify the minimum intersection between hyperedges
        :param start: Specify the start of a time window
        :param end: Specify the end of the interval
        :return: The s-line graph of the ASH
        """
        node_to_edges = defaultdict(list)
        for he in self.hyperedge_id_iterator(start=start, end=end):
            nodes = self.get_hyperedge_nodes(he)
            for node in nodes:
                node_to_edges[node].append(he)

        g = nx.Graph()
        edges = defaultdict(int)
        for eds in node_to_edges.values():
            if len(eds) > 0:
                for e in combinations(eds, 2):
                    e = sorted(e)
                    edges[tuple(e)] += 1

        for e, v in edges.items():
            if v >= s:
                g.add_edge(e[0], e[1], w=v)

        return g

    def bipartite_projection(self, start: int = None, end: int = None) -> object:
        """
        The bipartite_projection function creates a bipartite graph representation of the ASH leveraging NetworkX.
        The nodes of type 0 represent ASH nodes, while nodes of type 1 represent hyperedges.
        A type-0-node is connected to a type-1-node if the corresponding node is contained in the
        corresponding hyperedge.
        Parameters start and end define an optional time window to consider only entities active during those
        points in time.
        The function returns this new graph object.

        :param start: Specify the start of a time window
        :param end: Specify the end of a time window
        :return: A networkx graph object
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

    def dual_hypergraph(self, start: int = None, end: int = None) -> tuple:
        """
        The dual_hypergraph function takes a hypergraph and returns the dual of that hypergraph.
        The dual of a hypergraph is a graph where each hyperedge becomes a node, and each
        node is connected to every other node in its corresponding hyperedge. The function also
        returns an edge_to_nodes dictionary which maps edges to their corresponding nodes.


        :param start: Specify the start of a time window
        :param end: SpSpecify the end of a time window
        :return: the dual ASH and a node-to-edge mapping dictionary
        """

        b = ASH(hedge_removal=True)
        node_to_edges = defaultdict(list)
        for he in self.hyperedge_id_iterator(start=start, end=end):
            nodes = self.get_hyperedge_nodes(he)
            for node in nodes:
                node_to_edges[node].append(he)

        node_to_eid = {}
        for node, edges in node_to_edges.items():
            b.add_hyperedge(edges, 0, end=None, **{"name": node})
            eid = b.get_hyperedge_id(edges)
            node_to_eid[node] = eid

        return b, node_to_eid

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
                for tid in range(start, end + 1):
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

    def get_s_incident(
        self, hyperedge_id: str, s: int, start: int = None, end: int = None
    ) -> list:
        """
        Returns a list of 2-tuples of the form (**heid**, **comm**), where heid is the id of a hyperedge that
        intersects the input hyperedge by at least s nodes, and comm is the number of common nodes between the two
        hyperedges.

        :param hyperedge_id: Specify the hyperedge id
        :param s: Specify the minimum number of common nodes
        :param start: Specify the start of the time window
        :param end: Specify the end of the time window
        :return: the list of s_incident hyperedges
        """

        res = []
        nodes = set(self.get_hyperedge_nodes(hyperedge_id))
        for he in self.hyperedge_id_iterator(start=start, end=end):
            if he != hyperedge_id:
                he_nodes = set(self.get_hyperedge_nodes(he))
                incident = len(nodes & he_nodes)
                if incident >= s:
                    res.append((he, incident))

        return res

    def induced_hypergraph(self, hyperedge_set: list) -> object:
        """
        The induced_hypergraph function takes a list of hyperedge IDs and returns the induced hypergraph.
        The induced hypergraph is constructed by adding all nodes in the original graph that are included in any of
        the hyperedges in the list. It also preserves temporal and attributive information.

        :param hyperedge_set: Specify which hyperedges to induce
        :return: A new ASH object and a dictionary that maps the old hyperedge ids to the new ones
        """

        b = ASH()
        nodes_to_add = {}
        old_eid_to_new = {}
        for he in self.hyperedge_id_iterator():
            if he in hyperedge_set:
                att = self.get_hyperedge_attributes(he)["t"]
                nodes = self.get_hyperedge_nodes(he)
                for n in nodes:
                    nodes_to_add[n] = None

                for span in att:
                    b.add_hyperedge(self.get_hyperedge_nodes(he), span[0], span[1])
                he1 = b.get_hyperedge_id(nodes)
                old_eid_to_new[he] = he1

        for node in nodes_to_add:
            prof = self.get_node_profile(node)
            spans = prof.get_attribute("t")
            for t in spans:
                pt = self.get_node_profile(node, tid=t[0])
                b.add_node(node, t[0], t[1], attr_dict=pt)

        return b, old_eid_to_new
