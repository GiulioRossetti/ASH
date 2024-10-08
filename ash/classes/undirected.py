from collections import defaultdict
from itertools import combinations
from typing import Union, List, Tuple

import networkx as nx

from .node_profile import NProfile


class ASH(object):
    def __init__(self):
        """

        :param hedge_removal: whether to allow hyperedge removal or not
        """

        # edge data
        self._current_hyperedge_id = 0
        self._snapshots = {}  # {time: [edge_id, ...]}
        self._eid2nids = {}  # {edge_id: nodes}
        self._nids2eid = {}  # {nodes: edge_id}
        self._edge_attributes = defaultdict(
            lambda: defaultdict(dict)
        )  # {edge_id: {attr_name: {time: attr_value, ...}}}

        # node data
        self._node_attrs = defaultdict(
            lambda: defaultdict(dict)
        )  # {node: {time: {attr_name: attr_value, ...}}}
        self._stars = defaultdict(set)  # {node: {edge_id, ...}}

    def __presence_to_intervals(self, presence: list) -> List[Tuple[int, int]]:
        """
        takes a list of integers and converts it into a list of tuples.
        Each tuple represents a time interval where the node is present.

        :param presence: A list of integers representing the presence of a node
        :return: A list of tuples representing time intervals
        """
        # e.g. [1, 2, 3, 5, 6, 7, 9] -> [(1, 3), (5, 7), (9, 9)]
        presence = sorted(presence)
        intervals = []
        start = presence[0]
        end = presence[0]

        for i in range(1, len(presence)):
            if presence[i] == end + 1:
                # Extend the current interval
                end = presence[i]
            else:
                # Add the completed interval and start a new one
                intervals.append((start, end))
                start = presence[i]
                end = presence[i]

        # Append the last interval
        intervals.append((start, end))

        return intervals

    def __time_window(self, start, end):

        if start is None:
            time_window = self.temporal_snapshots_ids()
        else:
            if end is None:
                end = start

            time_window = list(range(start, end + 1))

        return time_window

    def stream_interactions(self) -> Tuple[int, str, str]:
        """
        Yields the interactions in the ASH as a stream of tuples (t, hedge_id, op).
        op is a string indicating the operation performed on the hyperedge.
        The '+' indicates the addition at time t, and '-' indicates the removal.


        :return: A stream of interactions

        :Example:
        >>> h = ASH()
        >>> h.add_hyperedge([1, 2, 3], start=1, end=3)
        >>> h.add_hyperedge([1, 2, 4], start=2, end=4)
        >>> for t, hedge_id, op in h.stream_interactions():
        >>>     print(t, hedge_id, op)
        """

        # absence of previously present hyperedges is a removal
        # presence of previously absent hyperedges is an addition

        # yield the first snapshot
        tids = self.temporal_snapshots_ids()
        for hedge_id in self._snapshots[tids[0]]:
            yield tids[0], hedge_id, "+"

        # yield additions and removals
        for t in tids[1:]:
            # additions
            for hedge_id in self._snapshots[t]:
                if hedge_id not in self._snapshots[t - 1]:
                    yield t, hedge_id, "+"
            # removals
            for hedge_id in self._snapshots[t - 1]:
                if hedge_id not in self._snapshots[t]:
                    yield t, hedge_id, "-"

    def temporal_snapshots_ids(self) -> list:
        """
        Returns the list of temporal snapshots ids for the ASH, i.e.,
        integers representing points in time.

        :return: An ordered list of temporal snapshot ids

        :Example:
        >>> h = ASH()
        >>> h.add_hyperedge([1, 2, 3], start=1, end=3)
        >>> h.temporal_snapshots_ids()
        >>> # [1, 2, 3]
        """
        return sorted(self._snapshots.keys())

    ##### Building the ASH #####
    def add_hyperedge(self, nodes: list, start: int, end: int = None, **kwargs) -> None:
        """
        Adds a hyperedge to the ASH.
        The start and end parameters define the time span in which the hyperedge is present.
        Both start and end are inclusive.
        To add a hyperedge at multiple time windows, call this function multiple times with different start and end values.
        :param nodes: A list of node ids
        :param start: The start of the time span
        :param end: The end of the time span
        :param attrs: A dictionary of attributes
        :return:

        :Example:
        >>> h = ASH()
        >>> h.add_hyperedge([1, 2, 3], start=1, end=3, label="a label")

        """
        if end is None:
            span = (start, start)
        else:
            span = (start, end)

        nodes = frozenset(nodes)

        if nodes in self._nids2eid:
            # hyperedge already present (also nodes)
            hyperedge_id = self._nids2eid[nodes]
        else:
            # new hyperedge
            self._current_hyperedge_id += 1
            hyperedge_id = f"e{self._current_hyperedge_id}"
            self._nids2eid[nodes] = hyperedge_id
            self._eid2nids[hyperedge_id] = nodes

            # add nodes if not present
            for n in nodes:
                # if not present at all
                if n not in self._node_attrs:
                    self.add_node(n, span[0], span[1], attr_dict={})
                else:
                    # if present but not in the time span
                    for t in range(span[0], span[1] + 1):
                        if t in self._node_attrs[n]:
                            self.add_node(n, t, attr_dict={})
                self._stars[n].add(hyperedge_id)

        # whether it was already present or not,
        # add the hyperedge to all specified time ids
        for t in range(span[0], span[1] + 1):
            if t not in self._snapshots:
                self._snapshots[t] = set()
            self._snapshots[t].add(hyperedge_id)

        for k, v in kwargs.items():
            if k not in self._edge_attributes[hyperedge_id]:
                self._edge_attributes[hyperedge_id][k] = {}
            self._edge_attributes[hyperedge_id][k][start] = v

    def add_hyperedges(self, hyperedges: list, start: int, end: int = None) -> None:
        """
        Adds a list of hyperedges to the ASH.
        The start and end parameters define the time span in which the hyperedges are present.
        Both start and end are inclusive.

        :param hyperedges: A list of hyperedges, where each hyperedge is a list of node ids
        :param start: The start of the time span
        :param end: The end of the time span
        :return:

        :Example:
        >>> h = ASH()
        >>> h.add_hyperedges([[1, 2, 3], [4, 5, 6]], start=1, end=3)
        >>> ### add hyperedges from graph cliques
        >>> G = nx.barabasi_albert_graph(100, 3)
        >>> h.add_hyperedges(nx.find_cliques(G), start=1, end=3)
        """
        for hedge in hyperedges:
            self.add_hyperedge(hedge, start, end)

    def add_node(
        self, node: int, start: int, end: int = None, attr_dict: object = None
    ) -> None:
        """
        Adds a node to the ASH. If the node is already present, it will update the node's profile.
        The start and end parameters define the time span in which the node is present.
        Both start and end are inclusive.
        The attr_dict parameter is a dictionary of attributes to be added to the node's profile.
        To add a node at multiple time points, call this function multiple times with different start and end values.

        :param node: The node id
        :param start: The start of the time span
        :param end: The end of the time span
        :param attr_dict: A dictionary of attributes
        :return:

        :Example:
        >>> h = ASH()
        >>> attr_dict = {"age": 25, "gender": "M"} # can also be a NProfile object
        >>> h.add_node(1, start=1, end=3, attr_dict=attr_dict)

        """
        if end is None:
            span = (start, start)
        else:
            span = (start, end)
        if attr_dict is None:
            attr_dict = {}

        for t in range(span[0], span[1] + 1):
            self._node_attrs[node][t] = dict(attr_dict)

    def add_nodes(
        self, nodes: list, start: int, end: int = None, node_attr_dict: dict = None
    ) -> None:
        """
        Adds a list of nodes to the ASH. If the nodes are already present, it will update the nodes' profiles.
        See the add_node function for more details.

        :param nodes: A list of node ids
        :param start: The start of the time span
        :param end: The end of the time span
        :param node_attr_dict: A dictionary mapping node ids to attribute dictionaries
        :return: None

        :Example:
        >>> h = ASH()
        >>> g = nx.barabasi_albert_graph(100, 3)
        >>> nodes = list(g.nodes())
        >>> node_attr_dict = {n: {"age": random.randint(20, 50)} for n in nodes}
        >>> h.add_nodes(nodes, start=1, end=3, node_attr_dict=node_attr_dict)
        """
        if node_attr_dict is None:
            node_attr_dict = {}
        for node in nodes:
            attr = None if node not in node_attr_dict else node_attr_dict[node]
            self.add_node(node, start, end, attr)

    def remove_hyperedge(
        self, hyperedge_id: str, start: int = None, end: int = None
    ) -> None:
        """
        Removes a hyperedge from the ASH. The function takes the hyperedge id and the
        time span in which the hyperedge is to be removed.
        If no time span is specified, the hyperedge is removed from all time points.

        :param hyperedge_id: The hyperedge id
        :param start: The start of the time span
        :param end: The end of the time span
        :return: None

        :Example:
        >>> h = ASH()
        >>> h.add_hyperedge([1, 2, 3], start=1, end=3)
        >>> h.add_hyperedge([1, 2, 4], start=2, end=4)
        >>> h.remove_hyperedge("e1", start=1) # remove the hyperedge at time 1, but not at time 2 and 3
        >>> h.remove_hyperedge("e2") # remove the hyperedge at all time points
        """

        time_window = set(self.__time_window(start, end))

        still_exists = False
        for t in self.temporal_snapshots_ids():
            if t in time_window:
                self._snapshots[t].remove(hyperedge_id)
                self._edge_attributes[hyperedge_id].pop(t, None)
            elif self.has_hyperedge(hyperedge_id, t):
                still_exists = True

        if not still_exists:
            del self._eid2nids[hyperedge_id]
            del self._edge_attributes[hyperedge_id]
            nodes = self.get_hyperedge_nodes(hyperedge_id)
            del self._nids2eid[nodes]
            for n in nodes:
                self._stars[n].remove(hyperedge_id)

    def remove_hyperedges(
        self, hyperedges: list, start: int = None, end: int = None
    ) -> None:
        """
        Removes a list of hyperedges from the ASH. The function takes the list of
        hyperedge ids and the time span in which the hyperedges are to be removed.
        If no time span is specified, the hyperedges are removed from all time points.

        :param hyperedges: The list of hyperedge ids
        :param start: The start of the time span
        :param end: The end of the time span
        :return: None

        :Example:
        >>> h = ASH()
        >>> h.add_hyperedges([[1, 2, 3], [4, 5, 6]], start=1, end=3)
        >>> h.remove_hyperedges(["e1", "e2"], start=3) # remove the hyperedges at time 3
        """

        for hedge in hyperedges:
            self.remove_hyperedge(hedge, start, end)

    def remove_node(self, node: int, start: int = None, end: int = None) -> None:
        """
        The remove_node function removes a node from the ASH. The function takes the node id and the time span in which
        the node is to be removed.

        :param node: The node id
        :param start: The start of the time span
        :param end: The end of the time span

        :Example:
        >>> h = ASH()
        >>> h.add_node(1, start=1, end=3)
        >>> h.add_node(2, start=2, end=4)
        >>> h.remove_node(1, start=1) # remove node 1 at time 1, but not at time 2 and 3
        >>> h.remove_node(2) # remove node 2 at all time points
        """

        time_window = self.__time_window(start, end)

        for t in time_window:
            if t in self._node_attrs[node]:
                del self._node_attrs[node][t]
        if not self._node_attrs[node]:
            del self._node_attrs[node]

        for hedge in self._stars[node]:
            self.remove_hyperedge(hedge, start, end)
        if not self._stars[node]:
            del self._stars[node]

    def remove_nodes(self, nodes: list, start: int = None, end: int = None) -> None:
        """
        Removes a list of nodes from the ASH. The function takes the list of node ids and the
        time span in which the nodes are to be removed.

        :param nodes: The list of node ids
        :param start: The start of the time span
        :param end: The end of the time span

        :Example:
        >>> h = ASH()
        >>> h.add_nodes([1, 2, 3], start=0, end=3)
        >>> h.add_nodes([4,5,6,7], start=2)
        >>> h.remove_nodes([1, 2, 3], start=1) # remove nodes 1, 2, and 3 at time 1, but not at time 0, 2 and 3
        >>> h.remove_nodes([4, 5, 6, 7]) # remove nodes 4, 5, 6, and 7 at all time points

        """
        for node in nodes:
            self.remove_node(node, start, end)

    def remove_unlabelled_nodes(
        self, attr_name: str, start: int = None, end: int = None
    ) -> None:
        """
        Removes nodes that do not have a specific attribute.
        The function takes the attribute name and the time span in which the nodes are to be removed.

        :param attr_name: The attribute name
        :param start: The start of the time span
        :param end: The end of the time span

        :Example:
        >>> h = ASH()
        >>> h.add_node(1, start=1, end=3, attr_dict={"age": 25})
        >>> h.add_node(2, start=2, end=4, attr_dict={"age": 30})
        >>> h.add_node(3, start=1, end=3)

        >>> h.remove_unlabelled_nodes("age") # removes node 3
        """

        time_window = self.__time_window(start, end)

        for node, t_attrs in self._node_attrs.items():
            for t in time_window:
                if t in t_attrs and attr_name not in t_attrs[t]:
                    self.remove_node(node, t, t)

    ##### Node and Hyperedge Queries #####
    def nodes(self, start: int = None, end: int = None) -> list:
        """
        Returns the list of nodes in the ASH.
        If a snapshot id (tid) is specified, it will return only those nodes that appear in that snapshot.

        :param tid: Optional temporal snapshot id
        :return: The list of nodes in the ASH

        :Example:
        >>> h = ASH()
        >>> h.add_node(1, start=1, end=3)
        >>> h.add_node(2, start=1, end=3)
        >>> h.add_hyperedge([ 3, 4, 5], start=1, end=3)
        >>> h.nodes()
        >>> # [1, 2, 3, 4, 5]
        """
        if start is None:
            return list(self._node_attrs.keys())
        res = set()
        for he in self.hyperedges(start, end):
            res.update(self.get_hyperedge_nodes(he))

        return list(res)

    def hyperedges(
        self,
        start: int = None,
        end: int = None,
        hyperedge_size: int = None,
        as_ids: bool = True,
    ) -> list:
        """ """
        if start is None:
            hyperedges = list(self._eid2nids.keys())
        else:
            time_window = self.__time_window(start, end)
            hyperedges = set()
            for t in time_window:
                hyperedges.update(self._snapshots[t])
        if hyperedge_size is not None:
            hyperedges = [
                h
                for h in hyperedges
                if len(self.get_hyperedge_nodes(h)) == hyperedge_size
            ]

        if as_ids:
            return list(hyperedges)
        return [self._eid2nids[h] for h in hyperedges]

    def has_hyperedge(
        self, edge: Union[str, list, set, tuple], start: int = None, end: int = None
    ) -> bool:
        """
        Checks if a hyperedge is present in the ASH. The edge parameter can be either a list of nodes or a hyperedge id.
        If the tid parameter is specified, it will check if the hyperedge
        is active in that snapshot.

        :param edge: Either a list of nodes or a hyperedge id
        :param tid: The temporal snapshot id
        :return: True if the hyperedge is present, False otherwise
        """
        if not isinstance(edge, str):
            edge = self.get_hyperedge_id(edge)

        if start is None:
            return edge in self._eid2nids
        return edge in set(self.hyperedges(start, end))

    def has_node(self, node: int, start: int = None, end: int = None) -> bool:
        """
        Checks if a node is present in the ASH. If the tid parameter is specified, it will check if the node is active in that snapshot.

        :param node: The node id
        :param tid: The temporal snapshot id
        :return: True if the node is present, False otherwise
        """
        return node in set(self.nodes(start=start, end=end))

    def get_hyperedge_nodes(self, hyperedge_id: str) -> tuple:
        """
        Retrieve the nodes contained in a hyperedge.

        :param hyperedge_id: The hyperedge id
        :return: The list of nodes in the hyperedge
        """
        return self._eid2nids[hyperedge_id]

    def get_hyperedge_id(self, nodes: list) -> str:
        """
        Retrieve the hyperedge id given a list of nodes.

        :param nodes: The list of nodes
        :return: The hyperedge id
        """
        return self._nids2eid[tuple(sorted(nodes))]

    ##### Attribute-related methods #####

    def get_node_profile(self, node: int, tid: int = None) -> NProfile:
        """
        The get_node_profile function returns the profile of a node at a specific time point.

        :param node: The node id
        :param tid: The temporal snapshot id
        :return: The profile of the node
        """

        if tid is None:
            attr_dict = defaultdict(dict)
            for t in self._node_attrs[node]:
                attr_names = self._node_attrs[node][t].keys()
                for attr in attr_names:
                    attr_dict[attr][t] = self._node_attrs[node][t][attr]
            return NProfile(node, **attr_dict)

        return NProfile(node, **self._node_attrs[node][tid])

    def get_node_attribute(
        self,
        node: int,
        attr_name: str,
        tid: int = None,
    ) -> object:
        """
        Returns the value of a specific attribute of a node.
        If a tid is specified, it will return the attribute value at that time point.
        Otherwise, it will return the aggregate value over all time points.
        See the get_node_profile function for more details.

        :param node: The node id
        :param attr_name: The attribute name
        :param tid: The temporal snapshot id
        :return: The attribute value
        """
        return self.get_node_profile(node, tid=tid).get_attribute(attr_name)

    def get_node_attributes(self, node: int, tid: int = None) -> dict:
        """
        Returns a dictionary of the attributes associated with a node.
        If tid is specified, it will only return the attribute values for that time point.

        :param node: The node id
        :param tid: The temporal snapshot id
        :return: The attributes of a node
        """
        return self.get_node_profile(node, tid).get_attributes()

    def list_node_attributes(self, categorical=False, tid: int = None) -> dict:
        """
        Returns a dictionary of the attributes and their values. The
        function takes in two parameters: categorical, which is a boolean that determines whether to include
        numerical attributes, and tid, which is an integer that represents the temporal id. The default value for
        categorical is False and the default value for tid is None.

        :param self: Bind the method to the object
        :param categorical: Specify whether the attributes are categorical or not
        :param tid: Specify the temporal id
        :return: A dictionary of attribute names and the values they can take
        """

        attributes = defaultdict(set)
        if tid is None:
            for node in self.nodes():
                for t, attr_dict in self._node_attrs[node].items():
                    for attr in attr_dict:
                        attributes[attr].add(attr_dict[attr])
        else:
            for node in self.nodes(tid):
                for attr in self._node_attrs[node][tid]:
                    attributes[attr].add(self._node_attrs[node][tid][attr])

        if categorical:
            numerical = [
                attribute
                for attribute in attributes
                if not isinstance(list(attributes[attribute])[0], str)
            ]
            for attribute in numerical:
                del attributes[attribute]

        return attributes

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
        if tid is None:
            return self._edge_attributes[hyperedge_id][attribute_name]
        return self._edge_attributes[hyperedge_id][tid][attribute_name]

    def get_hyperedge_attributes(self, hyperedge_id: str, tid: int = None) -> dict:
        """
        The get_hyperedge_attributes function returns a dictionary of the attributes associated with a hyperedge.
        If tid is specified, it will only return the attribute values for that timeslot.

        :param hyperedge_id: Specify the hyperedge to get the attributes of
        :param tid: Specify a snapshot
        :return: The attributes of a hyperedge
        """
        if tid is None:
            return self._edge_attributes[hyperedge_id]
        return self._edge_attributes[hyperedge_id][tid]

    def list_hyperedge_attributes(self, categorical=False, tid: int = None) -> dict:
        """
        The list_hyperedge_attributes function returns a dictionary of the attributes and their values.
        The function takes in two parameters: categorical, which is a boolean that determines whether to include
        numerical attributes, and tid, which is an integer that represents the temporal id. The default value for
        categorical is False and the default value for tid is None.

        :param categorical: Specify whether the attributes are categorical or not
        :param tid: Specify the temporal id
        :return: A dictionary of attribute names and the values they can take
        """
        attributes = defaultdict(set)
        if tid is None:
            for hedge in self.hyperedges():
                for t, attr_dict in self._edge_attributes[hedge].items():
                    for attr in attr_dict:
                        attributes[attr].add(attr_dict[attr])
        else:
            for hedge in self.hyperedges(tid):
                for attr in self._edge_attributes[hedge][tid]:
                    attributes[attr].add(self._edge_attributes[hedge][tid][attr])

        if categorical:
            numerical = [
                attribute
                for attribute in attributes
                if not isinstance(list(attributes[attribute])[0], str)
            ]
            for attribute in numerical:
                del attributes[attribute]

        return attributes

    def get_hyperedge_weight(self, hyperedge_id):
        """
        The get_hyperedge_weight function returns the weight of a hyperedge,

        :param hyperedge_id: The hyperedge id
        :return: The weight of the hyperedge
        """
        # TODO: account for time
        weight = self.get_hyperedge_attribute(hyperedge_id, "weight")
        if weight is None:
            return 1

    ##### Statistics #####
    def number_of_nodes(self, start: int = None, end: int = None) -> int:
        """
        The number_of_nodes function returns the number of nodes in the ASH.
        If start and end are specified, it will return the number of nodes that are active in that time window (both inclusive).
        Otherwise, it will return the total number of nodes that are active in tid.

        :param start: Optional temporal snapshot id
        :param end: Optional temporal snapshot id
        :return: The number of nodes in the ASH
        """

        return len(self.nodes(start, end))

    def number_of_hyperedges(self, start: int = None, end: int = None) -> int:
        """
        Returns the number of hyperedges in the ASH.
        If start and end are specified, it will return the number of hyperedges that are active in that time window (both inclusive).

        :param start: Optional temporal snapshot id
        :param end: Optional temporal snapshot id
        :return: The number of hyperedges in the ASH
        """

        return self.size(start, end)

    def size(self, start: int = None, end: int = None) -> int:
        """
        Returns the number of hyperedges in the ASH.
        If start and end are specified, it will return the number of hyperedges that are active in that time window (both inclusive).

        :param start: Optional temporal snapshot id
        :param end: Optional temporal snapshot id
        :return: The number of hyperedges in the ASH
        """

        return len(self.hyperedges(start, end))

    def hyperedge_size_distribution(self, start: int = None, end: int = None) -> dict:
        """
        Returns a dictionary of the number of hyperedges with a given size.
        The function takes two optional arguments, start and end. If only start is provided, then it will return the
        distribution of hyperedge sizes starting from (and including) that time step. If both start and end are
        provided, then it will return the distribution between those two indices (inclusive). The default behavior is
        to return the distribution over all time steps.

        :param start: Specify the starting time of the temporal slice
        :param end: Specify the ending time of the temporal slice
        :return: A size_to_count dictionary that contains the number of hyperedges with a given size
        """

        distr = defaultdict(int)
        hes = self.hyperedges(start, end)

        for he in hes:
            distr[len(self.get_hyperedge_nodes(he))] += 1

        return dict(distr)

    def degree_distribution(self, start: int = None, end: int = None) -> dict:
        """
        Returns a dictionary of the number of nodes with a given degree.
        If start and end are specified, returns the distribution for that time window.

        :param start: Optional temporal snapshot id
        :param end: Optional temporal snapshot id
        :return: A degree_to_count dictionary that contains the number of nodes with a given degree
        """
        distr = defaultdict(int)

        for node in self.nodes(start, end):
            distr[self.degree(node, start, end)] += 1

        return dict(distr)

    ##### Node-centric Analysis #####
    def star(
        self,
        node: int,
        start: int = None,
        end: int = None,
        hyperedge_size: int = None,
        as_ids: bool = True,
    ) -> list:
        """
        Retrieve the star of a node in the ASH. The star of a node is the set of hyperedges that contain the node.
        If the hyperedge_size parameter is specified, only hyperedges of that size are returned.
        If the tid parameter is specified, only hyperedges that are active in that snapshot are returned.

        :param node: The node id
        :param start: The start of the time window
        :param end: The end of the time window
        :param hyperedge_size: The size of the hyperedges
        :param as_ids: If True, return the hyperedge ids, otherwise return the hyperedges as sets of nodes
        :return: The set of hyperedge ids that contain the node
        """
        if start is None:
            star = self._stars[node]

        else:
            time_window = self.__time_window(start, end)
            star = set()
            for t in time_window:
                star.update(self._stars[node].intersection(self._snapshots[t]))

        if hyperedge_size is not None:
            star = [
                s for s in star if len(self.get_hyperedge_nodes(s)) == hyperedge_size
            ]

        if as_ids:
            return list(star)
        return [self.get_hyperedge_nodes(s) for s in star]

    def degree(
        self, node: int, start: int = None, end: int = None, hyperedge_size: int = None
    ) -> int:
        """
        The degree function returns the number of hyperedges that a given node is part of.
        If the optional argument hyperedge_size is specified, then it returns only the number
        of hyperedges that have exactly this many nodes.

        :param node: Specify the node id
        :param start: Specify the start of a time window
        :param end: Specify the end of a time window
        :param hyperedge_size: Specify the size of hyperedges to be counted
        :return: The degree of a node
        """
        return len(self.star(node, start, end, hyperedge_size, as_ids=False))

    def s_degree(self, node: int, s: int, start: int = None, end: int = None) -> int:
        """
        The s_degree function returns the s-degree of a specific node.

        :param node: Specify the node for which to compute the degree
        :param s: Specify the minimum number of nodes in each hyperedge to be counted
        :param tid: Temporal id
        :return: The number of hyperedges in the start that have at least s nodes
        """

        degs = self.degree_by_hyperedge_size(node, start=start, end=end)
        res = 0
        for k, v in degs.items():
            if k >= s:
                res += v
        return res

    def degree_by_hyperedge_size(
        self, node: int, start: int = None, end: int = None
    ) -> dict:
        """
        Given a node, the degree_by_hyperedge_size function returns a dictionary where keys are the sizes
        of the hyperedges in the node's star, and values are the raw frequency of that size in the star.
        For example, if there are two hyperedges with three nodes each and one
        hyperedge with four nodes, then the returned dictionary would be: {3: 2, 4: 1}.
        The optional parameter tid restricts the hyperedges to those active in that point in time.

        :param node: Specify the node for which we want to calculate the degree distribution
        :param start: Specify the start of a time window
        :param end: Specify the end of a time window

        :return: A dictionary where the keys are the node's star's hyperedge sizes and the values are their frequencies
        """

        distr = defaultdict(int)
        star = self.star(node, start=start, end=end, as_ids=False)
        for he in star:
            distr[len(he)] += 1
        return dict(distr)

    def neighbors(
        self,
        node: int,
        start: int = None,
        end: int = None,
        hyperedge_size: int = None,
    ) -> set:
        """
        Retrieve the neighbors of a node in the ASH. Neighbors are nodes that share at least one hyperedge with the input node.
        If the hyperedge_size parameter is specified, only neighbors that share a hyperedge of that size are returned.
        If the tid parameter is specified, only neighbors that are active in that snapshot are returned.

        :param node: The node id
        :param start: The start of the time window
        :param end: The end of the time window
        :param hyperedge_size: The size of the hyperedges
        :return: The set of neighbors of the node
        """
        neighbors = set()
        star = self.star(node, start, end, hyperedge_size, as_ids=False)
        for he in star:
            neighbors.update(he)
        neighbors.remove(node)
        return neighbors

    def number_of_neighbors(
        self, node: int, start: int = None, end: int = None, hyperedge_size: int = None
    ) -> int:
        """
        The number_of_neighbors function returns the number of neighbors of a node in the ASH.

        Otherwise, it will return the total number of neighbors that are active in tid.

        :param node: The node id
        :param start: Optional temporal snapshot id
        :param end: Optional temporal snapshot id
        :param hyperedge_size: Optional hyperedge size
        :return: The number of neighbors of the node in the ASH
        """
        return len(self.neighbors(node, start, end, hyperedge_size))

    ### Transformations and Projections ###

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
        hes = self.hyperedges(start, end)

        g = nx.Graph()
        for he in hes:
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
        The dual of a hypergraph is a hypergraph where each hyperedge becomes a node, and each
        node is connected to every other node in its corresponding hyperedge. The function also
        returns an edge_to_nodes dictionary which maps edges to their corresponding nodes.


        :param start: Specify the start of a time window
        :param end: SpSpecify the end of a time window
        :return: the dual ASH and a node-to-edge mapping dictionary
        """
        hes = self.hyperedges(start, end)
        b = ASH()
        node_to_edges = defaultdict(list)
        for he in hes:
            nodes = self.get_hyperedge_nodes(he)
            for node in nodes:
                node_to_edges[node].append(he)

        node_to_eid = {}
        for node, edges in node_to_edges.items():
            b.add_hyperedge(edges, 0, end=None, **{"name": node})
            eid = b.get_hyperedge_id(edges)
            node_to_eid[node] = eid

        return b, node_to_eid

    def s_line_graph(self, s: int = 1, start: int = None, end: int = None) -> nx.Graph:
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
        hes = self.hyperedges(start, end)

        for he in hes:
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

    ##### Temporal Analysis #####

    def avg_number_of_nodes(self) -> float:
        """
        The avg_number_of_nodes function returns the average number of nodes in an ASH over time, i.e., the
        sum of each snapshot's nodes divided by the number of _snapshots.

        :return: The average number of nodes in the ASH over all _snapshots
        """

        nodes_snapshots = [
            self.number_of_nodes(tid) for tid in self.temporal_snapshots_ids()
        ]
        return sum(nodes_snapshots) / len(nodes_snapshots)

    def avg_number_of_hyperedges(self) -> float:
        """
        The avg_number_of_hyperedges function returns the average number of
        hyperedges in each snapshot.

        :return: The average number of hyperedges per snapshot
        """

        hes_snapshots = [self.size(tid) for tid in self.temporal_snapshots_ids()]
        return sum(hes_snapshots) / len(hes_snapshots)

    def node_presence(self, node: int, as_intervals: bool = False) -> list:
        """
        The node_presence function returns the time span in which a node is present in the ASH.

        :param node: The node id
        :param as_intervals: If True, the function will return the time span as a list of tuples
        :return: A list of tuples representing the time span in which the node is present
        """
        if as_intervals:
            return self.__presence_to_intervals(list(self._node_attrs[node].keys()))
        return sorted(self._node_attrs[node].keys())

    def hyperedge_presence(self, hyperedge_id: str, as_intervals: bool = False) -> list:
        """
        The hyperedge_presence function returns the time span in which a hyperedge is present in the ASH.

        :param hyperedge_id: The hyperedge id
        :param as_intervals: If True, the function will return the time span as a list of tuples
        :return: A list of tuples representing the time span in which the hyperedge is present
        """
        pres = []
        for t in self.temporal_snapshots_ids():
            if self.has_hyperedge(hyperedge_id, t):
                pres.append(t)

        if as_intervals:
            return self.__presence_to_intervals(pres)
        return pres

    def node_contribution(self, node) -> float:
        """
        The node_contribution function returns the fraction of _snapshots where the node is present.
        This is used to determine how important a node is within the ASH.

        :param node: Node id
        :return: The contribution of a node to the overall coverage
        """

        ucov = 0
        for tid in self.temporal_snapshots_ids():
            ucov += 1 if self.has_node(node, tid) else 0
        return ucov / len(self.temporal_snapshots_ids())

    def hyperedge_contribution(self, hyperedge_id: str) -> float:
        """
        The hyperedge_contribution function calculates the fraction of _snapshots where
        a given hyperedge is active.

        :param hyperedge_id: Specify which hyperedge to calculate the contribution for
        :return: The contribution of a hyperedge
        """

        contr = 0
        for tid in self.temporal_snapshots_ids():
            contr += 1 if self.has_hyperedge(hyperedge_id, tid) else 0
        return contr / len(self.temporal_snapshots_ids())

    def coverage(self) -> float:
        """
        The coverage function is the fraction of nodes in the ASH that are
        present in at least one snapshot over all possible nodes and _snapshots.
        This measure is used to quantify how well the ASH has been preserved over time.

        :return: The fraction of the nodes in the graph that are covered by at least one snapshot
        """

        t = len(self.temporal_snapshots_ids())
        v = self.number_of_nodes()
        w = 0
        for tid in self.temporal_snapshots_ids():
            w += self.number_of_nodes(tid)

        return w / (t * v)

    def uniformity(self) -> float:
        """
        Temporal hypergraph uniformity

        :return: uniformity value for the hypergraph
        """
        nds = self.nodes()
        numerator, denominator = 0, 0
        for u, v in combinations(nds, 2):
            for t in self.temporal_snapshots_ids():
                if self.has_node(u, t) and self.has_node(v, t):
                    numerator += 1
                if self.has_node(u, t) or self.has_node(v, t):
                    denominator += 1
        return numerator / denominator

    ### Slice and Filter ###

    def temporal_slice(
        self, start: int, end: int = None, keep_attrs: bool = True
    ) -> Tuple[object, dict]:
        """
        The temporal_slice constructs a new ASH instance that contains hyperedges active in the given
        time window. It returns both the new instance and a dictionary mapping old hyperedge ids to new hyperedge ids.
        If no end time is specified, then all temporal ids greater or equal to start are considered.

        :param start: Specify the start time of the temporal slice
        :param end: Specify the end of the temporal slice
        :param keep_attrs: Specify whether to keep the attributes of the original nodes
        :return: an ASH instance and a dictionary mapping old hyperedge ids to new hyperedge ids
        """

        res = ASH()
        eid_to_new_eid = {}  # mapping of old hyperedge ids to new hyperedge ids
        if end is None:
            end = start
        for t in range(start, end + 1):
            for hedge_id in self.hyperedges(t):
                nodes = self.get_hyperedge_nodes(hedge_id)
                res.add_hyperedge(nodes, t, t)
                eid_to_new_eid[hedge_id] = res._nids2eid[frozenset(nodes)]

        # copy node profiles
        if keep_attrs:
            for t in range(start, end + 1):
                nodes = self.nodes(t)
                for node in nodes:
                    profile = self.get_node_profile(node, t)
                    res.add_node(node, t, t, profile)

        return res, eid_to_new_eid

    def induced_hypergraph(
        self, hyperedge_set: list, keep_attrs: bool = True
    ) -> Tuple[object, dict]:
        """
        Creates a new ASH instance that contains only the hyperedges specified in the hyperedge_set list.
        The function returns the new ASH instance and a dictionary that maps old hyperedge ids to new hyperedge ids.

        :param hyperedge_set: Specify which hyperedges to induce
        :param keep_attrs: Specify whether to keep the attributes of the original nodes
        :return: A new ASH object and a dictionary that maps the old hyperedge ids to the new ones
        """

        b = ASH()
        old_eid_to_new = {}

        for he in hyperedge_set:
            presence = self.hyperedge_presence(he, as_intervals=True)
            nodes = self.get_hyperedge_nodes(he)
            for span in presence:
                b.add_hyperedge(nodes, span[0], span[1])
            new_he = b.get_hyperedge_id(nodes)
            old_eid_to_new[he] = new_he

            # copy node attributes
            if keep_attrs:
                for t in b.temporal_snapshots_ids():
                    for node in b.nodes(t):
                        profile = self.get_node_profile(node, t)
                        b.add_node(node, start=t, end=t, attr_dict=profile)

        return b, old_eid_to_new

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
        hes = self.hyperedges(start, end)

        for he in hes:
            if he != hyperedge_id:
                he_nodes = set(self.get_hyperedge_nodes(he))
                incident = len(nodes & he_nodes)
                if incident >= s:
                    res.append((he, incident))

        return res

    def incidence(self, edge_set: set, start: int = None, end: int = None) -> int:
        """
        Returns
        TODO understand what this does and if it's useful
        No usages in the codebase


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

    def to_json(self):
        """
        The to_json function returns a JSON representation of the ASH object.
        The function returns a dictionary with the following keys:
        """
        descr = {"nodes": {}, "hedges": {}}

        for hedge in self.hyperedges():
            edge_data = {
                "nodes": self.get_hyperedge_nodes(hedge),
                "attributes": self.get_hyperedge_attributes(hedge),
            }
            edge_data["attributes"]["presence"] = self.hyperedge_presence(
                hedge, as_intervals=True
            )

            descr["hedges"][hedge] = edge_data

        for node in self.nodes():
            npr = self.get_node_profile(node)
            descr["nodes"][node] = npr.get_attributes()
        return descr

    def __str__(self):
        descr = "ASH\n"
        descr += "Nodes: {}\n".format(self.number_of_nodes())
        descr += "Hyperedges: {}\n".format(self.number_of_hyperedges())
        descr += "Snapshots: {}\n".format(len(self.temporal_snapshots_ids()))
        return descr

    def __iter__(self):

        for t in self.temporal_snapshots_ids():
            yield self.hyperedges(t)
