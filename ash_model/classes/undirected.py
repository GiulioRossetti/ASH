from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import (
    Any,
    DefaultDict,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import networkx as nx

from .node_profile import NProfile
from .presence_store import DensePresenceStore, IntervalPresenceStore, PresenceStore


class ASH:
    def __init__(
        self,
        backend: str = "dense",
    ) -> None:
        """
            Initialize an ASH (Attributed Stream Hypergraph) instance.

        :param backend: The backend for storing temporal information on hyperedges. Supported values are "dense" (stores time → set[id]) and "interval" (stores id → list[(start, end)] disjoint intervals).
        :raises ValueError: If an unsupported backend is specified.

        Examples
        --------
        .. code-block:: python

            from ash_model.classes.undirected import ASH

            # Dense (default) backend
            H = ASH()

            # Interval backend
            H2 = ASH(backend="interval")

        """

        if backend == "dense":
            self._snapshots: PresenceStore = DensePresenceStore()
        elif backend == "interval":
            self._snapshots: PresenceStore = IntervalPresenceStore()
        else:
            raise ValueError("backend must be 'dense' or 'interval' (got %r)" % backend)

        # edge data
        self._current_hyperedge_id: int = 0
        self._eid2nids: Dict[str, frozenset[int]] = {}
        self._nids2eid: Dict[frozenset[int], str] = {}

        # {eid : {attr_name: value}}
        self._edge_attributes: DefaultDict[str, DefaultDict[int, Dict[str, Any]]] = (
            defaultdict(lambda: defaultdict(dict))
        )

        # node data
        self._node_attrs: DefaultDict[int, DefaultDict[int, Dict[str, Any]]] = (
            defaultdict(lambda: defaultdict(dict))
        )
        self._stars: DefaultDict[int, Set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def __presence_to_intervals(self, presence: List[int]) -> List[Tuple[int, int]]:
        """Convert a list of time instants into contiguous intervals."""
        presence = sorted(presence)
        intervals: List[Tuple[int, int]] = []
        start = presence[0]
        end = presence[0]

        for i in range(1, len(presence)):
            if presence[i] == end + 1:
                end = presence[i]
            else:
                intervals.append((start, end))
                start = presence[i]
                end = presence[i]

        intervals.append((start, end))
        return intervals

    def __time_window(self, start: Optional[int], end: Optional[int]) -> List[int]:
        if start is None:
            return self.temporal_snapshots_ids()
        if end is None:
            end = start
        return list(range(start, end + 1))

    # ------------------------------------------------------------------
    # Iterators / Streams
    # ------------------------------------------------------------------

    def temporal_snapshots_ids(self) -> List[int]:
        """
        Return a sorted list of all temporal snapshot ids (time instants) in the ASH.
        This method retrieves the keys from the internal snapshot store, which
        represent the time instants when hyperedges are present.

        :return: Sorted list of temporal snapshot ids.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_node(1, 0)
            H.add_node(2, 0)
            H.add_hyperedge([1, 2], start=0)
            tids = H.temporal_snapshots_ids()  # e.g., [0]

        """
        return sorted(self._snapshots.keys())

    def stream_interactions(self) -> Generator[Tuple[int, str, str], None, None]:
        """
        Generate a stream of interactions in the ASH, yielding tuples of the form
        (time_id, hyperedge_id, event_type), where event_type is either
        "+" for activation or "-" for deactivation of a hyperedge at that time.
        This method iterates through the temporal snapshots and compares the sets
        of hyperedges present in consecutive snapshots to determine hyperedge changes.

        :yield: Tuples of (time_id, hyperedge_id, event_type).

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_node(1, 0, 1)
            H.add_node(2, 0, 1)
            H.add_hyperedge([1, 2], start=0, end=1)
            events = list(H.stream_interactions())
            # Example output: [(0, 'e1', '+'), (1, 'e1', '-')]

        """
        tids = self.temporal_snapshots_ids()
        if not tids:
            return
        # First snapshot – pure additions
        for hid in self._snapshots[tids[0]]:
            yield tids[0], hid, "+"
        # Subsequent diffs
        for t in tids[1:]:
            prev, curr = self._snapshots[t - 1], self._snapshots[t]
            for hid in curr - prev:
                yield t, hid, "+"
            for hid in prev - curr:
                yield t, hid, "-"

    def add_hyperedge(
        self,
        nodes: Iterable[int],
        start: int,
        end: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Add a hyperedge to the ASH. If nodes are not already present, they will be added.
        If end time is not specified, then the hyperedge is considered to be present only at `start` (end = start).

        :param nodes: Iterable of node IDs that form the hyperedge.
        :param start: Start time of the hyperedge.
        :param end: End time of the hyperedge (inclusive). If None, the hyperedge is considered to be present only at `start`.
        :param kwargs: Optional attributes for the hyperedge.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_node(1, 0)
            H.add_node(2, 0)
            H.add_hyperedge([1, 2], start=0, weight=3)
            list(H.hyperedges())  # ['e1']
            H.get_hyperedge_weight('e1')  # 3

        """

        span = (start, start if end is None else end)
        f_nodes = frozenset(nodes)

        # Either fetch existing or mint a new hyperedge id ----------------
        if f_nodes in self._nids2eid:
            hid = self._nids2eid[f_nodes]
        else:
            self._current_hyperedge_id += 1
            hid = f"e{self._current_hyperedge_id}"
            self._eid2nids[hid] = f_nodes
            self._nids2eid[f_nodes] = hid
            for n in f_nodes:
                # Ensure node exists for the span (dummy attrs if absent)
                if n not in self._node_attrs:
                    self.add_node(n, span[0], span[1])
                else:
                    for t in range(span[0], span[1] + 1):
                        if t not in self._node_attrs[n]:
                            self.add_node(n, t)
                self._stars[n].add(hid)

        # Register presence ------------------------------------------------
        if isinstance(self._snapshots, IntervalPresenceStore):
            self._snapshots._add_interval(hid, span[0], span[1])
        else:
            for t in range(span[0], span[1] + 1):
                self._snapshots.setdefault(t, set()).add(hid)

        # Store attributes -------------------------------------------------
        if "weight" not in kwargs:
            kwargs["weight"] = 1

        for k, v in kwargs.items():
            self._edge_attributes[hid][k] = v

    def add_hyperedges(
        self,
        hyperedges: Iterable[Iterable[int]],
        start: int,
        end: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Add multiple hyperedges to the ASH. If nodes are not already present, they will be added.
        All hyperedges will be added with the same start and end times.
        if `end` is not specified, then the hyperedges are considered to be present only at `start` (end = start).

        :param hyperedges: Iterable of iterables, where each inner iterable contains node IDs for a hyperedge.
        :param start: Start time of the hyperedges.
        :param end: End time of the hyperedges (inclusive). If None, the hyperedges are considered to be present only at `start`.
        :param kwargs: Optional attributes for the hyperedges. These will be applied to all hyperedges.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_nodes([1, 2, 3, 4], start=0)
            H.add_hyperedges([[1, 2], [2, 3, 4]], start=0)
            H.number_of_hyperedges()  # 2

        """

        for hedge in hyperedges:
            self.add_hyperedge(hedge, start, end, **kwargs)

    # ------------------------------------------------------------------
    # Node management
    # ------------------------------------------------------------------

    def add_node(
        self,
        node: int,
        start: int,
        end: Optional[int] = None,
        attr_dict: Optional[Union[Dict[str, Any], NProfile]] = None,
    ) -> None:
        """
        Add a node to the ASH with optional attributes.
        Attributes can be provided as a dictionary of the form {attr_name: value} or as an NProfile instance.
        If the node is not already present, it will be created with the specified attributes.
        If end time is not specified, then the node is considered to be present only at `start` (end = start).

        To add a node with different attributes at different times, you can call this method multiple times with different `start` and `end` values.

        :param node: Node ID to add.
        :param start: Start time of the node.
        :param end: End time of the node (inclusive). If None, the node is considered to be present only at `start`.
        :param attr_dict: Optional attributes for the node. Can be a dictionary or an NProfile instance.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_node(1, start=0, end=2, attr_dict={"group": "A"})
            H.get_node_attribute(1, "group", tid=0)  # 'A'
            H.has_node(1, 1)  # True

        """

        span = (start, start if end is None else end)
        attrs_source: Dict[str, Any]
        if isinstance(attr_dict, NProfile):
            attrs_source = attr_dict.get_attributes()
        else:
            attrs_source = attr_dict or {}

        for t in range(span[0], span[1] + 1):
            self._node_attrs[node][t] = dict(attrs_source)
            self._snapshots.setdefault(t, set())  # ensure snapshot exists

    def add_nodes(
        self,
        nodes: Iterable[int],
        start: int,
        end: Optional[int] = None,
        node_attr_dict: Optional[Dict[int, Union[NProfile, Dict[str, Any]]]] = None,
    ) -> None:
        """
        Add multiple nodes to the ASH with optional attributes.
        Attributes can be provided as a dictionary mapping node IDs to their attributes.

        :param nodes: Iterable of node IDs to add.
        :param start: Start time of the nodes.
        :param end: End time of the nodes (inclusive). If None, the nodes are considered to be present only at `start`.
        :param node_attr_dict: Optional dictionary mapping node IDs to their attributes. If None, no attributes are added.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_nodes([1, 2, 3], start=0, node_attr_dict={1: {"color": "red"}})
            H.number_of_nodes(0)  # 3
            H.get_node_attribute(1, "color", 0)  # 'red'

        """
        node_attr_dict = node_attr_dict or {}
        for n in nodes:
            self.add_node(n, start, end, node_attr_dict.get(n))

    # ------------------------------------------------------------------
    # Removal helpers
    # ------------------------------------------------------------------

    def remove_hyperedge(
        self,
        hyperedge_id: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        """
        Remove a hyperedge from the ASH, including its presence in the specified time window.

        :param hyperedge_id: ID of the hyperedge to remove.
        :param start: Start time of the removal. If None, the hyperedge is removed from all times.
        :param end: End time of the removal (inclusive). If None, the hyperedge is removed only at `start`.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_nodes([1, 2], start=0)
            H.add_hyperedge([1, 2], start=0, end=2)
            H.remove_hyperedge('e1', start=1, end=2)
            H.has_hyperedge('e1', 0)  # True
            H.has_hyperedge('e1', 1)  # False

        """

        # Determine removal spans
        if isinstance(self._snapshots, IntervalPresenceStore):
            spans = (
                self.hyperedge_presence(hyperedge_id, as_intervals=True)
                if start is None
                else [(start, start if end is None else end)]
            )
            for s, e in spans:
                self._snapshots._remove_interval(hyperedge_id, s, e)
                # Remove edge attributes for these times
                for t in range(s, e + 1):
                    self._edge_attributes[hyperedge_id].pop(t, None)
        else:
            time_window = set(self.__time_window(start, end))
            for t in self.temporal_snapshots_ids():
                if t in time_window:
                    self._snapshots.setdefault(t, set()).discard(hyperedge_id)
                    self._edge_attributes[hyperedge_id].pop(t, None)

        # Cleanup metadata if no presence remains
        still_exists = False
        for t in self.temporal_snapshots_ids():
            if self.has_hyperedge(hyperedge_id, t):
                still_exists = True
                break
        if not still_exists:
            nodes = self.get_hyperedge_nodes(hyperedge_id)
            del self._eid2nids[hyperedge_id]
            del self._nids2eid[nodes]
            self._edge_attributes.pop(hyperedge_id, None)
            for n in nodes:
                self._stars[n].discard(hyperedge_id)

    def remove_hyperedges(
        self,
        hyperedges: Iterable[str],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        """
        Remove multiple hyperedges from the ASH, including their presence in the specified time window.

        :param hyperedges: Iterable of hyperedge IDs to remove.
        :param start: Start time of the removal. If None, the hyperedges are removed from all times.
        :param end: End time of the removal (inclusive). If None, the hyperedges are removed only at `start`.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_nodes([1, 2, 3], start=0)
            H.add_hyperedges([[1, 2], [2, 3]], start=0)
            ids = H.hyperedges()
            H.remove_hyperedges(ids, start=0)
            H.number_of_hyperedges(0)  # 0

        """
        for hid in hyperedges:
            self.remove_hyperedge(hid, start, end)

    def remove_node(
        self,
        node: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        """
        Remove a node from the ASH, including its attributes and all hyperedges it is part of,
        in the specified time window.

        :param node: Node ID to remove.
        :param start: Start time of the removal. If None, the node is removed from all times.
        :param end: End time of the removal (inclusive). If None, the node is removed only at `start`.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_nodes([1, 2], start=0)
            H.add_hyperedge([1, 2], start=0)
            H.remove_node(1, start=0)
            H.has_node(1, 0)  # False

        """

        for t in self.__time_window(start, end):
            self._node_attrs[node].pop(t, None)
        if not self._node_attrs[node]:
            self._node_attrs.pop(node, None)

        for hid in list(self._stars.get(node, [])):
            self.remove_hyperedge(hid, start, end)
        self._stars.pop(node, None)

    def remove_nodes(
        self,
        nodes: Iterable[int],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        """
        Remove multiple nodes from the ASH, including their attributes and all hyperedges they are part of,
        in the specified time window.

        :param nodes: Iterable of node IDs to remove.
        :param start: Start time of the removal. If None, the nodes are removed from all times.
        :param end: End time of the removal (inclusive). If None, the nodes are removed only at `start`.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_nodes([1, 2, 3], start=0)
            H.remove_nodes([1, 3], start=0)
            set(H.nodes(0))  # {2}

        """

        for n in nodes:
            self.remove_node(n, start, end)

    def remove_unlabelled_nodes(
        self,
        attr_name: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        """
        Remove nodes that do not have a specific attribute in the specified time window.
        If no time window is specified, the nodes are checked in all times.

        :param attr_name: The name of the attribute to check for.
        :param start: Start time of the removal. If None, the nodes are checked in all times.
        :param end: End time of the removal (inclusive). If None, the nodes are checked only at `start`.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_node(1, 0, attr_dict={"label": "A"})
            H.add_node(2, 0)
            H.remove_unlabelled_nodes("label", start=0)
            set(H.nodes(0))  # {1}

        """

        for node, t_attrs in list(self._node_attrs.items()):
            for t in self.__time_window(start, end):
                if t in t_attrs and attr_name not in t_attrs[t]:
                    self.remove_node(node, t, t)

    # ------------------------------------------------------------------
    # Node & Hyperedge queries
    # ------------------------------------------------------------------

    def nodes(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> List[int]:
        """
        Return a list of node IDs present in the ASH within the specified time window.

        :param start: Start time of the query. If None, all nodes are considered.
        :param end: End time of the query (inclusive). If None, only the start time is considered.
        :return: List of node IDs.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_nodes([1, 2, 3], start=0)
            H.nodes(0)  # [1, 2, 3] (order not guaranteed)

        """
        if start is None:
            return list(self._node_attrs.keys())
        res: List[int] = []
        if end is None:
            for n in self._node_attrs:
                if start in self._node_attrs[n]:
                    res.append(n)
        else:
            for n in self._node_attrs:
                if any(start <= t <= end for t in self._node_attrs[n]):
                    res.append(n)
        return res

    def hyperedges(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        hyperedge_size: Optional[int] = None,
        as_ids: bool = True,
    ) -> List[Union[str, frozenset[int]]]:
        """
        Return a list of hyperedge IDs or their node sets present in the ASH within the specified time window.
        Hyperedges are identified by their IDs (strings), or by sets of node IDs (frozensets).
        If `hyperedge_size` is specified, only hyperedges of that size are returned.
        The `as_ids` parameter determines whether to return hyperedge IDs or sets of node IDs.

        :param start: Start time of the query. If None, all hyperedges are considered.
        :param end: End time of the query (inclusive). If None, only the start time is considered.
        :param hyperedge_size: If specified, only hyperedges of this size are returned.
        :param as_ids: If True, return hyperedge IDs; if False, return sets of node IDs.
        :return: List of hyperedge IDs or sets of node IDs.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_nodes([1, 2, 3], start=0)
            H.add_hyperedges([[1, 2], [2, 3]], start=0)
            H.hyperedges(0)           # e.g., ['e1', 'e2']
            H.hyperedges(0, as_ids=False)  # [frozenset({1,2}), frozenset({2,3})]
            H.hyperedges(0, hyperedge_size=2)  # only size-2 hyperedges

        """
        if start is None:
            hyperedges_set: Set[str] = set(self._eid2nids.keys())
        else:
            hyperedges_set = set()
            for t in self.__time_window(start, end):
                hyperedges_set.update(self._snapshots[t])

        if hyperedge_size is not None:
            hyperedges_set = {
                h
                for h in hyperedges_set
                if len(self.get_hyperedge_nodes(h)) == hyperedge_size
            }

        if as_ids:
            return list(hyperedges_set)
        return [self._eid2nids[h] for h in hyperedges_set]

    def has_hyperedge(
        self,
        edge: Union[str, Iterable[int]],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> bool:
        """
        Check if a hyperedge is present in the ASH within the specified time window.
        Accepts either a hyperedge ID (string) or an iterable of node IDs.

        :param edge: Hyperedge ID (string) or an iterable of node IDs.
        :param start: Start time of the query. If None, all hyperedges are considered.
        :param end: End time of the query (inclusive). If None, only the start time is considered.
        :return: True if the hyperedge is present, False otherwise.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_nodes([1, 2], start=0)
            H.add_hyperedge([1, 2], start=0)
            H.has_hyperedge('e1', 0)        # True
            H.has_hyperedge([1, 2], 0)      # True
            H.has_hyperedge([2, 3], 0)      # False

        """

        if not isinstance(edge, str):
            try:
                edge = self.get_hyperedge_id(edge)
            except KeyError:
                return False
        if start is None:
            return edge in self._eid2nids
        return edge in set(self.hyperedges(start, end))

    def has_node(
        self, node: int, start: Optional[int] = None, end: Optional[int] = None
    ) -> bool:
        """
        Check if a node is present in the ASH within the specified time window.

        :param node: Node ID to check.
        :param start: Start time of the query. If None, all nodes are considered.
        :param end: End time of the query (inclusive). If None, only the start time is considered.
        :return: True if the node is present, False otherwise.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_node(1, 0)
            H.has_node(1, 0)  # True
            H.has_node(2, 0)  # False

        """
        return node in set(self.nodes(start, end))

    def get_hyperedge_nodes(self, hyperedge_id: str) -> frozenset[int]:
        """
        Get the set of node IDs that form a hyperedge.
        If the hyperedge does not exist, returns an empty frozenset.

        :param hyperedge_id: ID of the hyperedge.
        :return: A frozenset of node IDs that are part of the hyperedge.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_hyperedge([1, 2], start=0)
            H.get_hyperedge_nodes('e1')  # frozenset({1, 2})

        """
        return self._eid2nids.get(hyperedge_id, frozenset())

    def get_hyperedge_id(self, nodes: Iterable[int]) -> str:
        """
        Get the hyperedge ID for a given set of node IDs.
        If the hyperedge does not exist, raises a KeyError.

        :param nodes: Iterable of node IDs that form the hyperedge.
        :return: The ID of the hyperedge as a string.
        :raises KeyError: If the hyperedge does not exist.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_hyperedge([1, 2], start=0)
            eid = H.get_hyperedge_id([1, 2])  # 'e1'

        """
        return self._nids2eid[frozenset(nodes)]

    # ------------------------------------------------------------------
    # Node & Hyperedge attributes
    # ------------------------------------------------------------------

    def get_node_profiles_by_time(self, node: int) -> Dict[int, NProfile]:
        """
        Get the profiles of a node at all available time instants.

        :param node: Node ID for which to get the profiles.
        :return: A dictionary mapping time IDs to NProfile instances.

        Examples
        --------
        .. code-block:: python

            from ash_model.classes.node_profile import NProfile

            H = ASH()
            H.add_node(1, 0, attr_dict=NProfile(1, role='A'))
            H.add_node(1, 1, attr_dict=NProfile(1, role='B'))
            profiles = H.get_node_profiles_by_time(1)
            list(profiles.keys())  # [0, 1]

        """
        return {
            tid: self.get_node_profile(node, tid)
            for tid in self.node_presence(node, as_intervals=False)
        }

    def get_node_profile(self, node: int, tid: Optional[int] = None) -> NProfile:
        """
        Get the profile of a node.
        The profile contains the node's attributes at a specific time ID.
        If `tid` is None, the profile will be aggregated across all time IDs.

        :param node: Node ID for which to get the profile.
        :param tid: Time ID to filter the profile. If None, all time IDs are considered.
        :return: An NProfile instance containing the node's attributes.

        Examples
        --------
        .. code-block:: python

            from ash_model.classes.node_profile import NProfile

            H = ASH()
            H.add_node(1, 0, attr_dict=NProfile(1, team='X'))
            p0 = H.get_node_profile(1, 0)
            p_all = H.get_node_profile(1)  # aggregated profile

        """
        from ash_model.utils import aggregate_node_profile

        if tid is None:
            return aggregate_node_profile(self, node)
        return NProfile(node, **dict(self._node_attrs[node][tid]))  # type: ignore[arg-type]

    def get_node_attribute(
        self,
        node: int,
        attr_name: str,
        tid: Optional[int] = None,
    ) -> Union[Any, Dict[int, Any]]:
        """
        Get a specific attribute of a node at a given time.
        If `tid` is None, a dictionary mapping time IDs to attribute values is returned.

        :param node: Node ID for which to get the attribute.
        :param attr_name: Name of the attribute to retrieve.
        :param tid: Time ID to filter the attribute. If None, all time IDs are considered.
        :return: The value of the attribute for the node at the specified time, or a dictionary of values if `tid` is None.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_node(1, start=0, attr_dict={"label": "A"})
            H.get_node_attribute(1, "label", tid=0)  # 'A'
            H.get_node_attribute(1, "label")  # {0: 'A'}

        """

        if tid is None:
            return {
                t: self._node_attrs[node][t].get(attr_name, None)
                for t in self._node_attrs[node]
            }
        else:
            return self.get_node_attributes(node, tid).get(attr_name, None)

    def get_node_attributes(
        self, node: int, tid: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get all attributes of a node at a given time.
        If `tid` is None, returns attributes across all time IDs.

        :param node: Node ID for which to get the attributes.
        :param tid: Time ID to filter the attributes. If None, all time IDs are considered.
        :return: A dictionary of attributes for the node at the specified time, or across all times if `tid` is None.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_node(1, 0, attr_dict={"x": 10})
            H.get_node_attributes(1, 0)  # {'x': 10}
            H.get_node_attributes(1)     # {0: {'x': 10}}

        """

        if tid is None:
            return dict(self._node_attrs[node]) if node in self._node_attrs else {}
        else:
            return (
                dict(self._node_attrs[node][tid])
                if tid in self._node_attrs[node]
                else {}
            )

    def list_node_attributes(
        self, categorical: bool = False, tid: Optional[int] = None
    ) -> Dict[str, Set[Any]]:
        """
        List all attributes of nodes in the ASH, optionally filtered by time.
        If `tid` is None, all attributes across all times are considered.

        :param categorical: If True, only categorical attributes (strings) are returned.
        :param tid: Time ID to filter the attributes. If None, all time IDs are considered.
        :return: A dictionary where keys are attribute names and values are sets of attribute values.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_node(1, 0, attr_dict={"label": "A", "age": 30})
            H.add_node(2, 0, attr_dict={"label": "B"})
            all_attrs = H.list_node_attributes()
            cat_attrs = H.list_node_attributes(categorical=True)

        """
        attributes: DefaultDict[str, Set[Any]] = defaultdict(set)
        if tid is None:
            for node in self.nodes():
                for attr_dict in self._node_attrs[node].values():
                    for attr, value in attr_dict.items():
                        attributes[attr].add(value)
        else:
            for node in self.nodes(tid):
                for attr, value in self._node_attrs[node][tid].items():
                    attributes[attr].add(value)

        if categorical:
            attributes = defaultdict(
                set,
                {k: v for k, v in attributes.items() if isinstance(next(iter(v)), str)},
            )
        return dict(attributes)

    def get_hyperedge_attribute(self, hyperedge_id: str, attribute_name: str) -> Any:
        """
        Get a specific attribute of a hyperedge.

        :param hyperedge_id: ID of the hyperedge.
        :param attribute_name: Name of the attribute to retrieve.
        :return: The value of the attribute for the hyperedge, or None if not set

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_hyperedge([1, 2], start=0, weight=5)
            H.get_hyperedge_attribute('e1', 'weight')  # 5

        """
        if hyperedge_id not in self._edge_attributes:
            return None
        if attribute_name not in self._edge_attributes[hyperedge_id]:
            return None
        return self._edge_attributes[hyperedge_id][attribute_name]

    def get_hyperedge_attributes(
        self, hyperedge_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get all attributes of a hyperedge.
        If `hyperedge_id` is None, returns attributes for all hyperedges.
        If the hyperedge does not exist, returns an empty dictionary.

        :param hyperedge_id: ID of the hyperedge. If None, all hyperedges are considered.

        :return: A dictionary of attributes for the hyperedge, or an empty dictionary if not found.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_hyperedges([[1, 2], [2, 3]], start=0, weight=1)
            attrs_all = H.get_hyperedge_attributes()
            attrs_e1 = H.get_hyperedge_attributes('e1')

        """
        if hyperedge_id is None:
            return {he: attrs for he, attrs in self._edge_attributes.items()}
        return dict(self._edge_attributes[hyperedge_id])

    def list_hyperedge_attributes(
        self, categorical: bool = False
    ) -> Dict[str, Set[Any]]:
        """
        List all attributes of hyperedges in the ASH.
        If `categorical` is True, only categorical attributes (strings) are returned.

        :param categorical: If True, only categorical attributes (strings) are returned.
        :return: A dictionary where keys are attribute names and values are sets of attribute values.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_hyperedge([1, 2], start=0, color='blue')
            H.list_hyperedge_attributes()  # {'weight': {1}, 'color': {'blue'}}

        """

        attributes: DefaultDict[str, Set[Any]] = defaultdict(set)

        for hedge in self.hyperedges():
            for attr, value in self._edge_attributes[hedge].items():
                attributes[attr].add(value)

        if categorical:
            attributes = defaultdict(
                set,
                {k: v for k, v in attributes.items() if isinstance(next(iter(v)), str)},
            )
        return dict(attributes)

    def get_hyperedge_weight(self, hyperedge_id: str) -> Union[int, float]:
        """
        Get the weight of a hyperedge.
        If the weight is not set, returns 1 by default.

        :param hyperedge_id: ID of the hyperedge.
        :return: The weight of the hyperedge, or 1 if not set.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_hyperedge([1, 2], start=0)
            H.get_hyperedge_weight('e1')  # 1

        """
        weight = self.get_hyperedge_attribute(hyperedge_id, "weight")
        return 1 if weight is None else weight

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def number_of_nodes(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> int:
        """
            Return the number of unique nodes present in the ASH within the specified time window.

        :param start: Start time of the query. If None, all nodes are considered.
        :param end: End time of the query (inclusive). If None, only the start time is considered.
        :return: The number of unique nodes.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_nodes([1, 2, 3], start=0)
            H.number_of_nodes(0)  # 3

        """

        return len(self.nodes(start, end))

    def number_of_hyperedges(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> int:
        """
        Return the number of unique hyperedges present in the ASH within the specified time window.

        :param start: Start time of the query. If None, all hyperedges are considered
        :param end: End time of the query (inclusive). If None, only the start time is considered.
        :return: The number of unique hyperedges.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_hyperedges([[1, 2], [2, 3]], start=0)
            H.number_of_hyperedges(0)  # 2

        """
        return self.size(start, end)

    def size(self, start: Optional[int] = None, end: Optional[int] = None) -> int:
        """
            Return the number of hyperedges present in the ASH within the specified time window.

        :param start: Start time of the query. If None, all hyperedges are considered.
        :param end: End time of the query (inclusive). If None, only the start time is considered.
        :return: The number of hyperedges.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_hyperedge([1, 2], start=0)
            H.size(0)  # 1

        """

        return len(self.hyperedges(start, end))

    def hyperedge_size_distribution(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> Dict[int, int]:
        """
            Return the distribution of hyperedge sizes within the specified time window.
            The keys are the sizes of hyperedges (number of nodes in each hyperedge),
            and the values are the counts of hyperedges of that size.

        :param start: Start time of the query. If None, all hyperedges are considered.
        :param end: End time of the query (inclusive). If None, only the start time is considered.
        :return: A dictionary where keys are hyperedge sizes and values are counts of hyperedges of that size.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_hyperedges([[1, 2], [2, 3, 4]], start=0)
            H.hyperedge_size_distribution(0)  # {2: 1, 3: 1}

        """

        distr: DefaultDict[int, int] = defaultdict(int)
        for he in self.hyperedges(start, end):
            distr[len(self.get_hyperedge_nodes(he))] += 1
        return dict(distr)

    def degree_distribution(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> Dict[int, int]:
        """
            Return the distribution of node degrees within the specified time window.
            The keys are the degrees of nodes (number of hyperedges each node is part of),
            and the values are the counts of nodes with that degree.

        :param start: Start time of the query. If None, all nodes are considered.
        :param end: End time of the query (inclusive). If None, only the start time is considered.
        :return: A dictionary where keys are node degrees and values are counts of nodes with that degree.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_nodes([1, 2, 3], start=0)
            H.add_hyperedges([[1, 2], [2, 3]], start=0)
            H.degree_distribution(0)  # e.g., {1: 2, 2: 1}

        """

        distr: DefaultDict[int, int] = defaultdict(int)
        for node in self.nodes(start, end):
            distr[self.degree(node, start, end)] += 1
        return dict(distr)

    # ------------------------------------------------------------------
    # Node-centric analysis
    # ------------------------------------------------------------------

    def star(
        self,
        node: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
        hyperedge_size: Optional[int] = None,
        as_ids: bool = True,
    ) -> List[Union[str, frozenset[int]]]:
        """
            Return the star of a node, which is the set of hyperedges that contain the node
            within the specified time window. If `hyperedge_size` is specified, only hyperedges
            of that size are returned. The `as_ids` parameter determines whether to return hyperedge
            IDs (strings) or sets of node IDs (frozensets).

        :param node: Node ID for which to get the star.
        :param start: Start time of the query. If None, all hyperedges are considered.
        :param end: End time of the query (inclusive). If None, only the start time is considered.
        :param hyperedge_size: If specified, only hyperedges of this size are returned.
        :param as_ids: If True, return hyperedge IDs; if False, return sets of node IDs.

        :return: A list of hyperedge IDs or sets of node IDs that form the star of the node within the specified time window.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_nodes([1, 2, 3], start=0)
            H.add_hyperedges([[1, 2], [1, 3]], start=0)
            H.star(1, 0)  # e.g., ['e1', 'e2']
            H.star(1, 0, as_ids=False)  # [frozenset({1,2}), frozenset({1,3})]

        """

        if start is None:
            star_set: Set[str] = set(self._stars[node])
        else:
            star_set = set()
            for t in self.__time_window(start, end):
                star_set.update(self._stars[node].intersection(self._snapshots[t]))

        if hyperedge_size is not None:
            star_set = {
                s
                for s in star_set
                if len(self.get_hyperedge_nodes(s)) == hyperedge_size
            }

        if as_ids:
            return list(star_set)
        return [self.get_hyperedge_nodes(s) for s in star_set]

    def degree(
        self,
        node: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
        hyperedge_size: Optional[int] = None,
    ) -> int:
        """
            Return the degree of a node, which is the number of hyperedges that contain the node
            within the specified time window. If `hyperedge_size` is specified, only hyperedges
            of that size are counted.

        :param node: Node ID for which to get the degree.
        :param start: Start time of the query. If None, all hyperedges are considered.
        :param end: End time of the query (inclusive). If None, only the start time is considered.
        :param hyperedge_size: If specified, only hyperedges of this size are counted.

        :return: The degree of the node within the specified time window.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_nodes([1, 2, 3], start=0)
            H.add_hyperedges([[1, 2], [1, 3]], start=0)
            H.degree(1, 0)  # 2

        """

        return len(self.star(node, start, end, hyperedge_size, as_ids=False))

    def s_degree(
        self, node: int, s: int, start: Optional[int] = None, end: Optional[int] = None
    ) -> int:
        """
        Compute the s-degree of a node, summing degrees for hyperedges of size at least s.

        :param node: Node ID.
        :param s: Minimum hyperedge size to consider.
        :param start: Start time of the query. If None, all hyperedges are considered.
        :param end: End time of the query (inclusive). If None, only the start time is considered.
        :return: The s-degree value.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_nodes([1, 2, 3, 4], start=0)
            H.add_hyperedges([[1, 2], [1, 2, 3], [1, 3, 4]], start=0)
            H.s_degree(1, s=3, start=0)  # counts only size >= 3 -> 2

        """
        degs = self.degree_by_hyperedge_size(node, start, end)
        return sum(v for k, v in degs.items() if k >= s)

    def degree_by_hyperedge_size(
        self, node: int, start: Optional[int] = None, end: Optional[int] = None
    ) -> Dict[int, int]:
        """
            Return the degree of a node by hyperedge size, which is a dictionary where keys are
            hyperedge sizes (number of nodes in each hyperedge) and values are the counts of
            hyperedges of that size that contain the node within the specified time window.

        :param node: Node ID for which to get the degree by hyperedge size.
        :param start: Start time of the query. If None, all hyperedges are considered.
        :param end: End time of the query (inclusive). If None, only the start time is considered.

        :return: A dictionary where keys are hyperedge sizes and values are counts of hyperedges of that size that contain the node within the specified time window.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_nodes([1, 2, 3], start=0)
            H.add_hyperedges([[1, 2], [1, 2, 3]], start=0)
            H.degree_by_hyperedge_size(1, 0)  # {2: 1, 3: 1}

        """

        distr: DefaultDict[int, int] = defaultdict(int)
        for he in self.star(node, start, end, as_ids=False):
            distr[len(he)] += 1
        return dict(distr)

    def neighbors(
        self,
        node: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
        hyperedge_size: Optional[int] = None,
    ) -> Set[int]:
        """
            Return the set of neighbors of a node, which are the nodes that share hyperedges with
            the specified node within the given time window. If `hyperedge_size` is specified,
            only hyperedges of that size are considered for determining neighbors.

        :param node: Node ID for which to get the neighbors.
        :param start: Start time of the query. If None, all hyperedges are considered.
        :param end: End time of the query (inclusive). If None, only the start time is considered.
        :param hyperedge_size: If specified, only hyperedges of this size are considered for determining neighbors.

        :return: A set of node IDs that are neighbors of the specified node within the specified time window.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_nodes([1, 2, 3], start=0)
            H.add_hyperedges([[1, 2], [1, 3]], start=0)
            H.neighbors(1, 0)  # {2, 3}

        """

        neighbors: Set[int] = set()
        for he in self.star(node, start, end, hyperedge_size, as_ids=False):
            neighbors.update(he)
        neighbors.discard(node)
        return neighbors

    def number_of_neighbors(
        self,
        node: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
        hyperedge_size: Optional[int] = None,
    ) -> int:
        """
            Return the number of unique neighbors of a node, which are the nodes that share hyperedges
            with the specified node within the given time window. If `hyperedge_size` is specified
            only hyperedges of that size are considered for determining neighbors.

        :param node: Node ID for which to get the number of neighbors.
        :param start: Start time of the query. If None, all hyperedges are considered.
        :param end: End time of the query (inclusive). If None, only the start time is considered.
        :param hyperedge_size: If specified, only hyperedges of this size are considered for determining neighbors.

        :return: The number of unique neighbors of the specified node within the specified time window.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_nodes([1, 2, 3], start=0)
            H.add_hyperedges([[1, 2], [1, 3]], start=0)
            H.number_of_neighbors(1, 0)  # 2

        """

        return len(self.neighbors(node, start, end, hyperedge_size))

    # ------------------------------------------------------------------
    # Transformations & projections
    # ------------------------------------------------------------------

    def bipartite_projection(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        keep_attrs: bool = False,
    ) -> nx.Graph:
        """
            Create a bipartite projection of the ASH.

        :param start: Start time of the projection. If None, all hyperedges are considered.
        :param end: End time of the projection (inclusive). If None, only the start time is considered.
        :param keep_attrs: If True, keep node and hyperedge attributes in the projection.

        :return: A bipartite graph where nodes are hyperedges and nodes, and edges represent incidences between them.

        Examples
        --------
        .. code-block:: python

            import networkx as nx
            H = ASH()
            H.add_hyperedge([1, 2], start=0)
            G = H.bipartite_projection(0)
            isinstance(G, nx.Graph)  # True

        """

        from ash_model.utils import bipartite_projection

        return bipartite_projection(self, start, end, keep_attrs)

    def dual_hypergraph(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> Tuple["ASH", Dict[str, str]]:
        """
            Create a dual hypergraph projection of the ASH.

        :param start: Start time of the projection. If None, all hyperedges are considered.
        :param end: End time of the projection (inclusive). If None, only the start time is considered.

        :return: A tuple containing the dual hypergraph and a mapping of original hyperedge IDs to new hyperedge IDs in the dual hypergraph.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_hyperedges([[1, 2], [2, 3]], start=0)
            dual, mapping = H.dual_hypergraph(0)
            isinstance(dual, ASH)  # True

        """

        from ash_model.utils import dual_hypergraph_projection

        return dual_hypergraph_projection(self, start, end)

    def clique_projection(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        keep_attrs: bool = False,
    ) -> nx.Graph:
        """
            Create a clique projection of the ASH.

        :param start: Start time of the projection. If None, all hyperedges are considered.
        :param end: End time of the projection (inclusive). If None, only the start time is considered.
        :param keep_attrs: If True, keep node and hyperedge attributes in the projection.

        :return: A graph where each hyperedge is decomposed into a clique.

        Examples
        --------
        .. code-block:: python

            import networkx as nx
            H = ASH()
            H.add_hyperedge([1, 2, 3], start=0)
            G = H.clique_projection(0)
            isinstance(G, nx.Graph)  # True

        """

        from ash_model.utils import clique_projection

        return clique_projection(self, start, end, keep_attrs=keep_attrs)

    def s_line_graph(
        self, s: int = 1, start: Optional[int] = None, end: Optional[int] = None
    ) -> nx.Graph:
        """
            Create a line graph projection of the ASH, where each hyperedge is represented as a
            node, and edges represent shared nodes between hyperedges.

        :param s: Minimum size of hyperedges to consider in the projection.
        :param start: Start time of the projection. If None, all hyperedges are considered.
        :param end: End time of the projection (inclusive). If None, only the start time is considered.

        :return: A line graph where nodes represent hyperedges and edges represent shared nodes.

        Examples
        --------
        .. code-block:: python

            import networkx as nx
            H = ASH()
            H.add_hyperedges([[1, 2], [2, 3]], start=0)
            LG = H.s_line_graph(s=1, start=0)
            isinstance(LG, nx.Graph)  # True

        """

        from ash_model.utils import line_graph_projection

        return line_graph_projection(self, s, start, end)

    # ------------------------------------------------------------------
    # Temporal analysis
    # ------------------------------------------------------------------

    def avg_number_of_nodes(self) -> float:
        """
        Calculate the average number of nodes across all temporal snapshots.

        :return: The average number of nodes.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_node(1, 0)
            H.add_node(2, 1)
            avg = H.avg_number_of_nodes()  # e.g., 1.0

        """
        nodes_snapshots = [
            self.number_of_nodes(tid) for tid in self.temporal_snapshots_ids()
        ]
        return sum(nodes_snapshots) / len(nodes_snapshots) if nodes_snapshots else 0.0

    def avg_number_of_hyperedges(self) -> float:
        """
        Calculate the average number of hyperedges across all temporal snapshots.

        :return: The average number of hyperedges.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_hyperedge([1, 2], 0)
            H.add_hyperedge([2, 3], 1)
            H.avg_number_of_hyperedges()  # e.g., 1.0

        """

        hes_snapshots = [self.size(tid) for tid in self.temporal_snapshots_ids()]
        return sum(hes_snapshots) / len(hes_snapshots) if hes_snapshots else 0.0

    def node_presence(
        self, node: int, as_intervals: bool = False
    ) -> List[Union[int, Tuple[int, int]]]:
        """
            Get the presence of a node across all temporal snapshots.
            If `as_intervals` is True, returns the presence as intervals (start, end
            times). Otherwise, returns a list of time IDs where the node is present.

        :param node: Node ID for which to get the presence.
        :param as_intervals: If True, return presence as intervals; if False, return presence as a list of time IDs.

        :return: A list of time IDs or intervals where the node is present.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_node(1, start=0, end=2)
            H.node_presence(1)           # [0, 1, 2]
            H.node_presence(1, True)     # [(0, 2)]

        """
        times = sorted(self._node_attrs[node].keys())
        return (
            self.__presence_to_intervals(times) if as_intervals else times  # type: ignore[return-value]
        )

    def hyperedge_presence(
        self, hyperedge_id: str, as_intervals: bool = False
    ) -> List[Union[int, Tuple[int, int]]]:
        """
            Get the presence of a hyperedge across all temporal snapshots.
            If `as_intervals` is True, returns the presence as intervals (start, end
            times). Otherwise, returns a list of time IDs where the hyperedge is present.

        :param hyperedge_id: ID of the hyperedge for which to get the presence.
        :param as_intervals: If True, return presence as intervals; if False, return presence as a list of time IDs.

        :return: A list of time IDs or intervals where the hyperedge is present.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_hyperedge([1, 2], start=0, end=2)
            H.hyperedge_presence('e1')       # [0, 1, 2]
            H.hyperedge_presence('e1', True) # [(0, 2)]

        """

        pres = [
            t
            for t in self.temporal_snapshots_ids()
            if self.has_hyperedge(hyperedge_id, t)
        ]
        return (
            self.__presence_to_intervals(pres) if as_intervals else pres  # type: ignore[return-value]
        )

    def node_contribution(self, node: int) -> float:
        """
        Calculate the contribution of a node to the ASH, defined as the fraction of temporal snapshots
        in which the node is present.

        :param node: Node ID for which to calculate the contribution.
        :return: The contribution of the node as a float between 0 and 1.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            # Node 1 present in 1 out of 2 snapshots
            H.add_node(1, 0)
            H.add_node(2, 1)
            H.node_contribution(1)  # 0.5

        """

        total_snapshots = len(self.temporal_snapshots_ids())
        if total_snapshots == 0:
            return 0.0
        return (
            sum(1 for tid in self.temporal_snapshots_ids() if self.has_node(node, tid))
            / total_snapshots
        )

    def hyperedge_contribution(self, hyperedge_id: str) -> float:
        """
        Calculate the contribution of a hyperedge to the ASH, defined as the fraction of temporal
        snapshots in which the hyperedge is present.

        :param hyperedge_id: ID of the hyperedge for which to calculate the contribution.
        :return: The contribution of the hyperedge as a float between 0 and 1

        Examples
        --------
        .. code-block:: python

            H = ASH()
            # Edge present in 1 out of 2 snapshots
            H.add_hyperedge([1, 2], 0)
            H.add_node(3, 1)
            H.hyperedge_contribution('e1')  # 0.5

        """

        total_snapshots = len(self.temporal_snapshots_ids())
        if total_snapshots == 0:
            return 0.0
        return (
            sum(
                1
                for tid in self.temporal_snapshots_ids()
                if self.has_hyperedge(hyperedge_id, tid)
            )
            / total_snapshots
        )

    def coverage(self) -> float:
        """
        Calculate the coverage of the ASH, defined as the average number of nodes present across all
        temporal snapshots, normalized by the total number of nodes.

        :return: The coverage of the ASH as a float between 0 and 1

        Examples
        --------
        .. code-block:: python

            # Two snapshots, 2 nodes total; average present per snapshot = 1.0 -> coverage 0.5
            H = ASH()
            H.add_node(1, 0)
            H.add_node(2, 1)
            H.coverage()  # 0.5

        """

        tids = self.temporal_snapshots_ids()
        if not tids or self.number_of_nodes() == 0:
            return 0.0
        w = sum(self.number_of_nodes(tid) for tid in tids)
        return w / (len(tids) * self.number_of_nodes())

    def uniformity(self) -> float:
        """
        Calculate the uniformity of the ASH, defined as the fraction of pairs of nodes that
        are present together in at least one temporal snapshot, normalized by the total number of pairs.

        :return: The uniformity of the ASH as a float between 0 and 1

        Examples
        --------
        .. code-block:: python

            # Nodes (1,2) co-present at t=0; (1,3) never; (2,3) co-present at t=1
            H = ASH()
            H.add_node(1, 0)
            H.add_node(2, 0)
            H.add_node(2, 1)
            H.add_node(3, 1)
            u = H.uniformity()  # between 0 and 1

        """

        nds = self.nodes()
        numerator, denominator = 0, 0
        for u, v in combinations(nds, 2):
            for t in self.temporal_snapshots_ids():
                u_present = self.has_node(u, t)
                v_present = self.has_node(v, t)
                if u_present and v_present:
                    numerator += 1
                if u_present or v_present:
                    denominator += 1
        return numerator / denominator if denominator else 0.0

    # ------------------------------------------------------------------
    # Slice & filter
    # ------------------------------------------------------------------

    def temporal_slice(
        self, start: int, end: Optional[int] = None, keep_attrs: bool = True
    ) -> Tuple["ASH", Dict[str, str]]:
        """
            Create a temporal slice of the ASH from `start` to `end` (inclusive).
            If `end` is None, the slice will only include the time `start`.
            The `keep_attrs` parameter determines whether to keep node attributes in the slice.

        :param start: Start time of the slice.
        :param end: End time of the slice (inclusive). If None, the slice will only include the time `start`.
        :param keep_attrs: If True, node attributes are preserved in the slice.

        :return: A tuple containing the new ASH and a mapping from old hyperedge IDs to new hyperedge IDs.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_hyperedge([1, 2], start=0)
            H.add_hyperedge([2, 3], start=1)
            sub, mapping = H.temporal_slice(0)
            isinstance(sub, ASH)  # True

        """

        res = ASH()
        eid_to_new_eid: Dict[str, str] = {}
        if end is None:
            end = start
        for t in range(start, end + 1):
            for hedge_id in self.hyperedges(t):
                nodes = self.get_hyperedge_nodes(hedge_id)
                res.add_hyperedge(nodes, t, t)
                eid_to_new_eid[hedge_id] = res._nids2eid[frozenset(nodes)]

        if keep_attrs:
            for t in range(start, end + 1):
                for node in self.nodes(t):
                    profile = self.get_node_profile(node, t)
                    res.add_node(node, t, t, profile)
        return res, eid_to_new_eid

    def induced_hypergraph(
        self, hyperedge_set: Iterable[str], keep_attrs: bool = True
    ) -> Tuple["ASH", Dict[str, str]]:
        """
            Create an induced hypergraph from the ASH using a set of hyperedge IDs.
            The `keep_attrs` parameter determines whether to keep node attributes in the new hypergraph.

        :param hyperedge_set: Iterable of hyperedge IDs to include in the induced hypergraph.
        :param keep_attrs: If True, node attributes are preserved in the new hypergraph.

        :return: A tuple containing the new ASH and a mapping from old hyperedge IDs to new hyperedge IDs.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_hyperedges([[1, 2], [2, 3]], start=0)
            sub, mapping = H.induced_hypergraph(['e1'])
            set(sub.hyperedges())  # {'e1'} in subgraph namespace

        """

        b = ASH()
        old_eid_to_new: Dict[str, str] = {}

        for he in hyperedge_set:
            presence = self.hyperedge_presence(he, as_intervals=True)  # type: ignore[arg-type]
            nodes = self.get_hyperedge_nodes(he)
            for span in presence:  # type: ignore[assignment]
                b.add_hyperedge(nodes, span[0], span[1])  # type: ignore[index]
            new_he = b.get_hyperedge_id(nodes)
            old_eid_to_new[he] = new_he

            if keep_attrs:
                for t in b.temporal_snapshots_ids():
                    for node in b.nodes(t):
                        profile = self.get_node_profile(node, t)
                        b.add_node(node, start=t, end=t, attr_dict=profile)

        return b, old_eid_to_new

    def get_s_incident(
        self,
        hyperedge_id: str,
        s: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> List[Tuple[str, int]]:
        """
            Get hyperedges that are incident to a given hyperedge with at least `s` nodes
            in common within the specified time window. Returns a list of tuples containing
            hyperedge IDs and the number of nodes they share with the specified hyperedge.

        :param hyperedge_id: ID of the hyperedge to check against.
        :param s: Minimum number of nodes that must be shared with the specified hyperedge.
        :param start: Start time of the query. If None, all hyperedges are considered.
        :param end: End time of the query (inclusive). If None, only the start time is considered.

        :return: A list of tuples where each tuple contains a hyperedge ID and the number of nodes it shares with the specified hyperedge.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_hyperedges([[1, 2, 3], [1, 3, 4], [4, 5]], start=0)
            H.get_s_incident('e1', s=2, start=0)  # [('e2', 2)]

        """

        res: List[Tuple[str, int]] = []
        nodes = set(self.get_hyperedge_nodes(hyperedge_id))
        hes = self.hyperedges(start, end)
        for he in hes:
            if he != hyperedge_id:
                he_nodes = set(self.get_hyperedge_nodes(he))
                incident = len(nodes & he_nodes)
                if incident >= s:
                    res.append((he, incident))
        return res

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the ASH to a dictionary representation.
        This includes nodes, hyperedges, and their attributes.

        :return: A dictionary representation of the ASH.

        Examples
        --------
        .. code-block:: python

            H = ASH()
            H.add_hyperedge([1, 2], start=0)
            d = H.to_dict()
            set(d.keys())  # {'nodes', 'hedges'}

        """
        return self.__dict__()

    def __dict__(self) -> Dict[str, Any]:

        descr: Dict[str, Any] = {"nodes": {}, "hedges": {}}

        for hedge in self.hyperedges():
            edge_data: Dict[str, Any] = {
                "nodes": list(self.get_hyperedge_nodes(hedge)),
                "attributes": self.get_hyperedge_attributes(hedge),
            }
            edge_data["attributes"]["_presence"] = self.hyperedge_presence(
                hedge, as_intervals=True  # type: ignore[arg-type]
            )
            descr["hedges"][hedge] = edge_data

        for node in self.nodes():
            descr["nodes"][node] = self.get_node_attributes(node)

        return descr

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        descr = "Attributed Stream Hypergraph\n"
        descr += f"Nodes: {self.number_of_nodes()}\n"
        descr += f"Hyperedges: {self.number_of_hyperedges()}\n"
        descr += f"Snapshots: {len(self.temporal_snapshots_ids())}\n"
        descr += "Node attributes: " + ", ".join(self.list_node_attributes()) + "\n"
        return descr
