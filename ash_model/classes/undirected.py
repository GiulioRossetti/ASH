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
        edge_attributes: bool = False,
        backend: str = "dense",
    ) -> None:
        """
        :param edge_attributes: Whether to allow attributes for hyperedges
        """

        if backend == "dense":
            self._snapshots: PresenceStore = DensePresenceStore()
        elif backend == "interval":
            self._snapshots = IntervalPresenceStore()
        else:  # defensive – easier to spot typos
            raise ValueError(
                "backend must be 'dense' or 'interval' (got %r)"
                % backend
            )


        # edge data
        self._current_hyperedge_id: int = 0
        self._eid2nids: Dict[str, frozenset[int]] = {}
        self._nids2eid: Dict[frozenset[int], str] = {}
        self.is_edge_attr: bool = edge_attributes
        if self.is_edge_attr:
            self._edge_attributes: DefaultDict[
                str, DefaultDict[str, Dict[int, Any]]
            ] = defaultdict(lambda: defaultdict(dict))

        # node data
        self._node_attrs: DefaultDict[
            int, DefaultDict[int, Dict[str, Any]]
        ] = defaultdict(lambda: defaultdict(dict))
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
    
    def __time_window(
        self,
        start: Optional[int], end: Optional[int]
    ) -> List[int]:
        if start is None:
            return self.temporal_snapshots_ids()
        if end is None:
            end = start
        return list(range(start, end + 1))

    # ------------------------------------------------------------------
    # Iterators / Streams
    # ------------------------------------------------------------------

    def temporal_snapshots_ids(self) -> List[int]:
        return sorted(self._snapshots.keys())

    def stream_interactions(self) -> Generator[Tuple[int, str, str], None, None]:
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
        if kwargs and not self.is_edge_attr:
            raise ValueError("Edge attributes are not enabled")

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
        for t in range(span[0], span[1] + 1):
            self._snapshots.setdefault(t, set()).add(hid)

        # Store attributes -------------------------------------------------
        if self.is_edge_attr and kwargs:
            for k, v in kwargs.items():
                self._edge_attributes[hid][k][start] = v

    def add_hyperedges(
        self,
        hyperedges: Iterable[Iterable[int]],
        start: int,
        end: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
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
        time_window = set(self.__time_window(start, end))
        still_exists = False
        for t in self.temporal_snapshots_ids():
            if t in time_window:
                self._snapshots.setdefault(t, set()).discard(hyperedge_id)
                if self.is_edge_attr:
                    self._edge_attributes[hyperedge_id].pop(t, None)
            elif self.has_hyperedge(hyperedge_id, t):
                still_exists = True

        if not still_exists:
            nodes = self.get_hyperedge_nodes(hyperedge_id)
            del self._eid2nids[hyperedge_id]
            del self._nids2eid[nodes]
            if self.is_edge_attr:
                self._edge_attributes.pop(hyperedge_id, None)
            for n in nodes:
                self._stars[n].discard(hyperedge_id)

    def remove_hyperedges(
        self,
        hyperedges: Iterable[str],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        for hid in hyperedges:
            self.remove_hyperedge(hid, start, end)

    def remove_node(
        self,
        node: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
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
        for n in nodes:
            self.remove_node(n, start, end)

    def remove_unlabelled_nodes(
        self,
        attr_name: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        for node, t_attrs in list(self._node_attrs.items()):
            for t in self.__time_window(start, end):
                if t in t_attrs and attr_name not in t_attrs[t]:
                    self.remove_node(node, t, t)

    def remove_unlabelled_nodes(
        self, attr_name: str, start: Optional[int] = None, end: Optional[int] = None
    ) -> None:
        time_window = self.__time_window(start, end)
        for node, t_attrs in list(self._node_attrs.items()):
            for t in time_window:
                if t in t_attrs and attr_name not in t_attrs[t]:
                    self.remove_node(node, t, t)

    # ------------------------------------------------------------------
    # Node & Hyperedge queries
    # ------------------------------------------------------------------

    def nodes(self, start: Optional[int] = None, end: Optional[int] = None) -> List[int]:
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
        return node in set(self.nodes(start, end))

    def get_hyperedge_nodes(self, hyperedge_id: str) -> frozenset[int]:
        return self._eid2nids.get(hyperedge_id, frozenset())

    def get_hyperedge_id(self, nodes: Iterable[int]) -> str:
        return self._nids2eid[frozenset(nodes)]

    # ------------------------------------------------------------------
    # Node & Hyperedge attributes
    # ------------------------------------------------------------------

    def get_node_profile(self, node: int, tid: Optional[int] = None) -> NProfile:
        if tid is None:
            attr_dict: DefaultDict[str, Dict[int, Any]] = defaultdict(dict)
            for t in self._node_attrs[node]:
                for attr, value in self._node_attrs[node][t].items():
                    attr_dict[attr][t] = value
            return NProfile(node, **attr_dict)  # type: ignore[arg-type]
        return NProfile(node, **dict(self._node_attrs[node][tid]))  # type: ignore[arg-type]

    def get_node_attribute(
        self,
        node: int,
        attr_name: str,
        tid: Optional[int] = None,
    ) -> Any:
        return self.get_node_profile(node, tid=tid).get_attribute(attr_name)

    def get_node_attributes(self, node: int, tid: Optional[int] = None) -> Dict[str, Any]:
        return self.get_node_profile(node, tid).get_attributes()

    def list_node_attributes(
        self, categorical: bool = False, tid: Optional[int] = None
    ) -> Dict[str, Set[Any]]:
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
                {
                    k: v
                    for k, v in attributes.items()
                    if isinstance(next(iter(v)), str)
                },
            )
        return attributes  

    def get_hyperedge_attribute(
        self, hyperedge_id: str, attribute_name: str, tid: Optional[int] = None
    ) -> Any:
        if not self.is_edge_attr:
            return None
        if tid is None:
            return self._edge_attributes[hyperedge_id][attribute_name]
        return self._edge_attributes[hyperedge_id][attribute_name][tid]

    def get_hyperedge_attributes(
        self, hyperedge_id: str, tid: Optional[int] = None
    ) -> Dict[str, Any]:
        if not self.is_edge_attr:
            return {}
        if tid is None:
            return self._edge_attributes[hyperedge_id]
        return self._edge_attributes[hyperedge_id][tid]

    def list_hyperedge_attributes(
        self, categorical: bool = False, tid: Optional[int] = None
    ) -> Dict[str, Set[Any]]:
        attributes: DefaultDict[str, Set[Any]] = defaultdict(set)
        if not self.is_edge_attr:
            return attributes  # empty

        if tid is None:
            for hedge in self.hyperedges():
                for attr, value in self._edge_attributes[hedge].items():
                    attributes[attr].update(value.values())
        else:
            for hedge in self.hyperedges(tid):
                for attr, value in self._edge_attributes[hedge].items():
                    if tid in value:
                        attributes[attr].add(value[tid])

        if categorical:
            attributes = defaultdict(
                set,
                {
                    k: v
                    for k, v in attributes.items()
                    if isinstance(next(iter(v)), str)
                },
            )
        return attributes  # type: ignore[return-value]

    def get_hyperedge_weight(self, hyperedge_id: str) -> Union[int, float]:
        weight = self.get_hyperedge_attribute(hyperedge_id, "weight")
        return 1 if weight is None else weight

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def number_of_nodes(self, start: Optional[int] = None, end: Optional[int] = None) -> int:
        return len(self.nodes(start, end))

    def number_of_hyperedges(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> int:
        return self.size(start, end)

    def size(self, start: Optional[int] = None, end: Optional[int] = None) -> int:
        return len(self.hyperedges(start, end))

    def hyperedge_size_distribution(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> Dict[int, int]:
        distr: DefaultDict[int, int] = defaultdict(int)
        for he in self.hyperedges(start, end):
            distr[len(self.get_hyperedge_nodes(he))] += 1
        return dict(distr)

    def degree_distribution(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> Dict[int, int]:
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
        if start is None:
            star_set: Set[str] = set(self._stars[node])
        else:
            star_set = set()
            for t in self.__time_window(start, end):
                star_set.update(self._stars[node].intersection(self._snapshots[t]))

        if hyperedge_size is not None:
            star_set = {
                s for s in star_set if len(self.get_hyperedge_nodes(s)) == hyperedge_size
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
        return len(self.star(node, start, end, hyperedge_size, as_ids=False))

    def s_degree(
        self, node: int, s: int, start: Optional[int] = None, end: Optional[int] = None
    ) -> int:
        degs = self.degree_by_hyperedge_size(node, start, end)
        return sum(v for k, v in degs.items() if k >= s)

    def degree_by_hyperedge_size(
        self, node: int, start: Optional[int] = None, end: Optional[int] = None
    ) -> Dict[int, int]:
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
        return len(self.neighbors(node, start, end, hyperedge_size))

    # ------------------------------------------------------------------
    # Transformations & projections
    # ------------------------------------------------------------------

    def bipartite_projection(
        self, start: Optional[int] = None, end: Optional[int] = None, keep_attrs: bool = False
    ) -> nx.Graph:
        from ash_model.utils import bipartite_projection

        return bipartite_projection(self, start, end, keep_attrs)

    def dual_hypergraph(
        self, start: Optional[int] = None, end: Optional[int] = None
    ) -> Tuple["ASH", Dict[str, str]]:
        from ash_model.utils import dual_hypergraph_projection

        return dual_hypergraph_projection(self, start, end)

    def clique_projection(
        self, start: Optional[int] = None, end: Optional[int] = None, keep_attrs: bool = False
    ) -> nx.Graph:
        from ash_model.utils import clique_projection

        return clique_projection(self, start, end, keep_attrs=keep_attrs)

    def s_line_graph(
        self, s: int = 1, start: Optional[int] = None, end: Optional[int] = None
    ) -> nx.Graph:
        from ash_model.utils import line_graph_projection

        return line_graph_projection(self, s, start, end)

    # ------------------------------------------------------------------
    # Temporal analysis
    # ------------------------------------------------------------------

    def avg_number_of_nodes(self) -> float:
        nodes_snapshots = [self.number_of_nodes(tid) for tid in self.temporal_snapshots_ids()]
        return sum(nodes_snapshots) / len(nodes_snapshots) if nodes_snapshots else 0.0

    def avg_number_of_hyperedges(self) -> float:
        hes_snapshots = [self.size(tid) for tid in self.temporal_snapshots_ids()]
        return sum(hes_snapshots) / len(hes_snapshots) if hes_snapshots else 0.0

    def node_presence(
        self, node: int, as_intervals: bool = False
    ) -> List[Union[int, Tuple[int, int]]]:
        times = sorted(self._node_attrs[node].keys())
        return (
            self.__presence_to_intervals(times) if as_intervals else times  # type: ignore[return-value]
        )

    def hyperedge_presence(
        self, hyperedge_id: str, as_intervals: bool = False
    ) -> List[Union[int, Tuple[int, int]]]:
        pres = [t for t in self.temporal_snapshots_ids() if self.has_hyperedge(hyperedge_id, t)]
        return (
            self.__presence_to_intervals(pres) if as_intervals else pres  # type: ignore[return-value]
        )

    def node_contribution(self, node: int) -> float:
        total_snapshots = len(self.temporal_snapshots_ids())
        if total_snapshots == 0:
            return 0.0
        return sum(1 for tid in self.temporal_snapshots_ids() if self.has_node(node, tid)) / total_snapshots

    def hyperedge_contribution(self, hyperedge_id: str) -> float:
        total_snapshots = len(self.temporal_snapshots_ids())
        if total_snapshots == 0:
            return 0.0
        return sum(
            1 for tid in self.temporal_snapshots_ids() if self.has_hyperedge(hyperedge_id, tid)
        ) / total_snapshots

    def coverage(self) -> float:
        tids = self.temporal_snapshots_ids()
        if not tids or self.number_of_nodes() == 0:
            return 0.0
        w = sum(self.number_of_nodes(tid) for tid in tids)
        return w / (len(tids) * self.number_of_nodes())

    def uniformity(self) -> float:
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
            npr = self.get_node_profile(node)
            descr["nodes"][node] = npr.get_attributes()
            descr["nodes"][node]["_presence"] = self.node_presence(
                node, as_intervals=True  # type: ignore[arg-type]
            )

        descr["is_edge_attr"] = self.is_edge_attr
        descr["current_eid"] = self._current_hyperedge_id
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
