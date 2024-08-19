import warnings
from collections import defaultdict
from itertools import combinations

import networkx as nx

from .node_profile import NProfile


class ASH(object):
    def __init__(self, hedge_removal: bool = False) -> None:
        """

        :param hedge_removal:
        """
        # edge data
        self.snapshots = {}  # {time: [edge_id, ...]}
        self.edge_mapping = {}  # {edge_id: nodes}
        self.node_set_to_hyperedge = {}  # {nodes: hyperedge_id}
        self.current_hyperedge_id = 0

        self.hedge_removal = hedge_removal
        if hedge_removal:  # warn user
            warnings.warn("Hyperedge removal is not implemented yet.")

        # node data
        self.node_profiles = defaultdict(lambda: defaultdict(NProfile))  # {node: {time: NProfile}}
        self.stars = defaultdict(set)  # {node: {edge_id, ...}}

    def add_hyperedge(self, nodes: list, start: int, end: int = None) -> None:
        """

        :param nodes:
        :param start:
        :param end:
        :param attrs:
        :return:
        """
        if end is None:
            span = (start, start)
        else:
            span = (start, end)

        nodes = tuple(sorted(nodes))

        if self.hedge_removal:
            pass
        else:
            if nodes in self.node_set_to_hyperedge:
                # hyperedge already present (also nodes)
                hyperedge_id = self.node_set_to_hyperedge[nodes]
            else:
                # new hyperedge
                self.current_hyperedge_id += 1
                hyperedge_id = f"e{self.current_hyperedge_id}"
                self.node_set_to_hyperedge[nodes] = hyperedge_id
                self.edge_mapping[hyperedge_id] = nodes

                # add nodes if not present
                for n in nodes:
                    # if not present at all
                    if n not in self.node_profiles:
                        self.add_node(n, span[0], span[1], attr_dict={})
                    else:
                        # if present but not in the time span
                        for t in range(span[0], span[1] + 1):
                            if t not in self.node_profiles[n]:
                                self.add_node(n, t, attr_dict={})
                    self.stars[n].add(hyperedge_id)

        for t in range(span[0], span[1] + 1):
            if t not in self.snapshots:
                self.snapshots[t] = set()
            self.snapshots[t].add(hyperedge_id)

    def add_hyperedges(self, hyperedges: list, start: int, end: int = None) -> None:
        """

        :param hyperedges:
        :param start:
        :param end:
        :return:
        """
        for hedge in hyperedges:
            self.add_hyperedge(hedge, start, end)

    def add_node(self, node: int, start: int, end: int = None, attr_dict: object = None) -> None:
        """

        :param node:
        :param start:
        :param end:
        :param attr_dict:
        :return:
        """
        if end is None:
            span = (start, start)
        else:
            span = (start, end)
        if attr_dict is None:
            attr_dict = {}

        for t in range(span[0], span[1] + 1):
            self.node_profiles[node][t] = NProfile(node, **attr_dict)

    def add_nodes(self, nodes: list, start: int, end: int = None, node_attr_dict: dict = None) -> None:
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

    def avg_number_of_nodes(self) -> float:
        """
        The avg_number_of_nodes function returns the average number of nodes in an ASH over time, i.e., the
        sum of each snapshot's nodes divided by the number of snapshots.

        :return: The average number of nodes in the ASH over all snapshots
        """

        nodes_snapshots = [self.get_number_of_nodes(tid) for tid in self.snapshots.keys()]
        return sum(nodes_snapshots) / len(self.snapshots.keys())

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

    def get_avg_number_of_hyperedges(self) -> float:
        """
        The get_avg_number_of_hyperedges function returns the average number of
        hyperedges in each snapshot.

        :return: The average number of hyperedges per snapshot
        """

        return sum([len(he) for he in self.snapshots.values()]) / len(self.snapshots)

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

    def get_hyperedge_attribute(self, hyperedge_id: str, attribute_name: str, tid: int = None) -> object:
        """
        The get_hyperedge_attribute function returns the value of a specific attribute of a hyperedge.

        :param hyperedge_id:str: Specify the hyperedge to get the attribute of
        :param attribute_name: Specify the attribute that is to be returned
        :param tid: Specify a time slot
        :return: The attribute of a hyperedge
        """

    def get_hyperedge_attributes(self, hyperedge_id: str, tid: int = None) -> dict:
        """
        The get_hyperedge_attributes function returns a dictionary of the attributes associated with a hyperedge.
        If tid is specified, it will only return the attribute values for that timeslot.

        :param hyperedge_id: Specify the hyperedge to get the attributes of
        :param tid: Specify a snapshot
        :return: The attributes of a hyperedge
        """
        pass

    def get_hyperedge_id_set(self, tid: int = None) -> set:
        """

        :return:
        """
        if tid is None:
            return set(self.edge_mapping.keys())
        return set(self.snapshots[tid])

    def get_hyperedge_nodes(self, hyperedge_id: str) -> list:
        """

        :param hyperedge_id:
        :return:
        """
        return self.edge_mapping[hyperedge_id]

    def get_hyperedge_weight(self, hyperedge_id):
        """
        The get_hyperedge_weight function returns the weight of a hyperedge,

        :param hyperedge_id: The hyperedge id
        :return: The weight of the hyperedge
        """
        pass

    def get_neighbors(self, node: int, hyperedge_size: int = None, tid: int = None) -> set:
        """

        :param node:
        :param hyperedge_size:
        :param tid:
        :return:
        """
        return set([n for he in self.get_star(node, hyperedge_size, tid) for n in self.edge_mapping[he] if n != node])

    def get_node_attribute(self, node: int, attr_name: str, tid: int = None) -> object:
        """

        :param node:
        :param attr_name:
        :param tid:
        :return:
        """
        pass

    def get_node_presence(self, node: int) -> list:
        """
        The get_node_presence function returns the time span in which a node is present in the ASH.

        :param node: The node id
        :return: A list of tuples representing the time span in which the node is present
        """
        return list(self.node_profiles[node].keys())

    def get_node_profile(self, node: int, tid: int = None) -> object:
        """

        :param node:
        :param tid:
        :return:
        """
        if tid is None:
            return self.node_profiles[node]
        return self.node_profiles[node][tid]

    def get_node_set(self, tid=None) -> set:
        """
        The get_node_set function returns the set of all the nodes in the ASH.
        If a snapshot id (tid) is specified, it will return only those nodes that appear in that snapshot.

        :param tid: Optional temporal snapshot id
        :return: The set of nodes in the ASH
        """
        if tid is None:
            return set(self.node_profiles.keys())
        else:
            return set([n for n in self.node_profiles if self.has_node(n, tid)])

    def get_number_of_neighbors(self, node: int, hyperedge_size: int = None, tid: int = None) -> int:
        """
        The get_number_of_neighbors function returns the number of neighbors of a node in the ASH.
        If no temporal snapshot id (tid) is specified, it will return the total number of neighbors.
        Otherwise, it will return the total number of neighbors that are active in tid.

        :param node: The node id
        :param hyperedge_size: Optional hyperedge size
        :param tid: Optional temporal snapshot id
        :return: The number of neighbors of the node in the ASH
        """
        return len(self.get_neighbors(node, hyperedge_size, tid))

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

    def get_s_incident(self, hyperedge_id: str, s: int, start: int = None, end: int = None) -> list:
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

    def get_size(self, tid: int = None) -> int:
        """
        The get_size function returns the number of hyperedges in the ASH.
        If a temporal id (tid) is specified, it will return the number of hyperedges in that snapshot.

        :param tid: Optional temporal snapshot id
        :return: The number of hyperedges in the ASH
        """
        return len(self.get_hyperedge_id_set(tid=tid))

    def get_star(self, node: int, hyperedge_size: int = None, tid: int = None) -> set:
        """

        :param node:
        :param hyperedge_size:
        :param tid:
        :return:
        """
        if tid is None and hyperedge_size is None:  # all hyperedges
            return self.stars[node]
        elif tid is None and hyperedge_size is not None:  # hyperedges of a specific size
            return set([he for he in self.stars[node] if len(self.edge_mapping[he]) == hyperedge_size])
        elif tid is not None and hyperedge_size is None:  # hyperedges in a specific time
            return set([he for he in self.stars[node] if he in self.snapshots[tid]])
        else:  # hyperedges of a specific size in a specific time
            return set([he for he in self.stars[node] if he in self.snapshots[tid] and len(self.edge_mapping[he]) == hyperedge_size])

    def has_hyperedge(self, nodes: list, tid: int = None) -> bool:
        """

        :param nodes:
        :param tid:
        :return:
        """
        if tid is None:
            return tuple(sorted(nodes)) in self.node_set_to_hyperedge
        return self.node_set_to_hyperedge[tuple(sorted(nodes))] in self.snapshots[tid]

    def has_hyperedge_id(self, hyperedge_id: str, tid: int = None) -> bool:
        """

        :param hyperedge_id:
        :param tid:
        :return:
        """
        if tid is None:
            return hyperedge_id in self.edge_mapping

        return hyperedge_id in self.snapshots[tid]

    def has_node(self, node: int, tid: int = None) -> bool:
        """

        :param node:
        :param tid:
        :return:
        """
        if tid is None:
            return node in self.node_profiles

        return tid in self.node_profiles[node]

    def hyperedge_contribution(self, hyperedge_id: str) -> float:
        """
        The hyperedge_contribution function calculates the fraction of snapshots where
        a given hyperedge is active.

        :param hyperedge_id: Specify which hyperedge to calculate the contribution for
        :return: The contribution of a hyperedge
        """

        contr = 0
        for tid in self.temporal_snapshots_ids():
            contr += 1 if self.has_hyperedge_id(hyperedge_id, tid) else 0
        return contr / len(self.snapshots)

    def hyperedge_id_iterator(self, start: int = None, end: int = None) -> list:
        """

        :param start:
        :param end:
        :return:
        """
        if start is None:
            yield from list(self.edge_mapping.keys())
        else:
            if end is None:
                end = start
            yielded = set()  # to avoid duplicates
            for t in range(start, end + 1):
                if t in self.snapshots:
                    for hedge_id in self.snapshots[t]:
                        if hedge_id not in yielded:
                            yielded.add(hedge_id)
                            yield hedge_id

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
            temp, _ = self.hypergraph_temporal_slice(start, end)
            for he in temp.get_hyperedge_id_set():
                dist[len(temp.get_hyperedge_nodes(he))] += 1

        return dist

    def hypergraph_temporal_slice(self, start: int, end: int = None) -> object:
        """
        The hypergraph_temporal_slice constructs a new ASH instance that contains hyperedges active in the given
        time window. It returns both the new instance and a dictionary mapping old hyperedge ids to new hyperedge ids.
        If no end time is specified, then all temporal ids greater or equal to start are considered.

        :param start: Specify the start time of the temporal slice
        :param end: Specify the end of the temporal slice
        :return: an ASH instance and a dictionary mapping old hyperedge ids to new hyperedge ids
        """

        res = ASH(hedge_removal=self.hedge_removal)
        eid_to_new_eid = {}  # mapping of old hyperedge ids to new hyperedge ids
        if end is None:
            end = start
        for t in range(start, end + 1):
            for hedge_id in self.snapshots[t]:
                nodes = self.get_hyperedge_nodes(hedge_id)
                res.add_hyperedge(nodes, t, t)
                eid_to_new_eid[hedge_id] = res.node_set_to_hyperedge[tuple(sorted(nodes))]

        # copy node profiles
        for node in self.get_node_set():
            for t in range(start, end + 1):
                if self.has_node(node, t):
                    profile = self.get_node_profile(node, t)
                    res.add_node(node, t, t, profile.get_attributes())

        return res, eid_to_new_eid

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

    def node_attributes_to_attribute_values(self, categorical=False, tid: int = None) -> dict:
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
        for n in self.get_node_set(tid=tid):
            for name, vals in self.get_node_profile(n, tid=tid).items():
                if name != "t":
                    if tid is None:
                        for value in vals.values():
                            attributes[name].add(value)
                    else:
                        attributes[name].add(vals)
        if categorical:
            numerical = [attribute for attribute in attributes if not isinstance(list(attributes[attribute])[0], str)]
            for attribute in numerical:
                del attributes[attribute]

        return attributes

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
        """ """
        dist = defaultdict(int)

        if start is None:

            for node in self.get_node_set():
                deg = self.get_degree(node)
                dist[deg] += 1
        else:
            temp, _ = self.hypergraph_temporal_slice(start, end)
            for node in temp.get_node_set():
                deg = temp.get_degree(node)
                dist[deg] += 1

        return dist

    def node_iterator(self, tid: int = None) -> list:
        """
        The node_iterator function returns an iterator over the nodes in the ASH.
        If no temporal id (tid) is specified, it will iterate over all nodes.
        Otherwise, it will iterate only over those nodes that exist at that temporal id.

        :param tid: Specify the temporal snapshot id
        :return: An iterator over the ASH's nodes
        """
        yield from self.get_node_set(tid)

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

    def stream_interactions(self) -> list:
        """
        Yields the interactions in the ASH as a stream of tuples (t, hedge_id, op).
        op is a string indicating the operation performed on the hyperedge.
        The '+' indicates the addition at time t, and '-' indicates the removal.
        If hedge_removal is False, then '-' is not used.

        :return: A stream of interactions
        """

        if self.hedge_removal:
            # absence of previously present hyperedges is a removal
            # presence of previously absent hyperedges is an addition

            # yield the first snapshot
            tids = self.temporal_snapshots_ids()
            for hedge_id in self.snapshots[tids[0]]:
                yield tids[0], hedge_id, "+"

            # yield additions and removals
            for t in tids[1:]:
                # additions
                for hedge_id in self.snapshots[t]:
                    if hedge_id not in self.snapshots[t - 1]:
                        yield t, hedge_id, "+"
                for hedge_id in self.snapshots[t - 1]:
                    if hedge_id not in self.snapshots[t]:
                        yield t, hedge_id, "-"
        else:
            yielded = set()
            for t in self.temporal_snapshots_ids():
                for hedge_id in self.snapshots[t]:
                    if hedge_id not in yielded:
                        yielded.add(hedge_id)
                        yield t, hedge_id, "+"

    def temporal_snapshots_ids(self) -> list:
        """
        Returns the list of temporal snapshots ids for the ASH, i.e.,
        integers representing points in time.

        :return: A list of temporal snapshot ids
        """
        return sorted(self.snapshots.keys())

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
