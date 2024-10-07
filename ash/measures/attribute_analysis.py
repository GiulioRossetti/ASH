from collections import defaultdict, Counter
from math import log, e

import numpy as np

from ash import ASH, NProfile


def __entropy(labels, base=None):
    """

    :param labels:
    :param base:
    :return:
    """

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.0

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent


def __purity(labels):
    """
    purity is the relative frequency of the most common class
    this returns a dictionary mapping the most common class to its purity

    :param labels:
    :return:
    """

    res = {}  # attr -> purity

    counter = Counter(labels).most_common(1)[0]
    if len(counter) == 0:
        return res
    return {counter[0]: counter[1] / len(labels)}


def aggregate_node_profile(
    h: ASH, node: int, categorical_aggr: str = "mode", numerical_aggr: str = "mean"
) -> NProfile:
    """
    Returns an aggregated profile of a node over all time points.
    The categorical_aggr parameter specifies the aggregation method for categorical attributes.
    The numerical_aggr parameter specifies the aggregation method for numerical attributes.

    :param h: The ASH object
    :param node: The node id
    :param categorical_aggr: The aggregation method for categorical attributes. Options: "mode", "first", "last"
    :param numerical_aggr: The aggregation method for numerical attributes. Options: "mean", "median", "first", "last"
    :return: The aggregated profile of the node
    """
    name_to_func = {
        "mode": lambda x: max(set(x), key=x.count),
        "first": lambda x: x[0],
        "last": lambda x: x[-1],
        "mean": lambda x: sum(x) / len(x),
        "median": lambda x: sorted(x)[len(x) // 2],
    }

    aggr_profile = NProfile(node)
    attr_dicts = [
        h.get_node_profile(node, tid=t).get_attributes()
        for t in sorted(h.node_presence(node))
    ]
    attribute_values = defaultdict(list)
    for attr_dict in attr_dicts:
        for name, value in attr_dict.items():
            attribute_values[name].append(value)

    for name, values in attribute_values.items():
        if isinstance(values[0], str):
            aggr_profile.add_attribute(name, name_to_func[categorical_aggr](values))
        else:
            aggr_profile.add_attribute(name, name_to_func[numerical_aggr](values))
    return aggr_profile


def hyperedge_aggregate_node_profile(
    h: ASH,
    hyperedge_id: str,
    tid: int,
    attr_name: str = None,
    categorical_aggr: str = "mode",
    numerical_aggr: str = "mean",
) -> NProfile:
    """
    Returns an aggregated profile of the nodes in a hyperedge

    """
    name_to_func = {
        "mode": lambda x: max(set(x), key=x.count),
        "first": lambda x: x[0],
        "last": lambda x: x[-1],
        "mean": lambda x: sum(x) / len(x),
        "median": lambda x: np.median(x),
    }

    aggr_profile = NProfile(None)
    nodes = h.get_hyperedge_nodes(hyperedge_id)
    attribute_values = defaultdict(list)
    for node in nodes:
        profile = h.get_node_profile(node, tid)
        if not attr_name:
            for name, value in profile.get_attributes().items():
                attribute_values[name].append(value)
        else:
            attribute_values[attr_name].append(profile.get_attribute(attr_name))

    for name, values in attribute_values.items():
        if isinstance(values[0], str):
            val = name_to_func[categorical_aggr](values)
            aggr_profile.add_attribute(name, val)

        else:
            val = name_to_func[numerical_aggr](values)
            aggr_profile.add_attribute(name, val)
            aggr_profile.add_statistic(name, "std", np.std(values))

    return aggr_profile


def hyperedge_profile_purity(h: ASH, hyperedge_id: str, tid: int) -> dict:

    nodes = h.get_hyperedge_nodes(hyperedge_id)

    attributes = set()
    attr_values = defaultdict(list)

    for node in nodes:
        profile = h.get_node_profile(node, tid)
        prof = profile.get_attributes()
        names = prof.keys()
        for name in names:
            if isinstance(profile.get_attribute(name), str):
                attributes.add(name)
                attr_values[name].append(profile.get_attribute(name))

    res = {}
    for attribute in attributes:
        res[attribute] = __purity(attr_values[attribute])  # {most_common_class: purity}

    return res


def hyperedge_profile_entropy(h: ASH, hyperedge_id: str, tid: int) -> dict:
    """

    :param h:
    :param hyperedge_id:
    :param tid:
    :return:
    """
    nodes = h.get_hyperedge_nodes(hyperedge_id)

    attributes = defaultdict(list)

    for node in nodes:
        profile = h.get_node_profile(node, tid)

        for name, value in profile.get_attributes().items():
            if isinstance(value, str):
                attributes[name].append(value)

    res = {}
    for attribute in attributes:
        res[attribute] = __entropy(
            attributes[attribute], len(set(attributes[attribute]))
        )

    return res


def star_profile_entropy(
    h: ASH, node_id: int, tid: int, method: str = "aggregate"
) -> dict:
    """
    Computes entropy for nodes in the star of node_id. If 'aggregate',
    it is computed by first aggregating the hyperedges into a single profile.
    If 'collapse', all the star nodes are considered.

    :param h:
    :param node_id:
    :param tid:
    :param method:
    :return:
    """
    star = h.star(node_id, tid=tid)

    if method == "aggregate":

        profiles = []
        for hyperedge_id in star:
            # build aggregated profile
            aggr = hyperedge_aggregate_node_profile(h, hyperedge_id, tid)
            profiles.append(aggr)

    elif method == "collapse":
        nodes = []
        for hyperedge_id in star:
            nodes += h.get_hyperedge_nodes(hyperedge_id)
        profiles = [h.get_node_profile(n, tid) for n in set(nodes)]

    else:
        raise ValueError("method must either be 'aggregate' or 'collapse'")

    attributes = defaultdict(list)
    for profile in profiles:
        for name, value in profile.get_attributes().items():
            if isinstance(value, str):
                attributes[name].append(value)

    res = {}
    for attribute in attributes:
        res[attribute] = __entropy(
            attributes[attribute], len(set(attributes[attribute]))
        )
    return res


def star_profile_homogeneity(
    h: ASH, node_id: int, tid: int, method: str = "aggregate"
) -> dict:
    """
    The star_profile_homogeneity function computes the homogeneity of a node with respect to its star. For each node
    attribute value regarding the node, the function computes the relative frequency of that value among the star's
    other nodes.
    If method is 'aggregate', it is computed by first aggregating the hyperedges into a single profile.
    Else if method is 'collapse', all the star nodes are considered.


    :param h: ASH instance
    :param node_id: Specify the node for which we want to calculate the star homogeneity
    :param tid: Specify the temporal id
    :param method: Specify the method to be used in calculating the star profile homogeneity
    :return: A dictionary with the homogeneity of each attribute
    """
    star = h.star(node_id, tid=tid)

    if method == "aggregate":
        profiles = []
        for hyperedge_id in star:
            # build aggregated profile
            aggr = hyperedge_aggregate_node_profile(h, hyperedge_id, tid)
            profiles.append(aggr)

    elif method == "collapse":
        nodes = []
        for hyperedge_id in star:
            nodes += h.get_hyperedge_nodes(hyperedge_id)
        profiles = [h.get_node_profile(n, tid) for n in set(nodes)]
    else:
        raise ValueError("method must either be 'aggregate' or 'collapse'")

    attributes = defaultdict(list)
    for profile in profiles:
        for attr_name, value in profile.get_attributes().items():
            if isinstance(value, str):
                attributes[attr_name].append(value)
    res = {}
    # count frequency of node_id's attribute and divide it by star size
    for attr_name in attributes:
        node_attr = h.get_node_profile(node_id, tid=tid).get_attribute(attr_name)
        res[attr_name] = attributes[attr_name].count(node_attr) / len(star)

    return res


def average_group_degree(h: ASH, tid: int, hyperedge_size: int = None) -> object:
    """
    The average_group_degree function calculates the average degree of each group (nodes having the same label in the
    ASH). The function takes as input an ASH object, and returns a dictionary with keys corresponding to attributes (
    e.g., 'gender') and values corresponding to dictionaries with keys corresponding to attribute values (e.g.,
    'male') and values corresponding to the average degree of nodes that have that attribute value.

    :param h: ASH instance
    :param hyperedge_size: Specify the size of the hyperedges
    :param tid: Specify the temporal id
    :return: A dictionary of dictionaries containing the average group degree

    """
    attributes = h.list_node_attributes(tid=tid, categorical=True)

    group_degrees = {
        attribute: {attribute_value: [] for attribute_value in attributes[attribute]}
        for attribute in attributes
    }

    for n in h.nodes(tid=tid):
        for attr_name in attributes:
            deg = h.degree(n, hyperedge_size=hyperedge_size, tid=tid)
            attr = h.get_node_attribute(n, attr_name=attr_name, tid=tid)
            group_degrees[attr_name][attr].append(deg)

    res = {
        attribute: {
            attribute_value: np.mean(group_degrees[attribute][attribute_value])
            for attribute_value in attributes[attribute]
        }
        for attribute in attributes
    }
    return res


def consistency(h: ASH, node: int = None) -> dict:
    """
    The consistency measures the extent to which a nodes' attribute value
    remains constant/change across time. The higher the value,
    the more consistent in time are the node's values for that attribute.

    :param h: ASH instance
    :param node: Specify the node for which we want to calculate the consistency
    :return: A dict containing, for each node, for each attribute, the consistency value
    """

    res = defaultdict(dict)
    attributes = h.list_node_attributes(categorical=True)
    if node:
        nodes = [node]
    else:
        nodes = h.nodes()
    for n in nodes:
        profile = h.get_node_profile(n)
        for attr_name in attributes:
            if profile.has_attribute(attr_name):
                value = profile.get_attribute(attr_name)
                labels = list(value.values())
                consist = 1 - __entropy(labels, base=len(attributes[attr_name]))
                res[n][attr_name] = consist
            else:
                res[n][attr_name] = None
    return res
