from collections import defaultdict, Counter
from math import log, e

import numpy as np

from ash_model import ASH, NProfile
from ash_model.utils import hyperedge_aggregate_node_profile


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
    purity is the relative frequency of the most common attribute value
    this returns a dictionary mapping the most common value to its purity

    :param labels:

    :return:

    """

    res = {}  # attr -> purity

    counter = Counter(labels).most_common(1)[0]
    if len(counter) == 0:
        return res
    return {counter[0]: counter[1] / len(labels)}


def hyperedge_profile_purity(h: ASH, hyperedge_id: str, tid: int) -> dict:
    """
    Computes the purity of the hyperedge profile, i.e., the relative frequency of the most common attribute value
    for each attribute in the hyperedge nodes' profiles.

    :param h: The ASH object
    :param hyperedge_id: The hyperedge id
    :param tid: The temporal id

    :return: A dictionary with attribute names as keys and their purity as values

    """

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
    Computes the entropy of the hyperedge profile, i.e., the entropy of the attribute values
    for each attribute in the hyperedge nodes' profiles.

    :param h: The ASH object
    :param hyperedge_id: The hyperedge id
    :param tid: The temporal id

    :return: A dictionary with attribute names as keys and their entropy as values

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
    Returns the entropy of the star profile of a node, i.e., the entropy of the attribute values
    for each attribute in the profiles of the nodes in the star of the given node.
    If method is 'aggregate', it is computed by first aggregating each hyperedge into a single profile
    (see hyperedge_aggregate_node_profile).
    Else if method is 'collapse', all the node's neighbors are considered.

    :param h: ASH instance
    :param node_id: Specify the node for which we want to calculate the star profile entropy
    :param tid: Specify the temporal id
    :param method: Specify the method to be used in calculating the star profile entropy. Options are 'aggregate' or 'collapse'.

    :return: A dictionary with the entropy of each attribute

    """
    star = h.star(node_id, start=tid)

    if method == "aggregate":

        profiles = []
        for hyperedge_id in star:
            # build aggregated profile
            aggr = hyperedge_aggregate_node_profile(h, hyperedge_id, tid)
            profiles.append(aggr)

    elif method == "collapse":
        nodes = h.neighbors(node_id, start=tid)
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
    Returns the homogeneity of the star profile of a node, i.e., the relative frequency of the node's attribute value
    for each attribute in the profiles of the nodes in the star of the given node.
    If method is 'aggregate', it is computed by first aggregating each hyperedge into a single profile
    (see hyperedge_aggregate_node_profile).
    Else if method is 'collapse', all the node's neighbors are considered.

    :param h: ASH instance
    :param node_id: node id
    :param tid: temporal id
    :param method: Specify the method to be used in calculating the star profile homogeneity. Options are 'aggregate' or 'collapse'.

    :return: A dictionary with the homogeneity of each attribute

    """
    star = h.star(node_id, start=tid)

    if method == "aggregate":
        profiles = []
        for hyperedge_id in star:
            # build aggregated profile
            aggr = hyperedge_aggregate_node_profile(h, hyperedge_id, tid)
            profiles.append(aggr)

    elif method == "collapse":
        profiles = [
            h.get_node_profile(n, tid) for n in set(h.neighbors(node_id, start=tid))
        ]
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
    Computes the average degree of each group (nodes having the same label in the attribute)

    :param h: ASH instance
    :param tid: the temporal id
    :param hyperedge_size: Specify the size of the hyperedges

    :return: A dictionary with attribute names as keys and a dictionary of average degrees for each attribute value

    """
    attributes = h.list_node_attributes(tid=tid, categorical=True)

    group_degrees = {
        attribute: {attribute_value: [] for attribute_value in attributes[attribute]}
        for attribute in attributes
    }

    for n in h.nodes(start=tid):
        for attr_name in attributes:
            deg = h.degree(n, hyperedge_size=hyperedge_size, start=tid)
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


def attribute_consistency(h: ASH, node: int = None) -> dict:
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
        all_profiles = h.get_node_profiles_by_time(n)
        for attr_name in attributes:
            labels = []
            for time, profile in all_profiles.items():
                if profile.has_attribute(attr_name):
                    label = profile.get_attribute(attr_name)
                    labels.append(label)

            consist = 1 - __entropy(labels, base=len(attributes[attr_name]))
            res[n][attr_name] = consist
    if node:
        return res[node]
    return res
