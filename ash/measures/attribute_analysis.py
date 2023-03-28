from collections import defaultdict, Counter
from math import log, e
from typing import Callable

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


def hyperedge_most_frequent_node_attribute_value(
        h: ASH, hyperedge_id: str, attribute: str, tid: int
) -> dict:
    """
    The hyperedge_most_frequent_node_attribute_value function returns the most frequent value of a given attribute
    for the nodes in a hyperedge. If there are multiple values with the same frequency, then only one is returned.
    The function returns an empty dictionary if no node has that attribute.

    :param h: ASH instance
    :param hyperedge_id: Specify the hyperedge of interest
    :param attribute: Specify the attribute name
    :param tid: Temporal snapshot id
    :return: A dictionary containing the attribute value and its frequency in the hyperedge

    """

    nodes = h.get_hyperedge_nodes(hyperedge_id)
    app = defaultdict(list)
    for node in nodes:
        profile = h.get_node_profile(node, tid=tid)

        if not profile.has_attribute(attribute):
            continue

        value = profile.get_attribute(attribute)

        if isinstance(value, str):
            app[attribute].append(value)

        if isinstance(value, dict):
            vals = [v for v in value.values()]
            app[attribute].extend(vals)

    count = Counter(app[attribute])

    if len(count) > 0:
        count = count.most_common(1)[0]
        res = {count[0]: count[1]}
        return res
    return {}


def hyperedge_aggregate_node_profile(
        h: ASH, hyperedge_id: str, tid: int, agg_function: Callable[[list], float] = np.mean
) -> NProfile:
    """
    The hyperedge_aggregate_node_profile function takes a hyperedge ID and an aggregation function, and returns the
    aggregate profile of all nodes in that hyperedge. The resulting NProfile will have as statistic the standard
    deviation of each attribute in the original nodes' profiles

    :param h: ASH instance
    :param hyperedge_id: Identify the hyperedge that we want to aggregate
    :param tid: Temporal snapshot id
    :param agg_function: Specify the aggregation function
    :return: The aggregated node profile of the hyperedge nodes
    """
    nodes = h.get_hyperedge_nodes(hyperedge_id)
    avg_profile = NProfile(None)
    res = defaultdict(list)
    for node in nodes:
        profile = h.get_node_profile(node, tid=tid)

        for key, value in profile.items():
            if not isinstance(value, str):
                res[key].append(value)

    for key, value in res.items():
        avg_profile.add_attribute(key, agg_function(value))
        avg_profile.add_statistic(key, "std", np.std(value))

    return avg_profile


def hyperedge_profile_purity(h: ASH, hyperedge_id: str, tid: int) -> dict:
    """
    The hyperedge_profile_purity function returns a dictionary of dictionaries.
    The outer dictionary is keyed by the attribute names, and each inner dictionary
    is keyed by the values of that attribute. Each inner dictionary contains two keys:
    'count' and 'purity'. The count represents how many times that value was seen
    in the hyperedge, while purity represents what percentage of nodes with that
    value make up the hyperedge.

    :param h: ASH instance
    :param hyperedge_id: Specify the hyperedge for which we want to compute the purity
    :param tid: Temporal snapshot id
    :return: A dictionary with the attribute names as keys and a dictionary of the most
    frequent value for that attribute in the hyperedge as values
    """

    nodes = h.get_hyperedge_nodes(hyperedge_id)

    attributes = set()

    for node in nodes:
        profile = h.get_node_profile(node, tid)
        prof = profile.get_attributes()
        names = prof.keys()
        keys = []
        for name in names:
            if isinstance(profile.get_attribute(name), str):
                keys.append(name)

        if len(attributes) == 0:
            attributes = set(keys)
        else:
            attributes = attributes & set(keys)

    res = {}
    for attribute in attributes:
        res[attribute] = hyperedge_most_frequent_node_attribute_value(
            h, hyperedge_id, attribute, tid
        )

    for attr, data in res.items():
        for k, _ in data.items():
            data[k] /= len(nodes)
        res[attr] = data

    return res


def average_hyperedge_profile_purity(h: ASH, tid: int, by_label: bool = True, min_hyperedge_size: int = 2) -> object:
    """
    The average_hyperedge_profile_purity function calculates the average purity value for each node attribute. If
    :by_label: is True, the average is computed for each attribute value. For instance, if the profiles have the
    attribute 'party', which can take the values 'L' or 'R', two averages for 'party' will be computed: one over all
    the hyperedges whose most frequent label is 'L', and one over all the hyperedges whose most frequent label is 'R'.

    Otherwise, the average purity is computed over all hyperedges.

    :param h: ASH instance
    :param tid: Specify the temporal id
    :param by_label: Determine if the purity is computed for each label or not
    :param min_hyperedge_size: Set the minimum size of a hyperedge to be considered in the average :return: A
    dictionary containing for each attribute, for each attribute value,  the average purity. If :by_label: is False,
    returns a dictionary mapping each attribute to the average purity over all hyperedges
    """

    attributes = h.node_attributes_to_attribute_values(categorical=True)
    if by_label:

        purities = {
            attribute: {
                attribute_value: [] for attribute_value in attributes[attribute]
            } for attribute in attributes

        }

    else:
        purities = {attribute: [] for attribute in attributes}

    for hyperedge_id in h.get_hyperedge_id_set(tid=tid):
        if len(h.get_hyperedge_nodes(hyperedge_id)) >= min_hyperedge_size:
            purity = hyperedge_profile_purity(h, hyperedge_id=hyperedge_id, tid=tid)
            for attr_name, result in purity.items():
                label = list(result.keys())[0]
                pur = list(result.values())[0]
                if by_label:
                    purities[attr_name][label].append(pur)
                else:
                    purities[attr_name].append(pur)
    if by_label:
        return {attribute: {val: np.mean(res) for val, res in results.items()} for attribute, results in
                purities.items()}
    return {k: np.mean(v) for k, v in purities.items()}


def hyperedge_profile_entropy(h: ASH, hyperedge_id: str, tid: int) -> dict:
    """
    The hyperedge_profile_entropy function calculates the entropy of each node attribute in a hyperedge.

    :param h: ASH instance
    :param hyperedge_id: Specify the hyperedge for which the entropy is calculated
    :param tid: Temporal snapshot id
    :return: A dictionary with the entropy of each attribute of the hyperedge's nodes
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


def average_hyperedge_profile_entropy(h: ASH, tid: int) -> dict:
    """
    Computes the average hyperedge profile entropy over all hyperedges in the ASH. For all attributes.

    :param h: ASH instance
    :param tid: Temporal snapshot id
    :return: A dictionary mapping each node attribute to the average entropy value
    """
    entropies = defaultdict(list)
    for hyperedge_id in h.get_hyperedge_id_set(tid=tid):
        ent = hyperedge_profile_entropy(h, hyperedge_id, tid=tid)
        for attr_name, val in ent.items():
            entropies[attr_name].append(val)

    return {k: np.mean(v) for k, v in entropies.items()}


def star_profile_entropy(
        h: ASH, node_id: int, tid: int, method: str = "aggregate"
) -> dict:
    """
    The star_profile_entropy function computes the entropy of each attribute in the star of a given node.
    If 'aggregate', it is computed by first aggregating the hyperedges into a single profile.
    If 'collapse', all the star nodes are considered.

    :param h: ASH instance
    :param node_id: Specify the node whose star we want to consider
    :param tid: Temporal snapshot id
    :param method: Select between two different methods of computing the star entropy for each node
    :return: A dictionary with the entropy for each attribute in the star of node_id
    """

    star = h.get_star(node_id, tid=tid)

    if method == "aggregate":
        attributes = list(h.get_node_profile(node_id, tid).get_attributes().keys())

        profiles = []
        for hyperedge_id in star:
            # build aggregated profile
            p = NProfile(None)
            for attr in attributes:
                value_ = hyperedge_most_frequent_node_attribute_value(
                    h, hyperedge_id, attribute=attr, tid=tid
                )
                if value_:
                    value = list(value_.keys())[0]
                    p.add_attribute(attr, value)
            profiles.append(p)

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


def average_star_profile_entropy(h, tid, method: str = "aggregate") -> dict:
    """
    Computes the average star profile entropy over all nodes in the ASH. For all attributes.
    If method is 'aggregate', it is computed by first aggregating star hyperedges into a single profile.
    Else if method is 'collapse', all the star nodes are considered.


    :param h: ASH instance
    :param tid: Temporal snapshot id
    :param method: Specify how to compute the star entropies
    :return: A dictionary mapping each node attribute to the average entropy value
    """
    entropies = defaultdict(list)
    for node_id in h.get_node_set(tid=tid):

        ent = star_profile_entropy(h, node_id, method=method, tid=tid)
        for attr_name, val in ent.items():
            entropies[attr_name].append(val)

    return {k: np.mean(v) for k, v in entropies.items()}


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

    star = h.get_star(node_id, tid=tid)

    attr_names = list(h.get_node_profile(node_id, tid).get_attributes().keys())

    profiles = []
    if method == "aggregate":

        for hyperedge_id in star:
            # build aggregated profile
            p = NProfile(None)
            for attr in attr_names:
                value_ = hyperedge_most_frequent_node_attribute_value(
                    h, hyperedge_id, attribute=attr, tid=tid
                )
                if value_:
                    value = list(value_.keys())[0]
                    p.add_attribute(attr, value)
            profiles.append(p)

    elif method == "collapse":
        nodes = []
        for hyperedge_id in star:
            nodes += h.get_hyperedge_nodes(hyperedge_id)
        profiles.extend([h.get_node_profile(n, tid) for n in set(nodes)])
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


def average_star_profile_homogeneity(h: ASH, tid: int, method: str = 'aggregate', by_label: bool = True,
                                     min_star_size: int = 2) -> object:
    """
    The average_star_profile_homogeneity function calculates the average homogeneity value for each node attribute.
    If :by_label: is True, the average is computed for each attribute value. For instance, if the profiles have the
    attribute 'party', which can take the values 'L' or 'R', two averages for 'party' will be computed: one over all
    the stars whose central node has 'party' equal to 'L',  and one over all the stars whose central node has 'party'
    equal to 'R'.

    Otherwise, the average homogeneity is computed over all stars.

    :param h: ASH instance
    :param tid: Specify the temporal id
    :param method: Specify how homogeneities are computed
    :param by_label: Determine if the homogeneity is computed for each label or not
    :param min_star_size: Set the minimum size of a star (in terms of number of hyperedges) to be considered
    :return: A dictionary containing for each attribute, for each attribute value, the average homogeneity.
    If :by_label: is False, returns a dictionary mapping each attribute to the average homogeneity over all stars.
    """
    attributes = h.node_attributes_to_attribute_values(categorical=True)
    if by_label:
        homogeneities = {
            attribute: {
                attribute_value: [] for attribute_value in attributes[attribute]
            } for attribute in attributes

        }

    else:
        homogeneities = {attribute: [] for attribute in attributes}

    for node_id in h.get_node_set(tid=tid):
        if len(h.get_star(node_id, tid=tid)) >= min_star_size:
            homogeneity = star_profile_homogeneity(h, node_id, method=method, tid=tid)
            for attr_name, hom in homogeneity.items():

                if by_label:
                    label = h.get_node_attribute(node_id, attr_name=attr_name, tid=tid)
                    homogeneities[attr_name][label].append(hom)
                else:
                    homogeneities[attr_name].append(hom)
    if by_label:
        return {attribute: {val: np.mean(res) for val, res in results.items()} for attribute, results in
                homogeneities.items()}
    return {k: np.mean(v) for k, v in homogeneities.items()}


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
    attributes = {attr: val for attr, val in h.node_attributes_to_attribute_values(tid=tid).items()
                  if isinstance(list(val)[0], str)
                  }
    group_degrees = {attribute: {attribute_value: []
                                 for attribute_value in attributes[attribute]}
                     for attribute in attributes}

    for n in h.get_node_set(tid=tid):
        for attr_name in attributes:
            deg = h.get_degree(n, hyperedge_size=hyperedge_size, tid=tid)
            attr = h.get_node_attribute(n, attr_name=attr_name, tid=tid)
            group_degrees[attr_name][attr].append(deg)

    res = {attribute:
               {attribute_value: np.mean(group_degrees[attribute][attribute_value])
                for attribute_value in attributes[attribute]}
           for attribute in attributes}
    return res


def consistency(h: ASH) -> dict:
    """
    The consistency measures the extent to which a nodes' attribute value
    remains constant/change across time. The higher the value,
    the more consistent in time are the node's values for that attribute.

    :param h: ASH instance
    :return: A dict containing, for each node, for each attribute, the consistency value
    """

    res = defaultdict(dict)
    attributes = h.node_attributes_to_attribute_values()

    for n in h.get_node_set():
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
