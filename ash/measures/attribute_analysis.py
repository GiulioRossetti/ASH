from ash import ASH, NProfile
from collections import defaultdict, Counter
import numpy as np
from math import log, e
from collections.abc import Callable


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

        :param h:
        :param hyperedge_id:
        :param attribute:
        :param tid:
        :return:
        """
    nodes = h.get_hyperedge_nodes(hyperedge_id)
    app = defaultdict(list)
    for node in nodes:
        profile = h.get_node_profile(node, tid=tid)
        value = profile.get_attribute(attribute)

        if isinstance(value, str):
            app[attribute].append(value)

        if isinstance(value, dict):
            vals = [v for v in value.values()]
            app[attribute].extend(vals)

    count = Counter(app[attribute])
    count = count.most_common(1)[0]

    res = {count[0]: count[1]}

    return res


def hyperedge_aggregate_node_profile(
    h: ASH, hyperedge_id: str, tid: int, agg_function: Callable[[list], float] = np.mean
) -> NProfile:
    """

        :return:
        :param h:
        :param hyperedge_id:
        :param tid:
        :param agg_function:
        :return:
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

    :param h:
    :param hyperedge_id:
    :param tid:
    :return:
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
