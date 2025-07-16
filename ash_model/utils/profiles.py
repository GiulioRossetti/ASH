from collections import defaultdict
import numpy as np
from ash_model import ASH, NProfile


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


def hyperedge_most_frequent_node_attribute_value(
    h: ASH, hyperedge_id: str, attr_name: str, tid: int = None
) -> dict:
    """
    Returns the most frequent value of a node attribute in a hyperedge.
    If tid is specified, it returns the value at that time point.
    If tid is None, it returns the most frequent value across all time points.

    :param h: The ASH object
    :param hyperedge_id: The hyperedge id
    :param attr_name: The attribute name to consider
    :param tid: The temporal id. If None, considers all time points
    :return: A dictionary with the most frequent value of the attribute in the hyperedge nodes
    """
    nodes = h.get_hyperedge_nodes(hyperedge_id)
    values = []
    for node in nodes:
        profile = h.get_node_profile(node, tid)
        if profile.has_attribute(attr_name):
            # If tid is specified, get the value at that time point
            # If tid is None, it's the aggregated value across all time points

            values.append(
                profile.get_attribute(attr_name)
                if tid is not None
                else profile.get_attribute(attr_name)
            )

    if not values:
        return {}

    return {
        max(set(values), key=values.count): values.count(
            max(set(values), key=values.count)
        )
    }


def hyperedge_aggregate_node_profile(
    h: ASH,
    hyperedge_id: str,
    tid: int,
    attr_name: str = None,
    categorical_aggr: str = "mode",
    numerical_aggr: str = "mean",
) -> NProfile:
    """
    Returns an aggregated profile of the nodes in a hyperedge.
    The categorical_aggr parameter specifies the aggregation method for categorical attributes.
    The numerical_aggr parameter specifies the aggregation method for numerical attributes.

    :param h: The ASH object
    :param hyperedge_id: The hyperedge id
    :param tid: The temporal id
    :param attr_name: The attribute name to aggregate. If None, all attributes are aggregated
    :param categorical_aggr: The aggregation method for categorical attributes. Options: "mode", "first", "last"
    :param numerical_aggr: The aggregation method for numerical attributes. Options: "mean", "median", "first", "last"
    :return: The aggregated profile of the hyperedge nodes
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
