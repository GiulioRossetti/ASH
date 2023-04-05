import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ash.measures import *


def __rolling_mean(it, window):
    """

    :param it:
    :param window:
    :return:
    """
    return pd.Series(it).rolling(window=window).mean()


def plot_structure_dynamics(
    h: ASH, func: Callable, func_params: dict = None, rolling: int = None, **kwargs
) -> object:
    """

    :param h: ASH instance
    :param func: function to be called at each timestamp
    :param func_params: parameters of the above function in key-value pairs
    :param rolling: size of the rolling window
    :param kwargs: Pass matplotlib keyword arguments to the function
    :return: The axes object
    """
    if func_params is None:
        func_params = {}

    if "h" in func.__code__.co_varnames:  # e.g., inclusiveness
        func_params.update({"h": h})

    if "tid" in func.__code__.co_varnames:
        y = [
            func(**func_params, tid=tid) for tid in h.temporal_snapshots_ids()
        ]  # e.g., inclusiveness

    else:  # has 'start' and 'end', e.g. average_s_local_clustering_coefficient
        y = [
            func(**func_params, start=tid, end=tid)
            for tid in h.temporal_snapshots_ids()
        ]

    if rolling:
        y = __rolling_mean(y, window=rolling)

    if "ax" not in kwargs:
        ax = plt.gca()
    else:
        ax = kwargs["ax"]

    ax.plot(y, **{k: v for k, v in kwargs.items() if k != "ax"})

    return ax


def plot_attribute_dynamics(
    h: ASH,
    attr_name: str,
    func: Callable,
    func_params: dict = None,
    rolling: int = None,
    **kwargs
) -> object:
    """

    :param h: ASH instance
    :param attr_name: attribute name
    :param func: function to be called at each timestamp
    :param func_params: parameters of the above function in key-value pairs
    :param rolling: size of the rolling window
    :param kwargs: Pass matplotlib keyword arguments to the function
    :return: The axes object
    """
    if func_params is None:
        func_params = {}

    if "h" in func.__code__.co_varnames:  # e.g., inclusiveness
        func_params.update({"h": h})

    res = defaultdict(list)

    for tid in h.temporal_snapshots_ids():
        res_t = func(**func_params, tid=tid)[attr_name]
        if isinstance(res_t, dict):  # if result is by label
            for label in res_t:
                res[label].append(res_t[label])
        else:  # if result is aggregated
            res[attr_name].append(res_t)

    if "ax" not in kwargs:
        ax = plt.gca()
    else:
        ax = kwargs["ax"]

    for name, y in res.items():
        if rolling:
            y = __rolling_mean(y, window=rolling)
        ax.plot(y, label=name, **{k: v for k, v in kwargs.items() if k != "ax"})
    ax.legend()
    return ax


# def plot_temporal_attribute_distribution(h, attr_name, **kwargs):
#
#    if 'ax' not in kwargs:
#        ax = plt.gca()
#    tids = h.temporal_snapshots_ids()
#
#    labels = h.node_attributes_to_attribute_values()[attr_name]
#
#    distr = {label: {t: 0 for t in tids} for label in labels}
#
#    for t in tids:
#        nodes = h.get_node_set(tid=t)
#        node_count = 0
#        for node in nodes:
#            prof = h.get_node_profile(node, t)
#            attr = prof.get_attribute(attr_name)
#            distr[attr][t] += 1
#            node_count += 1
#
#        for label in labels:
#            distr[label][t] = distr[label][t] / node_count
#    data = pd.DataFrame(distr, columns=sorted(distr.keys())).transpose()
#    sns.heatmap(data=data, ax=ax, **kwargs)
#
#    return ax


def plot_consistency(h, **kwargs):
    """
    Plot the kde of the distribution of the consistency values for all nodes in the ASH.
    Given a node and an attribute, the node's consistency w.r.t. the attribute is the complementary of the entropy
    computed on the set of the attribute's values the node holds in time.

    :param h: ASH instance
    :param kwargs: Pass matplotlib keyword arguments to the function
    :return: The axes object
    """

    if "ax" not in kwargs:
        ax = plt.gca()
    else:
        ax = kwargs["ax"]

    cons = consistency(h)
    for attr_name in list(cons.values())[0]:
        y = [v[attr_name] for v in cons.values()]

        sns.kdeplot(
            y, label=attr_name, ax=ax, **{k: v for k, v in kwargs.items() if k != "ax"}
        )
    ax.legend()
    ax.set_xlim((-0.2, 1.2))

    return ax
