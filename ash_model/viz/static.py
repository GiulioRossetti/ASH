import matplotlib.pyplot as plt

from ash_model.measures import *


def plot_s_degrees(h: ASH, smax: int, loglog: bool = True, **kwargs: object) -> object:
    """
    Plot the s-degree distribution of a hypergraph.

    A line for each s will be plotted, from 1 to smax inclusive.
    Matplotlib plotting parameters (e.g., color) can be passed as kwargs.

    :param h: ASH instance
    :param smax: Maximum value of s
    :param loglog: If True, plot in log-log scale
    :param kwargs: Matplotlib keyword arguments (e.g., ax, color, linewidth)
    :return: The matplotlib axes object
    """

    if "ax" not in kwargs:
        ax = plt.gca()
    else:
        ax = kwargs["ax"]

    nodes = h.nodes()
    ss = np.arange(1, smax + 1, 1)
    degs = {s: {n: 0 for n in nodes} for s in ss}
    for n in nodes:
        for s in ss:
            degs[s][n] = h.s_degree(n, s)
    for s in ss:
        deg = dict(sorted(degs[s].items(), key=lambda item: item[1], reverse=True))

        y = np.bincount(list(deg.values()))

        ax.plot(
            range(0, len(y)),
            y,
            ".",
            label="s_" + str(s),
            alpha=0.8,
            **{k: v for k, v in kwargs.items() if k != "ax"}
        )
        if loglog:
            ax.loglog()

        ax.legend()
    return ax


def plot_hyperedge_size_distribution(
    h: ASH, max_size: int = None, min_size: int = None, **kwargs: object
) -> object:
    """
    Plot the distribution of hyperedge sizes in a hypergraph.

    min_size and max_size can be used to filter out hyperedges.
    Matplotlib plotting parameters (e.g., color) can be passed as kwargs.

    :param h: ASH instance
    :param max_size: Maximum size of hyperedges to be plotted (optional)
    :param min_size: Minimum size of hyperedges to be plotted (optional)
    :param kwargs: Matplotlib keyword arguments (e.g., ax, color, alpha)
    :return: The matplotlib axes object
    """

    if "ax" not in kwargs:
        ax = plt.gca()
    else:
        ax = kwargs["ax"]

    size_dist = dict(h.hyperedge_size_distribution())
    if max_size:
        size_dist = {k: v for k, v in size_dist.items() if k <= max_size}
    if min_size:
        size_dist = {k: v for k, v in size_dist.items() if k >= min_size}
    x, y = zip(*size_dist.items())

    ax.bar(np.array(x), y, alpha=0.4, **{k: v for k, v in kwargs.items() if k != "ax"})
    return ax


def plot_degree_distribution(h: ASH, loglog: bool = True, **kwargs: object) -> object:
    """
    Plot the degree distribution of an ASH hypergraph.

    The default is to draw a log-log plot.
    Matplotlib plotting parameters (e.g., color) can be passed as kwargs.

    :param h: ASH instance
    :param loglog: If True, plot in log-log scale
    :param kwargs: Matplotlib keyword arguments (e.g., ax, color, alpha)
    :return: The matplotlib axes object
    """

    if "ax" not in kwargs:
        ax = plt.gca()
    else:
        ax = kwargs["ax"]
    nodes = h.nodes()
    degs = {n: 0 for n in nodes}
    for n in nodes:
        degs[n] = h.degree(n)

    deg = dict(sorted(degs.items(), key=lambda item: item[1], reverse=True))
    y = list(deg.values())
    y = np.bincount(y)
    x = range(0, len(y))
    ax.plot(x, y, ".", alpha=0.4, **{k: v for k, v in kwargs.items() if k != "ax"})
    if loglog:
        ax.loglog()
    return ax


def plot_s_ranks(h: ASH, smax: int, loglog: bool = True, **kwargs: object) -> object:
    """
    Plot the s-degree rank distribution of a hypergraph.

    A line for each s will be plotted, from 1 to smax inclusive.
    Matplotlib plotting parameters (e.g., color) can be passed as kwargs.

    :param h: ASH instance
    :param smax: Maximum value of s
    :param loglog: If True, plot in log-log scale
    :param kwargs: Matplotlib keyword arguments (e.g., ax, color, label)
    :return: The matplotlib axes object
    """
    if "ax" not in kwargs:
        ax = plt.gca()
    else:
        ax = kwargs["ax"]

    nodes = h.nodes()
    ss = np.arange(1, smax + 1, 1)
    degs = {s: {n: 0 for n in nodes} for s in ss}
    for n in nodes:
        for s in ss:
            degs[s][n] = h.s_degree(n, s)

    for s in ss:
        deg = dict(sorted(degs[s].items(), key=lambda item: item[1], reverse=True))

        y = list(deg.values())

        ax.plot(
            range(len(y)),
            y,
            ".",
            label="s_" + str(s),
            alpha=0.4,
            **{k: v for k, v in kwargs.items() if k != "ax"}
        )

        if loglog:
            ax.loglog()
    return ax
