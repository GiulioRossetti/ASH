import random
import numpy as np
from ash_model import ASH, NProfile


def _compute_probs_hba(h: ASH, node: int, homophily_rate: float) -> list:
    target_prob_dict = {}
    new_node_attr = h.get_node_attribute(node, attr_name="color", tid=0)
    for n in range(node):
        p = (
            homophily_rate
            if new_node_attr == h.get_node_attribute(n, attr_name="color", tid=0)
            else 1 - homophily_rate
        )
        p *= h.degree(n, start=0) + 0.00001  # avoid division by 0
        target_prob_dict[n] = p
    prob_sum = sum(target_prob_dict.values())
    return [v / prob_sum for _, v in target_prob_dict.items()]


def _random_hyperedge(n, probs, he_size):
    if he_size < 2:
        return
    try:
        new_hyperedge = list(
            np.random.choice(range(n), size=he_size - 1, p=probs, replace=False)
        )
        new_hyperedge.append(n)
        return set(new_hyperedge)
    except ValueError:
        _random_hyperedge(n, probs, he_size - 1)


def _truncated_pareto(
    alpha: float,
    size: int,
    xmin: float = 2.0,
    xmax: float = 10.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generates a truncated Pareto distribution.
    :param alpha: Shape parameter of the Pareto distribution.
    :param size: Number of samples to generate.
    :param xmin: Minimum value of the distribution.
    :param xmax: Maximum value of the distribution.
    :param rng: Optional random number generator.
    :return: An array of samples from the truncated Pareto distribution.
    """
    if xmin <= 0:
        raise ValueError("xmin must be positive.")
    if xmax < xmin:
        raise ValueError("xmax must be >= xmin.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    rng = rng or np.random.default_rng()

    # CDF of Pareto at xmax:  F_max = 1 - (xmin / xmax)**alpha
    F_max = 1.0 - (xmin / xmax) ** alpha

    # Draw uniform variables restricted to [0, F_max]
    u = rng.random(size)

    # Inverse CDF for the truncated case
    samples = xmin / (1.0 - u * F_max) ** (1.0 / alpha)
    return samples


def _sample_to_prob_distr(sample):
    """
    Converts a sample to a probability distribution.
    :param sample: A list of samples.
    :return: A list of probabilities normalized to sum to 1.
    """
    total = sum(sample)
    if total == 0:
        return [1.0 / len(sample)] * len(
            sample
        )  # Uniform distribution if total is zero
    return [x / total for x in sample]


def ba_with_homophily(
    num_nodes, m, homophily_rate, minority_size, n0, size_prob_distr=None
):
    """
    Generates a Barabasi-Albert hypergraph with homophily-driven connections.
    :param num_nodes: Total number of nodes in the hypergraph.
    :param m: Number of edges to attach from a new node to existing nodes.
    :param homophily_rate: Probability of connecting to nodes with the same attribute.
    :param minority_size: Proportion of nodes with the minority attribute.
    :param n0: Number of initial nodes to start the hypergraph.
    :param size_prob_distr: Optional probability distribution for hyperedge sizes.
                            If None, a truncated Pareto distribution is used.

    :return: An ASH object representing the generated hypergraph.
    :raises ValueError: If homophily_rate or minority_size is not in the range [0, 1].
    """
    if not (0 <= homophily_rate <= 1):
        raise ValueError("Homophily rate must be between 0 and 1.")
    if not (0 <= minority_size <= 1):
        raise ValueError("Minority size must be between 0 and 1.")

    h = ASH()
    # if size_prob_distr is None sample from pareto distribution
    if size_prob_distr is None:
        size_prob_distr = _truncated_pareto(
            alpha=1.5,
            size=num_nodes * m * 2,
            xmin=2.0,
            xmax=10.0,
            rng=np.random.default_rng(),
        )
        size_prob_distr = _sample_to_prob_distr(size_prob_distr)

    for n in range(num_nodes):
        if random.random() > minority_size:
            attr = "red"
        else:
            attr = "blue"
        p = NProfile(node_id=n, color=attr)
        h.add_node(n, start=0, end=0, attr_dict=p)
        if n < n0:
            continue

        # add m new hyperedges of random sizes
        new_hyperedges = set()
        for _ in range(m):
            he_size = np.random.choice(
                range(2, len(size_prob_distr) + 2), p=size_prob_distr
            )

            probs = _compute_probs_hba(
                h, n, homophily_rate
            )  # based on node degree and homophily
            new_hyperedge = _random_hyperedge(n, probs, he_size)
            if new_hyperedge:  # None in case n does not find another node to connect to
                new_hyperedges.add(frozenset(new_hyperedge))

        h.add_hyperedges(new_hyperedges, start=0, end=0)
    return h
