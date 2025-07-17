import random
from ash_model import ASH, NProfile


def random_hypergraph(
    num_nodes: int, size_distr: dict, node_attrs: dict = None, seed=None
) -> ASH:
    """
    Generates a random hypergraph with the specified number of nodes and size distribution.

    :param num_nodes: The number of nodes in the hypergraph.
    :param size_distr: A dictionary where keys are hyperedge sizes and values are their probabilities.
    :param seed: Optional seed for random number generation.
    :return: An ASH object representing the generated hypergraph.
    """
    if seed is not None:
        random.seed(seed)

    h = ASH()

    for i in range(num_nodes):
        profile = NProfile(
            node_id=i,
            **(
                {attr: random.choice(values) for attr, values in node_attrs.items()}
                if node_attrs
                else {}
            )
        )
        h.add_node(i, start=0, end=0, attr_dict=profile)

    for size, count in size_distr.items():
        for _ in range(count):
            # Randomly select nodes for the hyperedge
            nodes = random.sample(range(num_nodes), size)
            h.add_hyperedge(nodes, start=0, end=0)

    # Optionally, add node attributes
    if node_attrs:
        for i in range(num_nodes):
            profile = NProfile(
                id=i,
                **{attr: random.choice(values) for attr, values in node_attrs.items()}
            )

    return h


def random_ash(
    num_nodes: int,
    size_distr: dict,
    time_steps: int,
    node_attrs: dict = None,
    seed=None,
) -> ASH:
    """
    Generates a random ASH (Attributed Simple Hypergraph) with the specified number of nodes,
    size distribution, and time steps.

    :param num_nodes: The number of nodes in the ASH.
    :param size_distr: A dictionary where keys are hyperedge sizes and values are their probabilities.
    :param time_steps: The number of time steps for the ASH.
    :param node_attrs: Optional dictionary of attribute-to-values
    :param seed: Optional seed for random number generation.
    :return: An ASH object representing the generated hypergraph.
    """
    if seed is not None:
        random.seed(seed)

    G = ASH()

    # example node_attrs: {'color': ['red', 'blue', 'green'], 'age': [1, 2, 3]}

    # add nodes
    for i in range(num_nodes):
        profile = NProfile(
            node_id=i,
            start=0,
            end=time_steps - 1,
            **(
                {attr: random.choice(values) for attr, values in node_attrs.items()}
                if node_attrs
                else {}
            )
        )
        G.add_node(i, start=0, end=time_steps - 1, attr_dict=profile)

    for size, count in size_distr.items():
        for _ in range(count):
            # Randomly select nodes for the hyperedge
            nodes = random.sample(range(num_nodes), size)
            # Add the hyperedge to the ASH
            G.add_hyperedge(nodes, start=0, end=time_steps - 1)

    return G
