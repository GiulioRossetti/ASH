import random

from ash_model.classes import ASH


def random_ASH(
    n_nodes: int,
    n_hyperedges: int,
    max_edge_size: int = 10,
    n_tids: int = 3,
    attr_to_vals_dict: dict = None,
    seed: int = 42,
) -> ASH:
    """

    :param n_nodes: Number of total nodes in the ASH
    :param n_hyperedges: Number of hyperedges to be added at each tid
    :param max_edge_size: Maximum hyperedge size
    :param n_tids: Number of temporal snapshots
    :param attr_to_vals_dict: Attribute-to-attribute-values dict, to be added to each node profile
    :param seed: Random seed
    :return: random ASH instance
    """
    random.seed(seed)

    if attr_to_vals_dict is None:
        attr_to_vals_dict = {}

    h = ASH(hedge_removal=True)
    # Generate random hyperedge sizes
    hes_presence = dict()
    nodes_presence = dict()
    for tid in range(n_tids):
        # Generate random hyperedges
        hyperedges = []
        nodes = set()
        hyperedge_sizes = [
            random.randint(2, max_edge_size) for _ in range(n_hyperedges)
        ]
        for size in hyperedge_sizes:
            he_nodes = random.sample(range(n_nodes), size)
            for node in he_nodes:
                nodes.add(node)
            hyperedges.append(he_nodes)
        nodes_presence[tid] = nodes
        hes_presence[tid] = hyperedges

    for tid, hes in hes_presence.items():
        h.add_hyperedges(hes, start=tid)
        nad = {
            n: {
                attr_name: random.choice(attr_to_vals_dict[attr_name])
                for attr_name in attr_to_vals_dict
            }
            for n in nodes_presence[tid]
        }

        h.add_nodes(nodes_presence[tid], start=tid, node_attr_dict=nad)

    return h
