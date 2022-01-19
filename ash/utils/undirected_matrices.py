from ash import ASH
from halp.undirected_hypergraph import UndirectedHypergraph
import halp.utilities.undirected_matrices as um
from scipy import sparse
import numpy as np


def get_node_mapping(h: ASH) -> (dict, dict):
    """

    :param h:
    :param tid:
    :return:
    """
    return um.get_node_mapping(h.H)


def get_hyperedge_id_mapping(h: ASH) -> (dict, dict):
    """

    :param h:
    :return:
    """
    return um.get_hyperedge_id_mapping(h.H)


def get_incidence_matrix(
    h: ASH, nodes_to_indices: dict, hyperedge_ids_to_indices: dict, tid: int = None
) -> dict:
    """

    :param h:
    :param tid:
    :param nodes_to_indices:
    :param hyperedge_ids_to_indices:
    :return:
    """

    if tid is not None:
        tids = [tid]
    else:
        tids = h.snapshots

    res = {}
    for tid in tids:
        s = h.hypergraph_temporal_slice(tid).H

        rows, cols = [], []
        for hyperedge_id, hyperedge_index in hyperedge_ids_to_indices.items():
            try:
                for node in s.get_hyperedge_nodes(hyperedge_id):
                    # get the mapping between the node and its ID
                    rows.append(nodes_to_indices.get(node))
                    cols.append(hyperedge_index)
            except:
                pass

        values = np.ones(len(rows), dtype=int)
        node_count = len(nodes_to_indices)
        hyperedge_count = len(hyperedge_ids_to_indices)

        res[tid] = sparse.csc_matrix(
            (values, (rows, cols)), shape=(node_count, hyperedge_count)
        )

    return res


def get_hyperedge_weight_matrix(
    h: ASH, hyperedge_ids_to_indices: dict, tid: int = None
) -> dict:
    """

    :param h:
    :param nodes_to_indices:
    :param hyperedge_ids_to_indices:
    :param tid:
    :return:
    """
    if tid is not None:
        tids = [tid]
    else:
        tids = h.snapshots

    res = {}
    for tid in tids:
        hyperedge_weights = {}
        for hyperedge_id in h.hyperedge_id_iterator():
            hyperedge_weights.update(
                {
                    hyperedge_ids_to_indices[hyperedge_id]: h.get_hyperedge_weight(
                        hyperedge_id
                    )
                }
            )

        hyperedge_weight_vector = []
        for i in range(len(hyperedge_weights.keys())):
            hyperedge_weight_vector.append(hyperedge_weights.get(i))

        res[tid] = sparse.diags([hyperedge_weight_vector], [0])

    return res


def get_vertex_degree_matrix(h: ASH, tid: int = None) -> dict:
    """

    :param h:
    :param tid:
    :return:
    """

    if tid is not None:
        tids = [tid]
    else:
        tids = h.snapshots

    res = {}
    for tid in tids:
        _, node_to_index = get_node_mapping(h)
        _, edge_to_index = get_hyperedge_id_mapping(h)
        W = get_hyperedge_weight_matrix(
            h, hyperedge_ids_to_indices=edge_to_index, tid=tid
        )[tid]
        M = get_incidence_matrix(
            h,
            nodes_to_indices=node_to_index,
            hyperedge_ids_to_indices=edge_to_index,
            tid=tid,
        )[tid]
        res[tid] = um.get_vertex_degree_matrix(M, W)
    return res
