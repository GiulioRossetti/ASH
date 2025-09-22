from typing import Dict, Tuple, Union
from scipy import sparse

from ash_model import ASH


def get_hyperedge_id_mapping(h: ASH) -> Dict[int, int]:
    """
    Returns a mapping of hyperedge IDs to indices and vice versa.
    This function creates a mapping from hyperedge IDs to their respective indices

    :param h: The ASH object to be converted.

    :return: A dictionary mapping hyperedge IDs to indices.
    """
    hyperedges = h.hyperedges(as_ids=True)
    edge_to_index = {he: i for i, he in enumerate(sorted(hyperedges))}
    return edge_to_index


def get_node_id_mapping(h: ASH) -> Dict[int, int]:
    """
    Returns a mapping of node IDs to indices and vice versa.

    :param h: The ASH object to be converted.

    :return: A dictionary mapping node IDs to indices.
    """
    nodes = h.nodes()
    node_to_index = {node: i for i, node in enumerate(sorted(nodes))}
    return node_to_index


def _build_incidence_matrix(
    h: ASH,
    node_to_index: Dict[int, int],
    he_to_index: Dict[int, int],
    start: int = None,
    end: int = None,
) -> sparse.csr_matrix:
    """
    Builds the incidence matrix for the ASH object using the provided mappings.

    :param h: The ASH object to be converted.
    :param node_to_index: A mapping of node IDs to indices.
    :param he_to_index: A mapping of hyperedge IDs to indices.

    :return: A sparse matrix representing the incidence matrix of the ASH.
    """
    incidence = sparse.lil_matrix((len(node_to_index), len(he_to_index)), dtype=int)

    for he in h.hyperedges(start, end):
        he_index = he_to_index[he]
        for node in h.get_hyperedge_nodes(he):
            node_index = node_to_index[node]
            incidence[node_index, he_index] = 1

    return incidence.tocsr()


def incidence_matrix(
    h: ASH, start: int = None, end: int = None, return_mapping: bool = False
) -> Union[sparse.csr_matrix, Tuple[sparse.csr_matrix, Dict[int, int], Dict[str, int]]]:
    """
    Returns the incidence matrix of the ASH object.
    The incidence matrix is a sparse matrix where rows represent nodes and columns represent hyperedges.
    The value is 1 if the node belongs to the hyperedge, 0 otherwise.

    :param h: The ASH object to be converted.
    :param start: The start time of the projection (optional).
    :param end: The end time of the projection (optional).

    :return: A sparse matrix representing the binary incidence matrix of the ASH.
    """

    # build index maps
    node_to_index = get_node_id_mapping(h)
    he_to_index = get_hyperedge_id_mapping(h)

    incidence = _build_incidence_matrix(h, node_to_index, he_to_index, start, end)

    if return_mapping:
        return incidence, node_to_index, he_to_index
    else:
        return incidence


def incidence_matrix_by_time(h: ASH, return_mapping: bool = False) -> Union[
    Dict[int, sparse.csr_matrix],
    Tuple[Dict[int, sparse.csr_matrix], Dict[int, int], Dict[str, int]],
]:
    """
    Returns a dictionary of incidence matrices for each time step in the ASH object.
    The keys are the time steps and the values are the corresponding incidence matrices.

    :param h: The ASH object to be converted.

    :return: A dictionary where keys are time steps and values are sparse matrices representing the incidence matrix.
    """

    incidence_matrices = {}
    #
    node_to_index = get_node_id_mapping(h)
    he_to_index = get_hyperedge_id_mapping(h)

    for t in h.temporal_snapshots_ids():
        incidence = _build_incidence_matrix(
            h, node_to_index, he_to_index, start=t, end=t
        )
        incidence_matrices[t] = incidence.tocsr()
    if return_mapping:
        return incidence_matrices, node_to_index, he_to_index
    else:
        return incidence_matrices


def adjacency_matrix(
    h: ASH, start: int = None, end: int = None, return_mapping: bool = False
) -> Union[sparse.csr_matrix, Tuple[sparse.csr_matrix, Dict[int, int]]]:
    """
    Returns the adjacency matrix of the ASH object.
    The adjacency matrix is a sparse matrix where rows and columns represent nodes,
    and the value is 1 if there is a hyperedge connecting the two nodes, 0 otherwise.

    :param h: The ASH object to be converted.
    :param start: The start time of the projection (optional).
    :param end: The end time of the projection (optional).

    :return: A sparse matrix representing the adjacency matrix of the ASH.
    """
    incidence = incidence_matrix(h, start, end, return_mapping=False)
    adjacency = incidence @ incidence.transpose()
    adjacency.setdiag(0)
    if return_mapping:
        return adjacency, get_node_id_mapping(h), get_hyperedge_id_mapping(h)
    else:
        return adjacency


def adjacency_matrix_by_time(
    h: ASH, return_mapping: bool = False
) -> Union[
    Dict[int, sparse.csr_matrix], Tuple[Dict[int, sparse.csr_matrix], Dict[int, int]]
]:
    """
    Returns a dictionary of adjacency matrices for each time step in the ASH object.
    The keys are the time steps and the values are the corresponding adjacency matrices.

    :param h: The ASH object to be converted.

    :return: A dictionary where keys are time steps and values are sparse matrices representing the adjacency matrix.
    """

    adjacency_matrices = {}
    for t in h.temporal_snapshots_ids():
        adjacency_matrices[t] = adjacency_matrix(h, t, t, return_mapping=False)

    if return_mapping:
        return adjacency_matrices, get_node_id_mapping(h)
    else:
        return adjacency_matrices
