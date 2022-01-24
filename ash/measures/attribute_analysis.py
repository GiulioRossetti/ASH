from ash import ASH, NProfile


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
        res[attribute] = h.get_hyperedge_most_frequent_node_attribute_value(hyperedge_id, attribute, tid)

    for attr, data in res.items():
        for k, _ in data.items():
            data[k] /= len(nodes)
        res[attr] = data

    return res
