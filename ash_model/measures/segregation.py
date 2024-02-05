from collections import defaultdict, Counter

from ash_model import ASH


def hyperedge_affinity_count(h: ASH, tid: int, attr: str, criterion: str) -> tuple[dict, dict, dict]:
    """
    Returns the internal and external hyperedge counts for each attribute value.
    Also returns the number of hyperedges for each attribute value that is not internal nor external.
    Results are stored in dictionaries.
    :param h: an ASH object.
    :param tid: the temporal snapshot id.
    :param attr: the attribute to analyze.
    :param criterion:  the criterion to use in ['linear', 'majority', 'strict'].
    :return: tuple -- (internal_count, external_count, other_count)

    :Example:
    >>> import ash_model as ash
    >>> from ash_model import measures
    >>> import random
    >>> a = ash.ASH()
    >>> a.add_hyperedge([1, 3, 4], 1)
    >>> a.add_hyperedge([3, 4], 1)
    >>> for i in range(1, 5):
    ...     a.add_node(i, start=0, end=0, attr_dict=ash.NProfile(node_id=i, party=random.choice(["L", "R"])))
    >>> internal, external, other = measures.hyperedge_affinity_count(a, 0, "party", "linear")
    """
    res_int = defaultdict(int)  # {attr_value: count}
    res_ext = defaultdict(int)
    res_other = defaultdict(int)
    criterion_map = {
        'linear': linear,
        'majority': majority,
        'strict': strict
    }
    func = criterion_map[criterion]
    groups = h.list_node_attributes(categorical=True)[attr]

    for he in h.get_hyperedge_id_set(tid=tid):
        node_attrs = [h.get_node_profile(node, tid=tid).get_attribute(attr) for node in h.get_hyperedge_nodes(he)]
        he_size = len(node_attrs)
        c = Counter(node_attrs)
        for group in groups:
            freq = c[group] if group in c else 0
            if freq == 0:
                res_other[group] += 1
            else:
                r = func(he_size, freq, len(c))
                res_int[group] += r
                res_ext[group] += 1 - r

    return res_int, res_ext, res_other


def linear(size, c, n_classes) -> int:
    return c / size if c > size / n_classes else 0


def majority(size, c, n_classes) -> int:
    return 1 if c > size / n_classes else 0


def strict(size, c, n_classes) -> int:
    # n added for consistency with other functions
    return 1 if c == size else 0


def ei_index(
        h: ASH,
        tid: int,
        criterion: str = 'linear'
):
    """
    Returns the EI index for each attribute in the hypergraph.

    :param h:  an ASH object.
    :param tid: the temporal snapshot id.
    :param criterion: the criterion to use in ['linear', 'majority', 'strict'].
    :return: dict -- {attr: ei_index}

    :Example:
    >>> import ash_model as ash
    >>> from ash_model import measures
    >>> import random
    >>> a = ash.ASH()
    >>> a.add_hyperedge([1, 3, 4], 1)
    >>> a.add_hyperedge([3, 4], 1)
    >>> for i in range(1, 5):
    ...     a.add_node(i, start=0, end=0, attr_dict=ash.NProfile(node_id=i, party=random.choice(["L", "R"])))
    >>> ei_index = measures.ei_index(a, 0, "linear") # {'party': value}
    """
    res = {}
    attrs = h.list_node_attributes(categorical=True)

    for attr, groups in attrs.items():
        internal_count, external_count, _ = hyperedge_affinity_count(h, tid, attr, criterion)
        internal = sum(internal_count.values())
        external = sum(external_count.values())
        res[attr] = (internal - external) / (internal + external)

    return res


def guptas_Q(
        h: ASH,
        tid: int,
        criterion: str = 'linear',
) -> dict:
    """
    Returns Gupta's Q segregation measure for each attribute in the hypergraph.

    :param h: an ASH object.
    :param tid:  the temporal snapshot id.
    :param criterion:  the criterion to use in ['linear', 'majority', 'strict'].
    :return: dict -- {attr: guptas_Q}

    :Example:
    >>> import ash_model as ash
    >>> from ash_model import measures
    >>> import random
    >>> a = ash.ASH()
    >>> a.add_hyperedge([1, 3, 4], 1)
    >>> a.add_hyperedge([3, 4], 1)
    >>> for i in range(1, 5):
    ...     a.add_node(i, start=0, end=0, attr_dict=ash.NProfile(node_id=i, party=random.choice(["L", "R"])))
    >>> guptas_Q = measures.guptas_Q(a, 0, "linear") # {'party': value}

    """
    res = {}
    attrs = h.list_node_attributes(categorical=True)

    for attr, groups in attrs.items():
        internal_count, external_count, other_count = hyperedge_affinity_count(h, tid, attr, criterion)

        num = sum(internal_count.values()) / sum(external_count.values()) - len(groups)
        den = len(groups) - 1
        res[attr] = num / den

    return res
