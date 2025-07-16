from ash_model.paths.walks import all_shortest_s_walk_lengths, s_components
from ash_model.utils import hyperedge_most_frequent_node_attribute_value
from ash_model import ASH

import networkx as nx
from itertools import combinations
from collections import defaultdict
import tqdm


def __label_frequency(
    g: nx.Graph, u: object, nodes: list, labels: list, hierarchies: dict = None
) -> float:
    """
    Compute the similarity of node profiles

    :param g: a networkx Graph object
    :param u: node id
    :param labels: list of node categorical labels
    :param hierarchies: dict of labels hierarchies
    :return: node profiles similarity score in [-1, 1]
    """
    s = 1
    for label in labels:
        if label not in g.nodes[u]:
            continue
        a_u = g.nodes[u][label]

        # set of nodes at given distance
        sgn = {}
        for v in nodes:
            # indicator function that exploits label hierarchical structure

            if not g.has_node(v) or label not in g.nodes[v]:
                continue

            sgn[v] = (
                1
                if a_u == g.nodes[v][label]
                else __distance(label, a_u, g.nodes[v][label], hierarchies)
            )
            v_neigh = list(g.neighbors(v))
            if len(v_neigh) == 0:
                continue
            # compute the frequency for the given node at distance n over neighbors label
            f_label = len(
                [x for x in v_neigh if g.nodes[x][label] == g.nodes[v][label]]
            ) / len(v_neigh)
            f_label = f_label if f_label > 0 else 1
            sgn[v] *= f_label
        s *= sum(sgn.values()) / len(nodes)

    return s


def __distance(label: str, v1: str, v2: str, hierarchies: dict = None) -> float:
    """
    Compute the distance of two labels in a plain hierarchy
    :param label: label name
    :param v1: first label value
    :param v2: second label value
    :param hierarchies: labels hierarchies
    """
    if hierarchies is None or label not in hierarchies:
        return -1

    return -abs(hierarchies[label][v1] - hierarchies[label][v2]) / (
        len(hierarchies[label]) - 1
    )


def __normalize(u: object, scores: list, max_dist: int, alphas: list) -> list:
    """
    Normalize the computed scores in [-1, 1]

    :param u: node
    :param scores: datastructure containing the computed scores for u
    :param alphas: list of damping factor
    :return: scores updated
    """
    for alpha in alphas:
        norm = sum([(d**-alpha) for d in range(1, max_dist + 1)])

        for profile in scores[str(alpha)]:
            if u in scores[str(alpha)][profile]:
                scores[str(alpha)][profile][u] /= norm

    return scores


def hyper_conformity(
    h: ASH,
    alphas: list,
    labels: list,
    s: int = 1,
    profile_size: int = 1,
    hierarchies: dict = None,
    tid: int = None,
) -> dict:
    """
    Compute the Attribute-Profile Conformity for the considered graph
    :param h:
    :param alphas: list of damping factors
    :param labels: list of node categorical labels
    :param s:
    :param profile_size:
    :param hierarchies: label hierarchies
    :param tid:
    :return: conformity value for each node in [-1, 1]

    -- Example --
    >> g = nx.karate_club_graph()
    >> pc = profile_conformity(g, 1, ['club'])
    """

    full_res = []

    for comp in s_components(h, s):
        b, he_map = h.induced_hypergraph(comp)

        g = b.s_line_graph(s=s, start=tid, end=tid)

        for cmp in nx.connected_components(g):
            g1 = nx.subgraph(g, cmp).copy()

            if profile_size > len(labels):
                raise ValueError("profile_size must be <= len(labels)")

            if len(alphas) < 1 or len(labels) < 1:
                raise ValueError(
                    "At list one value must be specified for both alphas and labels"
                )

            profiles = []
            for i in range(1, profile_size + 1):
                profiles.extend(combinations(labels, i))

            # Attribute value frequency
            labels_value_frequency = defaultdict(lambda: defaultdict(int))

            # hyperedge most frequent label
            for he in b.hyperedges():
                for label in labels:
                    v = list(
                        hyperedge_most_frequent_node_attribute_value(
                            b, he, label, tid
                        ).keys()
                    )
                    if len(v) == 0:
                        continue
                    v = v[0]
                    labels_value_frequency[label][v] += 1
                    # annotate the line graph
                    g1.add_node(he, **{label: v})

            # Normalization
            df = defaultdict(lambda: defaultdict(int))
            for k, v in labels_value_frequency.items():
                tot = 0
                for p, c in v.items():
                    tot += c

                for p, c in v.items():
                    df[k][p] = c / tot

            res = {
                str(a): {
                    "_".join(profile): {n: 0 for n in g1.nodes()}
                    for profile in profiles
                }
                for a in alphas
            }

            for u in tqdm.tqdm(g1.nodes()):
                sp = dict(all_shortest_s_walk_lengths(b, s, u))

                dist_to_nodes = defaultdict(list)
                for _, nodelist in sp.items():
                    for node, dist in nodelist.items():
                        dist_to_nodes[dist].append(node)
                sp = dist_to_nodes

                for dist, nodes in sp.items():
                    if dist != 0:
                        for profile in profiles:
                            sim = __label_frequency(
                                g1, u, nodes, list(profile), hierarchies
                            )

                            for alpha in alphas:
                                partial = sim / (dist**alpha)
                                p_name = "_".join(profile)
                                if u in res[str(alpha)][p_name]:
                                    res[str(alpha)][p_name][u] += partial

                res = __normalize(u, res, max(sp.keys()), alphas)

            # remap
            for ap, conf_dict in res.items():
                for feat, node_dict in conf_dict.items():
                    node_dict = {
                        he_map[n]: v for n, v in node_dict.items() if n in he_map
                    }
                    res[ap][feat] = node_dict

            full_res.append(res)

    return full_res
