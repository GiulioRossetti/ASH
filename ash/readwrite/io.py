from ash import ASH, NProfile
import json
import gzip


__all__ = [
    "write_profiles_to_csv",
    "read_profiles_from_csv",
    "write_profiles_to_jsonl",
    "read_profiles_from_jsonl",
    "write_sh_to_csv",
    "read_sh_from_csv",
    "write_ash_to_json",
    "read_ash_from_json",
]


def __write_profile_to_csv(
    h: ASH, node: int, path: str, delimiter: str = ",", append: bool = False
) -> None:
    """

    :param h:
    :param node:
    :param path:
    :param delimiter:
    :param append:
    :return:
    """

    if append:
        op = open(path, "a")
    else:
        op = open(path, "w")

    with op as f:
        for tid in h.get_node_presence(node):
            res = f"{node}{delimiter}{tid}"
            profiles = h.get_node_profile(node, tid)
            descr = profiles.get_attributes()
            for k in sorted(descr):
                res = f"{res}{delimiter}{descr[k]}"
            f.write(f"{res}\n")


def write_profiles_to_csv(h: ASH, path: str, delimiter: str = ",") -> None:
    """

    :param h:
    :param path:
    :param delimiter:
    :return:
    """

    head = True
    for node in h.node_iterator():
        if head:
            prof = h.get_node_profile(node)
            heads = list(prof.get_attributes().keys())
            heads.remove("t")
            heads = f"{delimiter}".join(sorted(heads))
            head = f"node_id,tid,{heads}\n"

            with open(path, "w") as f:
                f.write(head)
            head = False

        __write_profile_to_csv(h, node, path, delimiter, append=True)


def read_profiles_from_csv(path: str, delimiter: str = ",") -> dict:
    """

    :param path:
    :param delimiter:
    :return:
    """
    res = {}
    with open(path) as f:
        head = f.readline().rstrip().split(delimiter)

        for row in f:
            row = row.rstrip().split(delimiter)
            prof = {head[i]: row[i] for i in range(2, len(row))}
            if int(row[0]) not in res:
                res[int(row[0])] = {int(row[1]): NProfile(node_id=int(row[0]), **prof)}
            else:
                res[int(row[0])][int(row[1])] = NProfile(node_id=int(row[0]), **prof)

    return res


def __write_profile_to_json(
    profile: NProfile, tid: int, path: str, compress: bool = False, append: bool = False
) -> None:
    """

    :param profile:
    :param path:
    :param compress:
    :param append:
    :return:
    """

    js_dmp = profile.to_dict()
    js_dmp["t"] = tid

    if compress:
        op = gzip.open
    else:
        op = open

    if not append:
        with op(path, "wt") as f:
            f.write(f"{json.dumps(js_dmp)}\n")
    else:
        with op(path, "at") as f:
            f.write(f"{json.dumps(js_dmp)}\n")


def write_profiles_to_jsonl(a: ASH, path: str, compress: bool = False) -> None:
    """

    :param a:
    :param path:
    :param compress:
    :return:
    """
    for node in a.node_iterator():
        for tid in a.get_node_presence(node):
            profile = a.get_node_profile(node, tid)
            __write_profile_to_json(profile, tid, path, compress, append=True)


def read_profiles_from_jsonl(path: str, compress: bool = False) -> dict:
    """

    :param path:
    :param compress:
    :return:
    """
    if compress:
        op = gzip.open
    else:
        op = open

    res = {}
    with op(path) as f:
        for l in f:
            rep = json.loads(l)

            if rep["node_id"] not in res:
                res[rep["node_id"]] = {
                    rep["t"]: NProfile(rep["node_id"], **rep["attrs"])
                }
            else:
                res[rep["node_id"]][rep["t"]] = NProfile(rep["node_id"], **rep["attrs"])

    return res


def write_sh_to_csv(h: ASH, path: str) -> None:
    """

    :param h:
    :param path:
    :return:
    """
    with open(path, "w") as o:
        o.write("nodes\tstart,end\n")
        for he in h.hyperedge_id_iterator():
            attrs = h.get_hyperedge_attributes(he)
            for span in attrs["t"]:
                nodes = [str(n) for n in attrs["nodes"]]
                desc = ",".join(nodes)
                desc = f"{desc}\t{span[0]},{span[1]}\n"
                o.write(desc)


def read_sh_from_csv(path: str) -> ASH:
    """

    :param path:
    :param delimiter:
    :param compress:
    :return:
    """
    a = ASH(hedge_removal=True)

    with open(path) as f:
        _ = f.readline()
        for row in f:
            nodes, span = row.rstrip().split("\t")
            nodes = [int(n) for n in nodes.split(",")]
            span = [int(t) for t in span.split(",")]
            a.add_hyperedge(nodes, start=span[0], end=span[1])
    return a


def write_ash_to_json(h: ASH, path: str, compress: bool = False) -> None:
    """

    :param h:
    :param path:
    :param compress:
    :return:
    """
    js_dmp = json.dumps(h.to_dict(), indent=2)

    if compress:
        op = gzip.open
    else:
        op = open

    with op(path, "wt") as f:
        f.write(js_dmp)


def read_ash_from_json(path: str, compress: bool = False) -> ASH:
    """

    :param path:
    :param compress:
    :return:
    """
    h = ASH(hedge_removal=True)

    if compress:
        op = gzip.open
    else:
        op = open

    with op(path, "rt") as f:
        data = json.loads(f.read())

    for _, v in data["hedges"].items():
        nodes = v["nodes"]
        spans = v["t"]
        for span in spans:
            h.add_hyperedge(nodes, start=span[0], end=span[1])

    for nid, attrs in data["nodes"].items():
        nid = int(nid)
        spans = attrs["t"]
        for span in spans:
            h.add_node(nid, span[0], span[1])

        keys = list(attrs.keys())
        keys.remove("t")

        t_to_values = {}

        for key in keys:
            values = attrs[key]
            for t, v in values.items():
                if t in t_to_values:
                    t_to_values[t][key] = v
                else:
                    t_to_values[t] = {key: v}

        for tid, prof in t_to_values.items():
            h.add_node(nid, int(tid), attr_dict=NProfile(node_id=nid, **prof))

    return h
