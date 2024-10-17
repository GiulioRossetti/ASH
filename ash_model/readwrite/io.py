import gzip
import json

from ash_model import ASH, NProfile

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
        for tid in h.node_presence(node):
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
    for node in h.nodes():
        if head:
            prof = h.get_node_profile(node)
            heads = list(prof.get_attributes().keys())
            heads = f"{delimiter}".join(sorted(heads))
            head = f"node_id{delimiter}tid{delimiter}{heads}\n"

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
    js_dmp["tid"] = tid

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
    for node in a.nodes():
        for tid in a.node_presence(node):
            profile = a.get_node_profile(node, tid)
            __write_profile_to_json(profile, tid, path, compress, append=True)


def read_profiles_from_jsonl(path: str, compress: bool = False) -> dict:
    """
    Read a dictionary of node profiles from a JSONL file.
    The dictionary maps node IDs to dictionaries of timestamps to node profiles.

    :param path: Path to the JSONL file.
    :param compress: If True, the file is assumed to be compressed using gzip.
    :return: Dictionary of node profiles.
    """
    if compress:
        op = gzip.open
    else:
        op = open

    res = {}
    with op(path) as f:
        for l in f:
            data = json.loads(l)
            node_id = data["node_id"]
            tid = data["tid"]
            del data["node_id"]
            del data["tid"]
            res.setdefault(node_id, {})[tid] = NProfile(node_id=node_id, **data)

    return res


def write_sh_to_csv(h: ASH, path: str) -> None:
    """
    Write a list of timestamped hyperedges to a CSV file.
    Does not support attributes.
    The CSV file will have the following format:
    n1,n2,...\tstart,end

    :param h: ASH object to write.
    :param path: Path to the CSV file.
    :return: None
    """
    with open(path, "w") as o:
        o.write("nodes\tstart,end\n")
        for he in h.hyperedges():
            presence = h.hyperedge_presence(he, as_intervals=True)
            nodes = h.get_hyperedge_nodes(he)
            for span in presence:
                desc = ",".join(nodes)
                desc = f"{desc}\t{span[0]},{span[1]}\n"
                o.write(desc)


def read_sh_from_csv(path: str) -> ASH:
    """
    Read a list of timestamped hyperedges from a CSV file.
    Does not support attributes.

    :param path: Path to the CSV file.
    :return:
    """
    a = ASH()

    with open(path) as f:
        _ = f.readline()
        for row in f:
            nodes, span = row.rstrip().split("\t")
            nodes = [int(n) for n in nodes.split(",")]
            span = [int(s) for s in span.split(",")]
            a.add_hyperedge(nodes, start=span[0], end=span[1])
    return a


def write_ash_to_json(h: ASH, path: str, compress: bool = False) -> None:
    """
    Write an ASH object to a JSON file.

    :param h: ASH object to write.
    :param path: Path to the JSON file.
    :param compress: If True, the file will be compressed using gzip.
    :return: None
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
    Read an ASH object from a JSON file.

    :param path: Path to the JSON file.
    :param compress: If True, the file is assumed to be compressed using gzip.
    :return: ASH object
    """
    h = ASH()

    if compress:
        op = gzip.open
    else:
        op = open

    with op(path, "rt") as f:
        data = json.loads(f.read())

    for he_id, edge_data in data["hyperedges"].items():
        for span in edge_data["presence"]:
            for t in range(span[0], span[1] + 1):
                h.add_hyperedge(
                    edge_data["nodes"],
                    start=t,
                    **{k: v for k, v in edge_data[t].items() if k != "presence"},
                )

    for node_id, node_data in data["nodes"].items():
        t_to_attrs = {}
        for attr_name, time_to_attr in node_data.items():
            if attr_name != "presence":
                for t, attr in time_to_attr.items():
                    t_to_attrs.setdefault(t, {})[attr_name] = time_to_attr[t]

        for span in node_data["presence"]:
            for t in range(span[0], span[1] + 1):
                h.add_node(
                    node_id,
                    start=t,
                    **t_to_attrs.get(t, {}),
                )

    return h
