import gzip
import json
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional

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
    "write_hif",
    "read_hif",
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
            for k in sorted(descr.keys()):
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
            heads = delimiter.join(sorted(h.list_node_attributes().keys()))
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
    with open(path, "r") as f:
        # readl all
        x = f.readlines()
        print(x)
    with open(path) as f:
        head = f.readline().rstrip().split(delimiter)

        for row in f:
            row = row.rstrip().split(delimiter)
            prof = {head[i]: row[i] for i in range(2, len(row))}
            # convert types
            for k, v in prof.items():
                if v.isdigit():
                    prof[k] = int(v)
                elif v.replace(".", "", 1).isdigit():
                    prof[k] = float(v)
                # else keep as string
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
    Warning: there is no guarantee that the order of hyperedges will be preserved when reading back.

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
                desc = ",".join([str(n) for n in nodes])
                desc = f"{desc}\t{span[0]},{span[1]}\n"
                o.write(desc)


def read_sh_from_csv(path: str) -> ASH:
    """
    Read a list of timestamped hyperedges from a CSV file.
    Does not support attributes.
    Warning: there is no guarantee that the order of hyperedges will be preserved.

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

    # Add nodes
    attrs = defaultdict(lambda: defaultdict(dict))
    for node_id, node_data in data["nodes"].items():
        for tid, attr in node_data.items():
            attrs[int(node_id)][int(tid)] = attr
    h._node_attrs = attrs

    for _, edge_data in data["hedges"].items():
        for span in edge_data["attributes"]["_presence"]:
            for t in range(span[0], span[1] + 1):

                kwargs = {
                    k: v for k, v in edge_data["attributes"].items() if k != "_presence"
                }

                h.add_hyperedge(edge_data["nodes"], start=t, **kwargs)

    return h


def __to_hif(h: ASH, metadata: Optional[Dict[str, Any]] = None) -> None:

    hif: Dict[str, Any] = {}
    hif["network-type"] = "undirected"
    if metadata is not None:
        hif["metadata"] = metadata

    # --- Incidences: one record per node–edge membership -------------
    incidences: List[Dict[str, Any]] = []
    for hid in h.hyperedges():
        weight = h.get_hyperedge_weight(hid)
        for n in h.get_hyperedge_nodes(hid):
            rec = {"node": n, "edge": hid}
            if weight != 1:
                rec["weight"] = weight
            incidences.append(rec)
    hif["incidences"] = incidences

    # --- Nodes: include time‐varying attrs as intervals ---------------
    nodes_list: List[Dict[str, Any]] = []
    # first, get full set of node‐attr names
    all_node_attrs = set(h.list_node_attributes().keys())
    for n in h.nodes():
        attrs: Dict[str, Any] = {}
        # for each attr, build list of (start, end, value) intervals
        for attr in all_node_attrs:
            time_values = h.get_node_attribute(n, attr)  # {t: value}
            if not any(v is not None for v in time_values.values()):
                continue
            # collapse into contiguous spans with same value
            times = sorted(time_values.keys())
            spans: List[Tuple[int, int, Any]] = []
            s = times[0]
            e = s
            cur = time_values[s]
            for t in times[1:]:
                v = time_values[t]
                if v == cur and t == e + 1:
                    e = t
                else:
                    spans.append((s, e, cur))
                    s, e, cur = t, t, v
            spans.append((s, e, cur))
            attrs[attr] = spans
        # also include node presence intervals explicitly
        attrs["_presence"] = h.node_presence(n, as_intervals=True)
        nodes_list.append({"node": n, "attrs": attrs})
    hif["nodes"] = nodes_list

    # --- Edges: existing attrs + temporal presence ------------------
    edges_list: List[Dict[str, Any]] = []
    for hid in h.hyperedges():
        attrs = dict(h.get_hyperedge_attributes(hid))
        # embed temporal presence as intervals
        attrs["_presence"] = h.hyperedge_presence(hid, as_intervals=True)  # type: ignore[arg-type]
        edges_list.append({"edge": hid, "attrs": attrs})
    hif["edges"] = edges_list

    return hif


def __from_hif(data: dict) -> ASH:
    """
    Convert HIF data dictionary to ASH object.

    :param data: Dictionary containing HIF data

    :return: ASH object
    """
    h = ASH()

    if data.get("network-type") != "undirected":
        raise NotImplementedError("Only undirected hypernetworks are supported.")

    # Process nodes and their attributes
    if "nodes" in data:
        for node_data in data["nodes"]:
            node_id = node_data["node"]
            attrs = node_data.get("attrs", {})

            # Choose a reference time for static attributes:
            # prefer the start of the first presence interval, else 0
            t0 = 0
            presence_intervals = attrs.get("_presence", [])
            if isinstance(presence_intervals, list) and len(presence_intervals) > 0:
                first = presence_intervals[0]
                if isinstance(first, (list, tuple)) and len(first) == 2:
                    try:
                        t0 = int(first[0])
                    except Exception:
                        t0 = 0
                elif isinstance(first, (int,)):
                    t0 = int(first)

            # Process each attribute (except _presence which is handled separately)
            for attr_name, spec in attrs.items():
                if attr_name == "_presence":
                    continue  # Skip presence; node presence is implied by attrs/edges

                # Case 1: temporal attribute encoded as list of (start, end, value)
                if isinstance(spec, list) and all(
                    isinstance(x, (list, tuple)) and len(x) == 3 for x in spec
                ):
                    for start, end, value in spec:
                        # for each t in the closed interval [start, end]
                        for t in range(int(start), int(end) + 1):
                            if not hasattr(h, "_node_attrs"):
                                h._node_attrs = defaultdict(lambda: defaultdict(dict))
                            h._node_attrs[node_id][t][attr_name] = value
                else:
                    # Case 2: static attribute -> assign at a single time instant t0
                    if not hasattr(h, "_node_attrs"):
                        h._node_attrs = defaultdict(lambda: defaultdict(dict))
                    h._node_attrs[node_id][t0][attr_name] = spec

    # Process hyperedges
    if "edges" in data:
        for edge_data in data["edges"]:
            edge_id = edge_data["edge"]
            attrs = edge_data.get("attrs", {})

            # Get presence specification and normalize to list of (start,end)
            presence_spec = attrs.get("_presence", None)
            presence_intervals: List[Tuple[int, int]] = []
            if presence_spec is None or (
                isinstance(presence_spec, list) and len(presence_spec) == 0
            ):
                # No presence provided -> treat as static at t=0
                presence_intervals = [(0, 0)]
            elif isinstance(presence_spec, list):
                # Allow either list of intervals or list of instants
                if all(
                    isinstance(x, (list, tuple)) and len(x) == 2 for x in presence_spec
                ):
                    presence_intervals = [(int(s), int(e)) for s, e in presence_spec]  # type: ignore[misc]
                elif all(isinstance(x, (int,)) for x in presence_spec):
                    presence_intervals = [(int(t), int(t)) for t in presence_spec]  # type: ignore[misc]
                else:
                    # Fallback: treat as static at t=0
                    presence_intervals = [(0, 0)]
            else:
                # Unexpected type -> default to static at t=0
                presence_intervals = [(0, 0)]

            # Get other attributes (excluding _presence)
            edge_attrs = {k: v for k, v in attrs.items() if k != "_presence"}

            # We need to get the nodes for this edge from incidences
            edge_nodes = []
            if "incidences" in data:
                edge_nodes = [
                    inc["node"] for inc in data["incidences"] if inc["edge"] == edge_id
                ]
                # Remove duplicates while preserving order
                seen = set()
                edge_nodes = [x for x in edge_nodes if not (x in seen or seen.add(x))]

            # Add hyperedge for each presence interval
            for start, end in presence_intervals:
                h.add_hyperedge(edge_nodes, start=start, end=end, **edge_attrs)

    return h


def write_hif(
    h: ASH, path: str, metadata: Optional[Dict[str, Any]] = None, compress: bool = False
) -> None:
    """
    Write an ASH object to a HIF file.

    :param h: ASH object to write.
    :param path: Path to the HIF file.
    :param compress: If True, the file will be compressed using gzip.

    :return: None
    """
    hif = __to_hif(h, metadata)

    if compress:
        op = gzip.open
    else:
        op = open

    with op(path, "wt") as f:
        json.dump(hif, f, indent=2)


def read_hif(path: str, compress: bool = False) -> ASH:
    """
    Read an ASH object from a HIF file.

    :param path: Path to the HIF file.
    :param compress: If True, the file is assumed to be compressed using gzip.

    :return: ASH object
    """
    if compress:
        op = gzip.open
    else:
        op = open

    with op(path, "rt") as f:
        data = json.loads(f.read())

    h = __from_hif(data)

    return h
