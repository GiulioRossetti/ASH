from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ash_model.classes import ASH  # type: ignore

__all__ = [
    "plot_hyperedge_activity_series",
    "plot_node_activity_series",
    "plot_presence_timeline",
    "plot_inter_event_time_distribution",
    "plot_hyperedge_lifespan_distribution",
    "plot_node_lifespan_distribution",
]


# ---------------------------------------------------------------------------
# Helper utilities (internal)
# ---------------------------------------------------------------------------


def _get_ax(kwargs):
    if "ax" in kwargs and kwargs["ax"] is not None:
        return kwargs["ax"]
    return plt.gca()


# ---------------------------------------------------------------------------
# Public plotting helpers
# ---------------------------------------------------------------------------


def plot_hyperedge_activity_series(h: ASH, normalize: bool = False, **kwargs):
    """Plot the number of active hyperedges at each temporal snapshot.

    :param h: ASH instance.
    :param normalize: If True divide activity by the maximum (y in [0,1]).
    :param kwargs: Matplotlib customisation (``color``, ``ax`` …).
    :return: Matplotlib Axes with the line plot.
    """

    ax = _get_ax(kwargs)
    tids = h.temporal_snapshots_ids()
    if not tids:
        return ax
    activity = [h.number_of_hyperedges(t) for t in tids]
    if normalize and any(activity):
        m = max(activity)
        activity = [a / m for a in activity]
    ax.plot(
        tids,
        activity,
        marker="o",
        linewidth=1,
        ms=3,
        **{k: v for k, v in kwargs.items() if k != "ax"},
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Activity" + (" (normalized)" if normalize else ""))
    ax.set_title("Hyperedge activity over time")
    return ax


# ---------------------------------------------------------------------------
# Backward-compat aliases (kept for older tests/examples)
# ---------------------------------------------------------------------------


def plot_node_activity_series(
    h: ASH,
    *,
    normalize: bool = False,
    **kwargs,
):
    """Plot the activity over time for selected nodes.

    :param h: ASH instance.
    :param normalize: If True divide node's activity by its maximum (y in [0,1]).
    :param kwargs: Matplotlib customisation (``color``, ``ax`` …).
    :return: Matplotlib Axes with the line plot.
    """

    ax = _get_ax(kwargs)
    tids = h.temporal_snapshots_ids()
    if not tids:
        return ax

    activity = [h.number_of_nodes(t) for t in tids]
    if normalize and any(activity):
        m = max(activity)
        activity = [a / m for a in activity]
    ax.plot(
        tids,
        activity,
        marker="o",
        linewidth=1,
        ms=3,
        **{k: v for k, v in kwargs.items() if k != "ax"},
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Activity" + (" (normalized)" if normalize else ""))
    ax.set_title("Node activity over time")
    ax.legend()
    return ax


def plot_presence_timeline(
    h: ASH,
    *,
    hyperedges: Optional[Iterable[int]] = None,
    nodes: Optional[Iterable[int]] = None,
    **kwargs,
):
    """Plot a presence timeline (Gantt‑like) for given hyperedges or nodes.

    One of ``hyperedges`` or ``nodes`` must be provided.  If both are
    provided, ``hyperedges`` takes precedence.

    :param h: ASH instance.
    :param hyperedges: Iterable of hyperedge IDs to plot.
    :param nodes: Iterable of node IDs to plot.
    :param kwargs: Matplotlib customisation (``color``, ``ax`` …).
    :return: Matplotlib Axes with the timeline plot.
    """

    ax = _get_ax(kwargs)
    if hyperedges is not None:
        items = list(hyperedges)
        presences = [
            (hid, h.hyperedge_presence(hid))
            for hid in items
            if h.hyperedge_presence(hid)
        ]
        item_type = "Hyperedge"
    elif nodes is not None:
        items = list(nodes)
        presences = [
            (nid, h.node_presence(nid)) for nid in items if h.node_presence(nid)
        ]
        item_type = "Node"
    else:
        raise ValueError("One of 'hyperedges' or 'nodes' must be provided.")

    if not presences:
        return ax

    for i, (item_id, times) in enumerate(presences):
        ax.vlines(
            times,
            i + 0.5,
            i + 1.5,
            linewidth=4,
            **{k: v for k, v in kwargs.items() if k != "ax"},
        )

    ax.set_yticks(np.arange(1, len(presences) + 1))
    ax.set_yticklabels([str(item_id) for item_id, _ in presences])
    ax.set_xlabel("Time")
    ax.set_ylabel(item_type + " ID")
    ax.set_title(f"{item_type} presence timeline")
    return ax


def plot_inter_event_time_distribution(h: ASH, **kwargs):
    """Plot distribution of inter‑event times for hyperedge activations.

    We define an activation as a ``+`` event produced by
    ``ASH.stream_interactions``.  Inter‑event gaps are differences between
    consecutive activation times (across all hyperedges).

    :param h: ASH instance.
    :param kwargs: Matplotlib bar customisation (``color``, ``ax`` …).
    :return: Axes
    """

    ax = _get_ax(kwargs)
    events = [t for (t, _hid, et) in h.stream_interactions() if et == "+"]
    if len(events) < 2:
        return ax
    events = sorted(events)
    diffs = np.diff(events)
    unique, counts = np.unique(diffs, return_counts=True)
    ax.bar(
        unique,
        counts,
        width=0.8,
        alpha=0.6,
        **{k: v for k, v in kwargs.items() if k != "ax"},
    )
    ax.set_xlabel("Inter‑event time Δt")
    ax.set_ylabel("Count")
    ax.set_title("Inter‑event time distribution")
    return ax


def plot_hyperedge_lifespan_distribution(h: ASH, **kwargs):
    """Histogram of hyperedge lifespans (duration in snapshots).

    For each hyperedge we compute ``(last_presence - first_presence + 1)``.

    :param h: ASH instance.
    :param kwargs: Matplotlib customisation (``bins``, ``color``, ``ax`` …).
    :return: Axes
    """

    ax = _get_ax(kwargs)
    lifespans: List[int] = []
    for hid in h.hyperedges() if hasattr(h, "hyperedges") else []:  # fallback safety
        pres = h.hyperedge_presence(hid) if hasattr(h, "hyperedge_presence") else []
        if pres:
            lifespans.append(max(pres) - min(pres) + 1)
    if not lifespans:
        return ax
    bins = kwargs.get("bins", min(30, len(set(lifespans))))
    ax.hist(
        lifespans,
        bins=bins,
        alpha=0.6,
        **{k: v for k, v in kwargs.items() if k not in {"ax", "bins"}},
    )
    ax.set_xlabel("Lifespan (snapshots)")
    ax.set_ylabel("#Hyperedges")
    ax.set_title("Hyperedge lifespan distribution")
    return ax


def plot_node_lifespan_distribution(h: ASH, **kwargs):
    """Histogram of node lifespans (duration in snapshots).

    For each node we compute ``(last_presence - first_presence + 1)``.

    :param h: ASH instance.
    :param kwargs: Matplotlib customisation (``bins``, ``color``, ``ax`` …).
    :return: Axes
    """

    ax = _get_ax(kwargs)
    lifespans: List[int] = []
    for nid in h.nodes() if hasattr(h, "nodes") else []:  # fallback safety
        pres = h.node_presence(nid) if hasattr(h, "node_presence") else []
        if pres:
            lifespans.append(max(pres) - min(pres) + 1)
    if not lifespans:
        return ax
    bins = kwargs.get("bins", min(30, len(set(lifespans))))
    ax.hist(
        lifespans,
        bins=bins,
        alpha=0.6,
        **{k: v for k, v in kwargs.items() if k not in {"ax", "bins"}},
    )
    ax.set_xlabel("Lifespan (snapshots)")
    ax.set_ylabel("#Nodes")
    ax.set_title("Node lifespan distribution")
    return ax
