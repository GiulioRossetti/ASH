from __future__ import annotations

"""presence_store.py
~~~~~~~~~~~~~~~~~~~~
Tiny abstraction layer that lets **ASH** swap the internal representation of
*temporal presence* without touching the public API.

Only the very small subset of dictionary behaviour used by ``ASH`` is
re‑implemented:

* ``__getitem__(t)`` – return the set of IDs present at *t* (read‑only).
* ``setdefault(t, set())`` – return a *mutable* set‑like proxy so that calls
  like ``self._snapshots.setdefault(t, set()).add(hid)`` keep working.
* ``get(t, default)`` – same semantics as ``dict.get``.
* ``keys()`` – iterable of snapshot indices.

Two concrete stores are provided:

* :class:`DensePresenceStore` – thin subclass of ``defaultdict(set)`` that
    keeps a *dense* mapping ``time → set[id]``.  This
    is the default behavior.
* :class:`IntervalPresenceStore` – keeps ``id → list[(start, end)]`` disjoint
  intervals and a tiny ``time_counts`` map so that ``keys()`` is cheap.

Switching the back‑end is as simple as:

>>> h = ASH(backend="interval")

No other public API changed.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Set, Tuple

###############################################################################
# Abstract façade
###############################################################################

class PresenceStore(ABC):
    """Minimal dict‑like interface required by :class:`ASH`."""

    # ---------------------------------------------------------------------
    # Required by ASH
    # ---------------------------------------------------------------------

    @abstractmethod
    def __getitem__(self, t: int) -> Set[int]:
        """Return **read‑only** snapshot set for *t*."""

    @abstractmethod
    def setdefault(self, t: int, default: Set[int]) -> "_SnapshotMutable":
        """Return mutable set‑like view (creates snapshot if absent)."""

    @abstractmethod
    def get(self, t: int, default):
        """Dict‑style *get*."""

    @abstractmethod
    def keys(self) -> Iterable[int]:
        """Return iterable of snapshot indices."""

    # ------------------------------------------------------------------
    # Convenience – these are never called directly by ASH but make the
    # façade quack like a normal dict.
    # ------------------------------------------------------------------

    def __contains__(self, key: int) -> bool:  # pragma: no cover – trivial
        return key in self.keys()

    def __iter__(self) -> Iterator[int]:  # pragma: no cover – trivial
        return iter(self.keys())

    def __len__(self) -> int:  # pragma: no cover – trivial
        return len(list(self.keys()))

###############################################################################
# Dense implementation (status quo)
###############################################################################

class DensePresenceStore(defaultdict, PresenceStore):
    """Keep the original *dense* mapping ``time → set[id]`` intact."""

    def __init__(self):
        super().__init__(set)

    # defaultdict already provides all behaviours we need.  The overrides
    # below are just for static typing clarity.

    def __getitem__(self, t: int) -> Set[int]:  # type: ignore[override]
        return super().__getitem__(t)

    def setdefault(self, t: int, default: Set[int]) -> Set[int]:  # type: ignore[override]
        return super().setdefault(t, default)

###############################################################################
# Interval implementation 
###############################################################################

class _SnapshotMutable(set):
    """A *mutable* view returned by :pymeth:`IntervalPresenceStore.setdefault`."""

    __slots__ = ("_store", "_time")

    def __init__(self, store: "IntervalPresenceStore", time: int, data: Set[int]):
        super().__init__(data)  # materialised copy so we can do normal set ops
        self._store = store
        self._time = time

    # Mutators – keep the interval representation in sync -----------------

    def add(self, element: int):  # type: ignore[override]
        if element not in self:
            super().add(element)
            self._store._add_occurrence(element, self._time)

    def discard(self, element: int):  # type: ignore[override]
        if element in self:
            super().discard(element)
            self._store._remove_occurrence(element, self._time)

    def remove(self, element: int):  # type: ignore[override]
        if element not in self:
            raise KeyError(element)
        self.discard(element)

class IntervalPresenceStore(PresenceStore):
    """Sparse *interval* representation.

    Internally we keep:

    * ``_intervals`` – mapping ``id → List[(start, end)]`` (sorted, disjoint).
    * ``_time_counts`` – small helper ``time → #ids alive`` so that
      :pymeth:`keys` is O(1) instead of scanning every interval on demand.
    """

    def __init__(self):
        self._intervals: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self._time_counts: Dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Public façade – dict‑like behaviour expected by ASH
    # ------------------------------------------------------------------

    def __getitem__(self, t: int) -> Set[int]:  # noqa: Dunder – matches dict API
        """Return *read‑only* snapshot (materialised as an ordinary set)."""
        return self._materialise(t)

    def setdefault(self, t: int, default: Set[int]):  # noqa: Dunder
        """Return *mutable* view for snapshot *t* (creates if absent)."""
        # Ensures time bucket exists so later *discard* knows "empty" means
        # absence, not "store unaware".
        if t not in self._time_counts:
            self._time_counts[t] = 0  # really empty snapshot for now
        return _SnapshotMutable(self, t, self._materialise(t))

    def get(self, t: int, default):  # noqa: Dunder – dict API
        return self._materialise(t) if t in self else default

    def keys(self) -> Iterable[int]:  # noqa: Dunder – dict API
        return self._time_counts.keys()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # ---------- helpers for snapshot (de)materialisation -----------------

    def _materialise(self, t: int) -> Set[int]:
        """Compute *set* of IDs alive at time *t* (cheap in sparse data)."""
        present: Set[int] = set()
        for hid, intervals in self._intervals.items():
            # Intervals are sorted; break early if we overshoot *t*.
            for start, end in intervals:
                if start > t:
                    break
                if start <= t <= end:
                    present.add(hid)
                    break  # no need to look at later intervals for this *hid*
        return present

    # ---------- mutators -------------------------------------------------

    def _add_occurrence(self, hid: int, t: int) -> None:
        """Insert a *single* time‑point into ``hid``'s interval list."""
        intervals = self._intervals[hid]

        if not intervals:  # first ever
            intervals.append((t, t))
        else:
            for i, (s, e) in enumerate(intervals):
                if s <= t <= e:
                    break  # already covered – nothing to do
                # Extend forward
                if t == e + 1:
                    intervals[i] = (s, t)
                    # Merge with next if adjacent (…][…)
                    if i + 1 < len(intervals) and intervals[i + 1][0] == t + 1:
                        n_s, n_e = intervals.pop(i + 1)
                        intervals[i] = (s, n_e)
                    break
                # Extend backward
                if t == s - 1:
                    intervals[i] = (t, e)
                    # Merge with previous if adjacent ([…][…)
                    if i - 1 >= 0 and intervals[i - 1][1] == t - 1:
                        p_s, p_e = intervals.pop(i - 1)
                        intervals[i - 1] = (p_s, e)
                    break
                if t < s - 1:
                    intervals.insert(i, (t, t))
                    break
            else:
                intervals.append((t, t))

        # Update time counter ------------------------------------------------
        self._time_counts[t] += 1

    def _remove_occurrence(self, hid: int, t: int) -> None:
        """Remove a single time‑point from ``hid``'s intervals (if present)."""
        intervals = self._intervals.get(hid, [])
        for i, (s, e) in enumerate(intervals):
            if s <= t <= e:
                if s == e == t:  # whole interval goes away
                    intervals.pop(i)
                elif t == s:  # shrink from left
                    intervals[i] = (s + 1, e)
                elif t == e:  # shrink from right
                    intervals[i] = (s, e - 1)
                else:  # split interval in two
                    intervals[i] = (s, t - 1)
                    intervals.insert(i + 1, (t + 1, e))
                break  # done

        # Update time counter ------------------------------------------------
        if t in self._time_counts:
            self._time_counts[t] -= 1
            if self._time_counts[t] == 0:
                del self._time_counts[t]
