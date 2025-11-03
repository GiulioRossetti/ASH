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
    """A *mutable* view returned by :meth:`IntervalPresenceStore.setdefault`."""

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

    ``_intervals`` – mapping ``id → List[(start, end)]`` (sorted, disjoint).
    ``_starts`` – mapping ``id → List[start]`` (parallel to intervals, for bisect).
    ``_time_events`` – difference array ``time → delta`` for O(1) interval updates.
    ``_time_counts`` – rebuilt lazily from ``_time_events`` when needed.

    Optimizations:
    - Binary search (bisect) for presence checks: O(log k) per id.
    - Event-diff updates: O(1) per interval add/remove (vs O(length)).
    """

    def __init__(self):
        self._intervals: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self._starts: Dict[int, List[int]] = defaultdict(list)  # for bisect
        self._time_events: Dict[int, int] = defaultdict(int)  # difference array
        self._time_counts: Dict[int, int] = {}
        self._time_counts_valid: bool = True  # lazy rebuild flag

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
        self._ensure_time_counts()
        return self._time_counts.keys()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_time_counts(self) -> None:
        """Rebuild _time_counts from _time_events (lazy reconstruction)."""
        if self._time_counts_valid:
            return
        self._time_counts.clear()
        if not self._time_events:
            self._time_counts_valid = True
            return

        # Compute running count and track all times with positive count
        running = 0
        sorted_events = sorted(self._time_events.items())
        if not sorted_events:
            self._time_counts_valid = True
            return

        # Build ranges where count > 0
        current_t = sorted_events[0][0]
        for t, delta in sorted_events:
            # Fill in all times from current_t to t-1 with running count
            if running > 0:
                for tt in range(current_t, t):
                    self._time_counts[tt] = running
            running += delta
            current_t = t

        self._time_counts_valid = True

    # ---------- helpers for snapshot (de)materialisation -----------------

    def _materialise(self, t: int) -> Set[int]:
        """Compute *set* of IDs alive at time *t* using bisect (O(log k) per id)."""
        import bisect

        present: Set[int] = set()
        for hid, intervals in self._intervals.items():
            if not intervals:
                continue
            starts = self._starts[hid]
            # bisect_right(starts, t) gives index of first start > t
            i = bisect.bisect_right(starts, t) - 1
            if i >= 0:
                s, e = intervals[i]
                if s <= t <= e:
                    present.add(hid)
        return present

    # ---------- mutators -------------------------------------------------

    def _add_occurrence(self, hid: int, t: int) -> None:
        """Insert a *single* time‑point into ``hid``'s interval list."""
        intervals = self._intervals[hid]
        starts = self._starts[hid]

        # Check if already present
        already_present = False
        for s, e in intervals:
            if s <= t <= e:
                already_present = True
                break

        if already_present:
            return  # nothing to do

        # Find merge candidates and apply
        if not intervals:  # first ever
            intervals.append((t, t))
            starts.append(t)
            self._time_events[t] += 1
            self._time_events[t + 1] -= 1
            self._time_counts_valid = False
            return

        merged = False
        for i, (s, e) in enumerate(intervals):
            # Extend forward
            if t == e + 1:
                intervals[i] = (s, t)
                # Merge with next if adjacent
                if i + 1 < len(intervals) and intervals[i + 1][0] == t + 1:
                    n_s, n_e = intervals.pop(i + 1)
                    starts.pop(i + 1)
                    intervals[i] = (s, n_e)
                    # Net event: only t gets added (merging doesn't change event counts elsewhere)
                else:
                    # Just extending: add t
                    self._time_events[t] += 1
                    self._time_events[t + 1] -= 1
                merged = True
                break
            # Extend backward
            if t == s - 1:
                intervals[i] = (t, e)
                starts[i] = t
                # Merge with previous if adjacent
                if i - 1 >= 0 and intervals[i - 1][1] == t - 1:
                    p_s, p_e = intervals.pop(i - 1)
                    starts.pop(i - 1)
                    intervals[i - 1] = (p_s, e)
                    starts[i - 1] = p_s
                    # Net event: only t gets added
                else:
                    self._time_events[t] += 1
                    self._time_events[t + 1] -= 1
                merged = True
                break
            if t < s - 1:
                intervals.insert(i, (t, t))
                starts.insert(i, t)
                self._time_events[t] += 1
                self._time_events[t + 1] -= 1
                merged = True
                break

        if not merged:
            intervals.append((t, t))
            starts.append(t)
            self._time_events[t] += 1
            self._time_events[t + 1] -= 1

        self._time_counts_valid = False

    def _remove_occurrence(self, hid: int, t: int) -> None:
        """Remove a single time‑point from ``hid``'s intervals (if present)."""
        intervals = self._intervals.get(hid, [])
        starts = self._starts.get(hid, [])

        found = False
        for i, (s, e) in enumerate(intervals):
            if s <= t <= e:
                found = True
                if s == e == t:  # whole interval goes away
                    intervals.pop(i)
                    starts.pop(i)
                elif t == s:  # shrink from left
                    intervals[i] = (s + 1, e)
                    starts[i] = s + 1
                elif t == e:  # shrink from right
                    intervals[i] = (s, e - 1)
                else:  # split interval in two
                    intervals[i] = (s, t - 1)
                    intervals.insert(i + 1, (t + 1, e))
                    starts.insert(i + 1, t + 1)
                break  # done

        if found:
            # Update time events (difference array)
            self._time_events[t] -= 1
            self._time_events[t + 1] += 1
            self._time_counts_valid = False

    def _add_interval(self, hid: int, start: int, end: int) -> None:
        """Insert the entire [start,end] span into ``hid``'s interval list in one pass."""
        intervals = self._intervals[hid]
        starts = self._starts[hid]

        # Track old coverage before merge
        old_intervals = list(intervals)

        new_s, new_e = start, end
        i = 0
        # 1) Merge any overlapping or adjacent existing intervals
        merged_indices = []
        while i < len(intervals):
            s, e = intervals[i]
            if e + 1 < new_s:
                i += 1
                continue
            if s - 1 > new_e:
                break
            # overlapping or adjacent -> absorb
            new_s = min(new_s, s)
            new_e = max(new_e, e)
            merged_indices.append(i)
            i += 1

        # Remove merged intervals in reverse to preserve indices
        for idx in reversed(merged_indices):
            intervals.pop(idx)
            starts.pop(idx)

        # 2) Insert the merged interval at the correct position
        insert_pos = len([s for s in starts if s < new_s])
        intervals.insert(insert_pos, (new_s, new_e))
        starts.insert(insert_pos, new_s)

        # 3) Update events: add for [new_s, new_e], subtract for old intervals that were merged
        self._time_events[new_s] += 1
        self._time_events[new_e + 1] -= 1

        for idx in merged_indices:
            old_s, old_e = old_intervals[idx]
            self._time_events[old_s] -= 1
            self._time_events[old_e + 1] += 1

        self._time_counts_valid = False

    def _remove_interval(self, hid: int, start: int, end: int) -> None:
        """Remove the entire [start,end] span from ``hid``'s interval list in one pass."""
        intervals = self._intervals.get(hid, [])
        starts = self._starts.get(hid, [])

        # Track old intervals that will be affected
        old_intervals = list(intervals)
        affected_indices = []

        i = 0
        to_insert = []
        indices_to_remove = []
        while i < len(intervals):
            s, e = intervals[i]
            if e < start or s > end:
                i += 1
                continue
            # overlapping: may need to split or shrink
            affected_indices.append(i)
            before = (s, start - 1) if s < start else None
            after = (end + 1, e) if e > end else None
            indices_to_remove.append(i)
            if before:
                to_insert.append((i, before))
            if after:
                to_insert.append((i, after))
            i += 1

        # Remove old intervals in reverse
        for idx in reversed(indices_to_remove):
            intervals.pop(idx)
            starts.pop(idx)

        # Insert new split intervals
        for idx, (new_s, new_e) in to_insert:
            insert_pos = len([s for s in starts if s < new_s])
            intervals.insert(insert_pos, (new_s, new_e))
            starts.insert(insert_pos, new_s)

        # Update events: remove [start, end], add back the split pieces
        self._time_events[start] -= 1
        self._time_events[end + 1] += 1

        for idx in affected_indices:
            old_s, old_e = old_intervals[idx]
            self._time_events[old_s] -= 1
            self._time_events[old_e + 1] += 1

        for _, (new_s, new_e) in to_insert:
            self._time_events[new_s] += 1
            self._time_events[new_e + 1] -= 1

        self._time_counts_valid = False
