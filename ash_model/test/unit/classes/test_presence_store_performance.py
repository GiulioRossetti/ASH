import time
import unittest

from ash_model.classes.presence_store import DensePresenceStore, IntervalPresenceStore


class PresenceStoreCorrectnessTestCase(unittest.TestCase):
    """Verify that optimized IntervalPresenceStore produces identical output to Dense."""

    def test_materialise_single_interval(self):
        dense = DensePresenceStore()
        interval = IntervalPresenceStore()

        # Add hyperedge 1 present at times 0-5
        for t in range(6):
            dense.setdefault(t, set()).add(1)
        interval._add_interval(1, 0, 5)

        for t in range(-1, 7):
            self.assertEqual(dense[t], interval[t], f"Mismatch at t={t}")

    def test_materialise_multiple_intervals(self):
        dense = DensePresenceStore()
        interval = IntervalPresenceStore()

        # Add hyperedge 1: [0-2], [5-7]
        for t in [0, 1, 2, 5, 6, 7]:
            dense.setdefault(t, set()).add(1)
        interval._add_interval(1, 0, 2)
        interval._add_interval(1, 5, 7)

        for t in range(-1, 10):
            self.assertEqual(dense[t], interval[t], f"Mismatch at t={t}")

    def test_materialise_many_ids(self):
        dense = DensePresenceStore()
        interval = IntervalPresenceStore()

        # Add 100 hyperedges with different intervals
        for hid in range(100):
            start = hid * 2
            end = hid * 2 + 10
            for t in range(start, end + 1):
                dense.setdefault(t, set()).add(hid)
            interval._add_interval(hid, start, end)

        # Check snapshot at various times
        for t in [0, 5, 50, 100, 150, 200]:
            self.assertEqual(dense[t], interval[t], f"Mismatch at t={t}")

    def test_keys_after_add_interval(self):
        dense = DensePresenceStore()
        interval = IntervalPresenceStore()

        # Add hyperedge 1: [10-20]
        for t in range(10, 21):
            dense.setdefault(t, set()).add(1)
        interval._add_interval(1, 10, 20)

        self.assertEqual(set(dense.keys()), set(interval.keys()))

    def test_keys_after_remove_interval(self):
        dense = DensePresenceStore()
        interval = IntervalPresenceStore()

        # Add hid 1: [0-10], then remove [5-7]
        for t in range(11):
            dense.setdefault(t, set()).add(1)
        for t in [5, 6, 7]:
            dense[t].discard(1)
            # Remove empty snapshots manually
            if not dense[t]:
                del dense[t]

        interval._add_interval(1, 0, 10)
        interval._remove_interval(1, 5, 7)

        self.assertEqual(set(dense.keys()), set(interval.keys()))

    def test_setdefault_add_remove(self):
        dense = DensePresenceStore()
        interval = IntervalPresenceStore()

        # Add via setdefault (mimics ASH usage)
        dense.setdefault(0, set()).add(1)
        dense.setdefault(0, set()).add(2)
        dense.setdefault(1, set()).add(1)

        interval.setdefault(0, set()).add(1)
        interval.setdefault(0, set()).add(2)
        interval.setdefault(1, set()).add(1)

        self.assertEqual(dense[0], interval[0])
        self.assertEqual(dense[1], interval[1])
        self.assertEqual(set(dense.keys()), set(interval.keys()))

        # Remove via mutable view
        interval.setdefault(0, set()).discard(1)
        dense[0].discard(1)

        self.assertEqual(dense[0], interval[0])


class PresenceStorePerformanceTestCase(unittest.TestCase):
    """Benchmark optimized IntervalPresenceStore vs DensePresenceStore."""

    def test_materialise_performance_many_ids(self):
        """Measure time to materialise snapshot with many IDs and intervals."""
        interval = IntervalPresenceStore()
        n_ids = 1000
        intervals_per_id = 5

        # Setup: add many intervals
        for hid in range(n_ids):
            for i in range(intervals_per_id):
                start = hid * 100 + i * 20
                end = start + 10
                interval._add_interval(hid, start, end)

        # Benchmark: materialise at various times
        times_to_check = [50, 5000, 10000, 50000, 100000]
        start_time = time.perf_counter()
        for t in times_to_check:
            snapshot = interval[t]
        elapsed = time.perf_counter() - start_time

        # Just ensure it completes in reasonable time and produces valid output
        self.assertIsInstance(snapshot, set)
        print(
            f"\n[PERF] Materialise {len(times_to_check)} snapshots with {n_ids} ids: {elapsed:.4f}s"
        )

    def test_add_interval_performance_long_spans(self):
        """Measure time to add long intervals (event-diff should be O(1))."""
        interval = IntervalPresenceStore()
        n_intervals = 1000
        span_length = 10000  # very long interval

        start_time = time.perf_counter()
        for i in range(n_intervals):
            interval._add_interval(i, 0, span_length)
        elapsed = time.perf_counter() - start_time

        # With event-diff, this should be fast (O(1) per interval)
        print(
            f"\n[PERF] Add {n_intervals} intervals of length {span_length}: {elapsed:.4f}s"
        )
        self.assertLess(elapsed, 1.0, "Adding intervals should be fast with event-diff")

    def test_keys_performance_after_many_intervals(self):
        """Measure time to call keys() after many interval operations."""
        interval = IntervalPresenceStore()
        n_ids = 500
        intervals_per_id = 10

        # Setup
        for hid in range(n_ids):
            for i in range(intervals_per_id):
                start = hid * 50 + i * 10
                end = start + 5
                interval._add_interval(hid, start, end)

        # Benchmark keys() (triggers lazy rebuild)
        start_time = time.perf_counter()
        keys_list = list(interval.keys())
        elapsed = time.perf_counter() - start_time

        print(
            f"\n[PERF] keys() after {n_ids*intervals_per_id} intervals: {elapsed:.4f}s, {len(keys_list)} keys"
        )
        self.assertIsInstance(keys_list, list)

    def test_comparative_dense_vs_interval(self):
        """Compare dense vs interval for typical workload."""
        n_ids = 200
        span_length = 50

        # Dense benchmark
        dense = DensePresenceStore()
        start_time = time.perf_counter()
        for hid in range(n_ids):
            for t in range(hid, hid + span_length):
                dense.setdefault(t, set()).add(hid)
        dense_add_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        for t in [0, 100, 200, 300]:
            _ = dense[t]
        dense_query_time = time.perf_counter() - start_time

        # Interval benchmark
        interval = IntervalPresenceStore()
        start_time = time.perf_counter()
        for hid in range(n_ids):
            interval._add_interval(hid, hid, hid + span_length - 1)
        interval_add_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        for t in [0, 100, 200, 300]:
            _ = interval[t]
        interval_query_time = time.perf_counter() - start_time

        print(
            f"\n[PERF] Dense: add={dense_add_time:.4f}s, query={dense_query_time:.4f}s"
        )
        print(
            f"[PERF] Interval: add={interval_add_time:.4f}s, query={interval_query_time:.4f}s"
        )

        # Interval add should be much faster for long spans
        self.assertLess(
            interval_add_time,
            dense_add_time,
            "Interval should be faster for adding long spans",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
