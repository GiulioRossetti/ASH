import unittest
import time
import random
from ash_model.classes import DensePresenceStore, IntervalPresenceStore, ASH


class TestDensePresenceStore(unittest.TestCase):
    def setUp(self):
        self.store = DensePresenceStore()

    def test_initially_empty(self):
        self.assertEqual(list(self.store.keys()), [])
        # get should return default when missing
        self.assertEqual(self.store.get(5, {"a"}), {"a"})

    def test_setdefault_and_item(self):
        # setdefault should create the key and return a mutable set
        snap = self.store.setdefault(10, set())
        self.assertIsInstance(snap, set)
        self.assertEqual(list(self.store.keys()), [10])
        # adding to snapshot
        snap.add(42)
        self.assertIn(42, self.store[10])
        # __getitem__ should return the same underlying set
        self.store[10].add(99)
        self.assertIn(99, snap)

    def test_contains_and_len(self):
        self.store.setdefault(3, set()).add(7)
        self.assertIn(3, self.store)
        self.assertEqual(len(self.store), 1)
        self.assertEqual(list(iter(self.store)), [3])


class TestIntervalPresenceStore(unittest.TestCase):
    def setUp(self):
        self.store = IntervalPresenceStore()

    def test_initially_empty(self):
        self.assertEqual(list(self.store.keys()), [])
        # get should return default when missing
        self.assertEqual(self.store.get(100, {"x"}), {"x"})
        # __getitem__ on missing t returns empty set
        self.assertEqual(self.store[100], set())

    def test_add_occurrence_via_setdefault(self):
        snap = self.store.setdefault(5, set())
        # add element 1 at time=5
        snap.add(1)
        self.assertIn(5, self.store.keys())
        # __getitem__ now shows that element
        self.assertEqual(self.store[5], {1})
        # adding twice has no duplicate effects
        snap.add(1)
        self.assertEqual(self.store[5], {1})

    def test_remove_occurrence_via_setdefault_discard(self):
        snap = self.store.setdefault(7, set())
        snap.add(10)
        self.assertIn(7, self.store.keys())
        # discard should remove and clean up keys
        snap.discard(10)
        self.assertEqual(self.store.get(7, set()), set())
        self.assertNotIn(7, self.store.keys())

    def test_materialise_read_only(self):
        # __getitem__ returns a materialised set; mutating it doesn't affect store
        snap = self.store.setdefault(2, set())
        snap.add(99)
        view = self.store[2]
        view.add(123)
        # underlying store unchanged by mutating view
        self.assertNotIn(123, self.store[2])
        self.assertIn(99, self.store[2])

    def test_interval_merging(self):
        # add element 1 at time=5
        snap5 = self.store.setdefault(5, set())
        snap5.add(1)
        # add same element at time=6 (adjacent), should merge interval to (5,6)
        snap6 = self.store.setdefault(6, set())
        snap6.add(1)
        self.assertEqual(self.store._intervals[1], [(5, 6)])

        # add element at time=4 (adjacent to start), should extend to (4,6)
        snap4 = self.store.setdefault(4, set())
        snap4.add(1)
        self.assertEqual(self.store._intervals[1], [(4, 6)])

        # add element at time=8 (non‐adjacent), should create a separate interval
        snap8 = self.store.setdefault(8, set())
        snap8.add(1)
        self.assertCountEqual(self.store._intervals[1], [(4, 6), (8, 8)])

    def test_add_interval_direct(self):
        # directly add a span [3,5] for hid=42
        self.store._add_interval(42, 3, 5)
        # interval list should reflect exactly one interval
        self.assertEqual(self.store._intervals[42], [(3, 5)])
        # keys() should return 3,4,5
        self.assertCountEqual(self.store.keys(), [3, 4, 5])
        # each timepoint should contain hid=42
        for t in (3, 4, 5):
            self.assertIn(42, self.store[t])

    def test_remove_interval_direct(self):
        # seed with a bigger interval, then cut out a middle piece
        self.store._add_interval(7, 1, 5)
        # remove timepoint 3 from hid=7
        self.store._remove_interval(7, 3, 3)
        # should split into [1,2] and [4,5]
        self.assertCountEqual(self.store._intervals[7], [(1, 2), (4, 5)])
        # keys now are 1,2,4,5
        self.assertCountEqual(self.store.keys(), [1, 2, 4, 5])
        # ensure 3 is gone
        self.assertNotIn(3, self.store.keys())
        self.assertEqual(self.store[3], set())


class TestASHBackends(unittest.TestCase):
    """
    Compare two ASH instances seeded with the same hyperedges
    but using different presence backends (dense vs interval).
    """

    def setUp(self):
        # prepare identical hyperedge data: (nodes, start, end)
        self.hyperedges = [
            ([1, 2], 1, 3),
            ([2, 3], 2, 5),
            ([4], 5, 5),
        ]
        # ASH with dense snapshots
        self.ash_dense = ASH(backend="dense")
        # ASH with interval snapshots
        self.ash_interval = ASH(backend="interval")
        # insert same data
        for nodes, start, end in self.hyperedges:
            self.ash_dense.add_hyperedge(nodes, start, end)
            self.ash_interval.add_hyperedge(nodes, start, end)

    def test_snapshot_keys_equal(self):
        # snapshot timestamps should match
        keys_dense = set(self.ash_dense.temporal_snapshots_ids())
        keys_interval = set(self.ash_interval.temporal_snapshots_ids())
        self.assertEqual(keys_dense, keys_interval)

    def test_hyperedge_presence_times_equal(self):
        # for each hyperedge id, compare presence lists
        for hid in self.ash_dense._eid2nids:
            # dense returns list of times
            times_dense = self.ash_dense.hyperedge_presence(hid)
            times_int = self.ash_interval.hyperedge_presence(hid)
            self.assertEqual(sorted(times_dense), sorted(times_int))

    def test_hyperedge_presence_intervals_equal(self):
        # compare merged intervals
        for hid in self.ash_dense._eid2nids:
            iv_dense = self.ash_dense.hyperedge_presence(hid, as_intervals=True)
            iv_int = self.ash_interval.hyperedge_presence(hid, as_intervals=True)
            # sort intervals for comparison
            self.assertCountEqual(iv_dense, iv_int)

    def test_edge_attributes_and_nodes(self):
        # ensure node sets match per hyperedge
        for hid, nodes in self.ash_dense._eid2nids.items():
            self.assertEqual(nodes, self.ash_interval._eid2nids[hid])
            # star sets equal
            self.assertEqual(
                set(self.ash_dense._stars[hid]), set(self.ash_interval._stars[hid])
            )


"""
class TestASHBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate benchmark hyperedges: source node list, start, end
        cls.num_edges = 5000
        cls.time_horizon = 100
        cls.max_span = 50
        cls.hyperedges = []
        for i in range(cls.num_edges):
            length = random.randint(1, cls.max_span)
            start = random.randint(0, cls.time_horizon - length)
            end = start + length
            # random small hyperedge of size 3
            nodes = random.sample(range(1000), 3)
            cls.hyperedges.append((nodes, start, end))

        # Initialize ASH instances
        cls.ash_dense = ASH(backend="dense")
        cls.ash_interval = ASH(backend="interval")

    def test_add_hyperedges(self):
        # Benchmark add_hyperedge
        t0 = time.perf_counter()
        for nodes, start, end in self.hyperedges:
            self.ash_dense.add_hyperedge(nodes, start, end)
        t1 = time.perf_counter()
        dense_time = t1 - t0

        t0 = time.perf_counter()
        for nodes, start, end in self.hyperedges:
            self.ash_interval.add_hyperedge(nodes, start, end)
        t1 = time.perf_counter()
        interval_time = t1 - t0

        print(f"add_hyperedge: dense={dense_time:.4f}s, interval={interval_time:.4f}s")

    def test_hyperedge_presence(self):
        # Ensure hyperedges built first
        # Benchmark hyperedge_presence
        sample_ids = list(self.ash_dense._eid2nids.keys())[:100]

        t0 = time.perf_counter()
        for hid in sample_ids:
            _ = self.ash_dense.hyperedge_presence(hid)
            _ = self.ash_dense.hyperedge_presence(hid, as_intervals=True)
        t1 = time.perf_counter()
        dense_time = t1 - t0

        t0 = time.perf_counter()
        for hid in sample_ids:
            _ = self.ash_interval.hyperedge_presence(hid)
            _ = self.ash_interval.hyperedge_presence(hid, as_intervals=True)
        t1 = time.perf_counter()
        interval_time = t1 - t0

        print(
            f"hyperedge_presence: dense={dense_time:.4f}s, interval={interval_time:.4f}s"
        )

    def test_remove_hyperedges(self):
        # Benchmark removal of half the hyperedges
        ids = list(self.ash_dense._eid2nids.keys())
        to_remove = ids[: len(ids) // 2]

        t0 = time.perf_counter()
        for hid in to_remove:
            self.ash_dense.remove_hyperedge(hid)
        t1 = time.perf_counter()
        dense_time = t1 - t0

        t0 = time.perf_counter()
        for hid in to_remove:
            self.ash_interval.remove_hyperedge(hid)
        t1 = time.perf_counter()
        interval_time = t1 - t0

        print(
            f"remove_hyperedge: dense={dense_time:.4f}s, interval={interval_time:.4f}s"
        )

"""
