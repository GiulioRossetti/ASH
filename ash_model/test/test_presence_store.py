import unittest
from ash_model.classes import DensePresenceStore, IntervalPresenceStore


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
        # Order may vary, so use count equal assertion
        self.assertCountEqual(self.store._intervals[1], [(4, 6), (8, 8)])
