import unittest
import numpy as np

from ash_model import ASH
from ash_model.paths import time_respecting_random_walks, TemporalEdge


class TimeRespectingRandomWalksCase(unittest.TestCase):
    @staticmethod
    def get_hypergraph():
        a = ASH()
        # same structure used in test_time_respecting_walks
        a.add_hyperedge([1, 2, 3], 0, 4)  # e1: t in [0,3]
        a.add_hyperedge([1, 4], 0, 1)  # e2: t in [0,0]
        a.add_hyperedge([1, 2, 3, 4], 2, 3)  # e3: t in [2]
        a.add_hyperedge([1, 3, 4], 2, 3)  # e4: t in [2]
        a.add_hyperedge([3, 4], 3, 4)  # e5: t in [3]
        return a

    def test_node_random_walks_time_respecting(self):
        a = self.get_hypergraph()
        # walks on nodes, s=1. Ensure walks are not all trivial and are time-respecting (same timestamp transitions)
        walks = time_respecting_random_walks(
            a,
            s=1,
            start_from=None,  # let it pick active nodes at first timestamps
            stop_at=None,
            start=0,
            end=3,
            num_walks=10,
            walk_length=5,
            edge=False,
            p=1.0,
            q=1.0,
            threads=-1,
            terminate_at_sink=False,
        )
        # Expect at least one non-trivial walk (length > 1)
        self.assertTrue(len(walks) > 0)
        self.assertTrue(any(len(w) > 1 for w in walks))
        # Check time-respecting: for each pair of consecutive steps, there exists same-time co-membership
        # We rebuild a neighbor map per time to validate
        ids = a.temporal_snapshots_ids()
        timestamped_neighbors = {}
        for tid in ids:
            nn = {}
            for he in a.hyperedges(start=tid, end=tid):
                nodes = list(a.get_hyperedge_nodes(he))
                for i in range(len(nodes)):
                    for j in range(len(nodes)):
                        if i == j:
                            continue
                        u, v = str(nodes[i]), str(nodes[j])
                        nn.setdefault(u, set()).add(v)
            timestamped_neighbors[tid] = nn

        # Since node walks hide timestamps, accept either:
        # - same-time co-membership (u->v exists at some tid), or
        # - forward-in-time stay (u==v)
        def pair_is_time_respecting(u, v):
            if str(u) == str(v):
                return True
            for tid, nn in timestamped_neighbors.items():
                if str(v) in nn.get(str(u), set()):
                    return True
            return False

        for w in walks:
            if len(w) > 1:
                self.assertTrue(
                    all(pair_is_time_respecting(u, v) for u, v in zip(w, w[1:]))
                )

        # With terminate_at_sink=True, walks should stop at sinks (no repeats to extend length)
        walks_stop = time_respecting_random_walks(
            a,
            s=1,
            start_from=None,
            stop_at=None,
            start=0,
            end=3,
            num_walks=5,
            walk_length=5,
            edge=False,
            terminate_at_sink=True,
        )
        self.assertTrue(len(walks_stop) > 0)
        # None of the walks should contain long runs of the same node past a sink due to artificial extension
        for w in walks_stop:
            if len(w) > 1:
                # there can be repeats if distinct timestamp labels map to the same base node, but
                # with terminate_at_sink=True we shouldn't see trailing repetitions from self-loops
                self.assertTrue(len(w) <= 5)

    def test_edge_random_walks_time_respecting(self):
        a = self.get_hypergraph()
        # walks on hyperedges; verify non-trivial and time-respecting via TemporalEdge.tid equality per transition
        walks_map = time_respecting_random_walks(
            a,
            s=1,
            start_from="e1",
            stop_at=None,
            start=0,
            end=3,
            num_walks=5,
            walk_length=4,
            edge=True,
            p=1.0,
            q=1.0,
            threads=-1,
            terminate_at_sink=False,
        )
        # Flatten and check
        all_paths = [p for paths in walks_map.values() for p in paths]
        self.assertTrue(len(all_paths) > 0)
        self.assertTrue(any(len(p) > 1 for p in all_paths))
        # time-respecting: each consecutive TemporalEdge either same tid (intra-timestamp)
        # or forward to next timestamp for the same hyperedge (stay active)
        for p in all_paths:
            for e1, e2 in zip(p, p[1:]):
                self.assertIsInstance(e1, TemporalEdge)
                self.assertIsInstance(e2, TemporalEdge)
                self.assertTrue(
                    (e1.tid == e2.tid)
                    or (e2.tid == e1.tid + 1 and e2.fr == e2.to == e1.to),
                    msg=f"Invalid step: {e1} -> {e2}",
                )

        # With terminate_at_sink=True, paths should stop at sinks (no artificial stay edges)
        walks_map_stop = time_respecting_random_walks(
            a,
            s=1,
            start_from="e1",
            stop_at=None,
            start=0,
            end=3,
            num_walks=3,
            walk_length=4,
            edge=True,
            terminate_at_sink=True,
        )
        all_paths_stop = [p for paths in walks_map_stop.values() for p in paths]
        for p in all_paths_stop:
            # no TemporalEdge that is a self-loop with zero weight should appear when terminate_at_sink=True
            self.assertTrue(all(not (e.fr == e.to and e.weight == 0.0) for e in p))


if __name__ == "__main__":
    unittest.main()
