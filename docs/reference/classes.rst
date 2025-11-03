Classes
=======

Overview of the core package classes.

Summary
-------

.. currentmodule:: ash_model

.. autosummary::
   :toctree: _autosummary/classes
   :nosignatures:

   ASH
   NProfile
   PresenceStore
   DensePresenceStore
   IntervalPresenceStore

Details
-------

.. note::

   The ASH class acts as the core data structure for representing temporal hypergraphs. It encapsulates nodes, hyperedges, attributes and their temporal dynamics, providing methods for manipulation and analysis.


.. automodule:: ash_model.classes.undirected
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :no-index:


.. note::

   The NProfile class is designed to manage all attributes related to a node within the ASH framework. It supports various data types and provides functionality for adding, updating, and querying node attributes over time.

.. automodule:: ash_model.classes.node_profile
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :no-index:


.. note::
   The PresenceStore class and its subclasses (DensePresenceStore, IntervalPresenceStore) are specialized data structures for efficiently storing and querying the temporal presence of hyperedges within the ASH model. They support different storage strategies to optimize for various use cases and performance requirements.

Backend Selection
~~~~~~~~~~~~~~~~~

ASH supports two backend implementations for managing temporal presence:

**Dense Backend** (default)
   - **Storage**: Direct mapping ``time → set(hyperedge_ids)``
   - **Best for**: Dense timelines, frequent snapshot queries, shorter time spans
   - **Complexity**: O(1) snapshot access, O(1) per-time updates
   - **Memory**: Proportional to number of distinct time points

**Interval Backend**
   - **Storage**: Sparse representation ``hyperedge_id → list[(start, end)]`` with optimized event-diff indexing
   - **Best for**: Long sparse intervals, bulk interval operations, memory-constrained environments
   - **Complexity**: O(log k) snapshot materialization via bisect (k = intervals per id), O(1) interval add/remove
   - **Memory**: Proportional to number of intervals (typically much smaller for sparse data)

Usage example::

    # Default dense backend
    H = ASH()

    # Interval backend for sparse temporal data
    H_sparse = ASH(backend="interval")

    # Both backends provide identical API
    H.add_hyperedge([1, 2, 3], start=0, end=100)
    H_sparse.add_hyperedge([1, 2, 3], start=0, end=100)

Technical Note
~~~~~~~~~~~~~~

The **IntervalPresenceStore** implements two key optimizations:

1. **Binary search for presence queries**: Maintains a parallel ``starts`` list for each hyperedge's intervals, enabling O(log k) lookup instead of O(k) linear scan.

2. **Event-diff for interval updates**: Uses a difference array (``_time_events``) to represent interval boundaries. Adding/removing an interval requires only O(1) event updates instead of O(length) per-time operations. The full time→count mapping is rebuilt lazily when ``keys()`` is called.

These optimizations make the interval backend practical for large-scale temporal networks with long, sparse activity patterns.

.. automodule:: ash_model.classes.presence_store
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :no-index:

