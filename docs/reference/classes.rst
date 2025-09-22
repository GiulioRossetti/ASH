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
   The PresenceStore class and its subclasses (DensePresenceStore, IntervalPresenceStore) are specialized data structures for efficiently storing and querying the presence intervals of nodes within the ASH model. They support different storage strategies to optimize for various use cases and performance requirements.

.. automodule:: ash_model.classes.presence_store
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :no-index:

