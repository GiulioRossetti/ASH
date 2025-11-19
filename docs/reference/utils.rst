Utils
=====

Summary
-------
The utils module provides various utility functions for working with ASH models, including matrix transformations, projections, and profile manipulations.

Matrices
~~~~~~~~~~~~
.. currentmodule:: ash_model.utils.matrices
.. autosummary::
   :toctree: _autosummary/utils
   :nosignatures:

   get_hyperedge_id_mapping
   get_node_id_mapping
   incidence_matrix
   incidence_matrix_by_time
   adjacency_matrix
   adjacency_matrix_by_time

Projections
~~~~~~~~~~~~
.. currentmodule:: ash_model.utils.projections
.. autosummary::
   :toctree: _autosummary/utils
   :nosignatures:

   clique_projection
   bipartite_projection
   line_graph_projection
   dual_hypergraph_projection
   clique_projection_by_time
   bipartite_projection_by_time
   line_graph_projection_by_time
   dual_hypergraph_projection_by_time

Networkx Utils
~~~~~~~~~~~~~~
.. currentmodule:: ash_model.utils.networkx
.. autosummary::
   :toctree: _autosummary/utils
   :nosignatures:

   from_networkx_graph
   from_networkx_graph_list
   from_networkx_maximal_cliques
   from_networkx_maximal_cliques_list
   from_networkx_bipartite
   from_networkx_bipartite_list


Profile Utils
~~~~~~~~~~~~~
.. currentmodule:: ash_model.utils.profiles
.. autosummary::
   :toctree: _autosummary/utils
   :nosignatures:

   aggregate_node_profile
   hyperedge_most_frequent_node_attribute_value
   hyperedge_aggregate_node_profile



Details
-------

.. automodule:: ash_model.utils.matrices
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :no-index:

.. automodule:: ash_model.utils.networkx
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :no-index:

.. automodule:: ash_model.utils.profiles
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :no-index:

.. automodule:: ash_model.utils.projections
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :no-index:
