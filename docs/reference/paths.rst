========
Analysis
========

-----
Paths
-----

.. automodule:: ash_model.paths
    :members:

.. autosummary::
    :toctree: paths/

    temporal_s_dag
    time_respecting_s_walks
    all_time_respecting_s_walks
    annotate_walks
    walk_length
    walk_duration
    walk_weight
    all_simple_paths
    shortest_s_path
    all_shortest_s_path
    all_shortest_s_path_length
    all_shortest_s_walk
    all_shortest_s_walk_length
    shortest_s_walk
    closed_s_walk
    s_distance
    average_s_distance
    has_s_walk
    s_diameter
    s_components
    is_s_path

------------------
Attribute Analysis
------------------


.. automodule:: ash_model.measures
    :members:

.. autosummary::
    :toctree: attribute/

    hyperedge_most_frequent_node_attribute_value
    hyperedge_aggregate_node_profile
    hyperedge_profile_purity
    average_hyperedge_profile_purity
    hyperedge_profile_entropy
    average_hyperedge_profile_entropy
    star_profile_entropy
    average_star_profile_entropy
    star_profile_homogeneity
    average_star_profile_homogeneity
    average_group_degree
    consistency

----------
Clustering
----------

.. autosummary::
    :toctree: clustering/

    s_local_clustering_coefficient
    average_s_local_clustering_coefficient
    s_intersections
    inclusiveness

------------
S-Centrality
------------

.. autosummary::
    :toctree: centrality/

    s_betweenness_centrality
    s_closeness_centrality
    s_eccentricity
    s_harmonic_centrality
    s_katz
    s_load_centrality
    s_eigenvector_centrality
    s_information_centrality
    s_second_order_centrality

----------------
Hyper-Conformity
----------------

.. autosummary::
    :toctree: conformity/

    hyper_conformity
