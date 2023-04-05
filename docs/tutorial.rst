***************
Tutorial
***************

In this tutorial we will introduce the ``ASH`` object that can be used to describe undirected, temporal hypergraphs with attributes on nodes.

Creating an ASH
----------------

Create an empty ASH with no nodes and no edges.

.. code:: python

	from ash import ASH
	h = ASH(hedge_removal=True)

During the construction phase the ``hedge_removal`` parameter allows to specify if the structure will allow hyperedge removal or not.


High-order Interactions: Hyperedges
--------------------

``h`` can be grown by adding one interaction (i.e., an hyperedge) at a time.
Each interaction is univocally defined by an arbitrarily-large set of nodes, and has a ``start`` and a ``end`` timestamp, indicating its appearance and disappearance times respectively.

.. code:: python

	h.add_hyperedge({1,2,3,4}, start=0, end=1)


If nodes are not already present in the ASH, they will be automatically added when they appear in an interaction.
In the above example the interaction ``{1,2,3,4}`` appears at time ``0`` and vanishes at time ``1``, thus being present in both timestamps.
If the ``end`` parameter is omitted, the hyperedge will only be active in ``start``.
Moreover, hyperedges are associated with a unique ``str`` identifier of the form "en" where n is a positive number:

.. code:: python

   h.get_hyperedge_id_set()

   >>> {"e1"}


Multiple interactions can be added at the same time: in such scenario all the interactions in the list will have the same appearance and disappearance times.

.. code:: python

	h.add_hyperedges([{1,2,3}, {2, 3}, {3, 1, 4}], start=1, end=2)

The same method can be used to add any ``ebunch`` of interaction.  An *ebunch* is any iterable container of interactions.

.. code:: python

   import networkx as nx
   g = nx.barabasi_albert_graph(n=500, m=3)
   h.add_hyperedges(nx.find_cliques(g), start=0)

One of the factors that make ``ash`` faster with respect to other hypergraph libraries is the ``O(1)`` access it provides to a node's star.
That is to say that given a node is is possible to easily get the set of hyperedges it is contained in. It is also possible to conveniently restrict the result to hyperedges of a given size and/or active at a given timestamp:

.. code:: python

   h.get_star(1) # set of ids of the hyperedges that contain node 1
   h.get_star(1, hyperedge_size=4, tid=2) # set of ids of the hyperedges that contain node 1,
                                          # have size 4
                                          # and are active at time 2



Node Profiles: Enrich Your Hypergraph!
--------------------

As for hyperedges, it is also possible to add one node at a time, specifying ``start``, or ``start`` and ``end`` timestamps.
The same can be done for multiple nodes at a time, which will all have the same ``start`` and ``end`` timestamps:

.. code:: python

   # one node at a time
   h.add_node(0, start=0)
   h.add_node(1, start=0, end=1)

   # multiple nodes, same time window
   h.add_nodes([1,2,3], start=0)
   h.add_nodes([4,5,6], start=0, end=1)


To incorporate the semantics on nodes,  ``ash`` leverages the  ``NProfile`` class.
This structure can be used to contain the attributes of a node:

.. code:: python

   from ash import NProfile
   profile = NProfile(node_id=1, name='Alice', party='L') # add attributes at creation
   profile.add_attribute('age', 24) # add attribute with dedicated method

Node profiles can be incorporated in the ``ASH`` when adding a new node. This allows to model change in the node attributes by adding the same node at different times with different profiles:

.. code:: python

   profile = NProfile(node_id=1, name='Alice', party='L', age=24)
   h.add_node(1, start=0, attr_dict=profile)
   profile2 = NProfile(node_id=1, name='Alice', party='R', age=25)
   h.add_node(1, start=1, attr_dict=profile2)

To see all information on a node profile through time, you can use the following:

.. code:: python

   profile = h.get_node_profile(1)
   attrs = profile.get_attributes()
   print(attrs)

   >>> {'t': [[0, 1]], 'name': {0: 'Alice', 1: 'Alice'}, 'party': {0: 'L', 1: 'R'}, 'age': {0: 24, 1: 25}}


Time Respecting s-Walks
--------------------

ASH integrates the s-analysis framework to generalize classic graph measures to hypergraphs, and extends it with temporal information.
In short, walks are extended to hypergraphs based on a parameter ``s``, which controls the minimum number of common nodes between subsequent hyperedges.
For instance:

.. code:: python

   from ash import ASH
   from ash import paths

   h = ASH()
   h.add_hyperedge([1,2,3], start=0) # e1
   h.add_hyperedge([2,3,4], start=0) # e2
   h.add_hyperedge([4,5,6], start=0) # e3
   h.add_hyperedge([1,2,3,4,5, 6], start=0) # e4
   s = 2

   paths.has_s_walk(h, s, "e1", "e3")

   >>> True

   paths.shortest_s_walk(h, s, "e1", "e3")

   >>> ['e1', 'e4', 'e3']

   paths.average_s_distance(h, s=2)

   >>> 1.6666666666666667

For more info on the s-analysis framework please refer to the original paper. The novelty with ``ASH`` is the extension to the dynamic scenario.
This means that ``ASH`` is able to compute time-respecting s-walks.

.. code:: python

   h = ASH()
   h.add_hyperedge([1, 2, 3], 0, 4)
   h.add_hyperedge([1, 4], 0, 1)
   h.add_hyperedge([1, 2, 3, 4], 2, 3)
   trsw = paths.time_respecting_s_walks(h, s=1, hyperedge_from='e1', hyperedge_to='e3')

   for _, ap in trsw.items():
      walks = paths.annotate_walks(ap)
      print('Shortest:', walks['shortest'])
      print('Foremost', walks['foremost'])

In the above example, first we get all time respecting s-walks via ``time_respecting_s_walks()``; then we annotate each path and print the fastest and the foremost ones.
Other choices include fastest, heaviest, and all possible combinations.

High-order Mixing Patterns
--------------------


Navigating in Time
-------------------------

The timestamps associated to the hyperedges can be retrieved through:

.. code:: python

	h.temporal_snapshots_ids()


Several methods return statistics about the flattened (i.e., static, aggregated) hypergraph.
To get the same statistic considering a specific timestamp, you can use the ``tid`` parameter

.. code:: python

   h.get_number_of_nodes() # count nodes in the hypergraph
   h.get_number_of_nodes(tid=3) # count nodes active at time 3


Once loaded, it is possible to extract a time slice from an ``ASH``, i.e., a time-span hypergraph.

.. code:: python

	s = h.hypergraph_temporal_slice(start=2, end=3)

the resulting ``ASH`` stored in ``s`` will be composed by nodes and interactions existing within the time span ``[2, 3]``.


A dynamic hypernetwork can be also described as stream of interactions, a chronologically ordered list of interactions

.. code:: python

   for e in h.stream_interactions():
      print(e)

the ``stream_interactions`` method returns a generator that streams the interactions in ``h``, where ``e`` is a 3-tuple ``(tid, he_id, op)``

 - ``tid`` is the interactions timestamp
 - ``he_id`` is the id associated with the corresponding hyperedge
 - ``op`` is a hyperedge creation or deletion event (respectively ``+``, ``-``)

Visual Analytics
--------------------
``ash`` also provides a module that facilitates visualizing both static and temporal statistics.

.. code:: python

   from ash import viz
   from matplotlib import pyplot as plt
   ax = viz.plot_hyperedge_size_distribution(h)
   ax.set_title('Hyperedge size distribution')
   plt.show()

All functions here are a wrapper around either  ``matplotlib`` or ``seaborn`` commands, and return the ``plt.Axes`` object associated with the plot.
You can also pass a ``**kwargs`` dictionary with ``matplotlib`` plotting parameters (e.g., ``marker``).
Since all functions return the ``plt.Axes`` object, you can customize it to your likings like the above.



To facilitate visualizing time series data emerging from the ``ASH``, the library provides two methods, one for the structural features,
one for the attribute-related dynamics. Let's start with the former:

.. code:: python

   from ash import measures
   func = measures.average_s_local_clustering_coefficient # function to be applied to all snapshots
   func_params = {'s': 2} # parameters of the function above, if any
   rolling = 3 # optional rolling window size for smoothing the time series
   ax = viz.plot_structure_dynamics(h, func, func_params, rolling)
   ax.set_title('Number of hyperedges in time')
   plt.show()

A couple of things to note here:
 - ``func`` is a ``Callable`` object that is applied to all snapshots.
 - ``func_params`` allows to pass arguments to ``func``. If the only parameters of a function are ``h`` and/or ``tid``, or ``start``, and ``end``, this parameter can be ignored/set to ``None``.
 - ``rolling`` is an optional parameter that is used to smooth the time series. In case you have high-resolution temporal data, it is best to set this parameter; it basically computes the mean over ``rolling`` timestamps.
Instead, for attribute dynamics we have – you guessed it: ``plot_attribute_dynamics()``

.. code:: python

   attr_name = "party" # compute measure for this attribute
   func = measures.average_hyperedge_profile_purity # function to be applied to all snapshots
   func_params = {'by_label': True} # optional parameters of the function above
   rolling = 3 # optional rolling window size for smoothing the time series

   ax = viz.plot_attribute_dynamics(h, attr_name, func, func_params, rolling)
   ax.set_title('Average hyperedge purity (by label)')
   plt.show()

The main differences here is that we have to set the attribute name for which we want to compute the statistics.

In conclusion, if you want to compare different dynamic hypernetworks, you can do something like:

.. code:: python

   hs = [h0, h1, h2] # three different ASH instances

   attr_name = "party" # compute measure for this attribute
   func = measures.average_hyperedge_profile_purity # function to be applied to all snapshots
   func_params = {'by_label': True} # optional parameters of the function above
   rolling = 3 # optional rolling window size for smoothing the time series

   fig, axs = plt.subplots(nrows=1, ncols=len(hs), figsize=(18, 6), sharey=True, sharex=True)
   plt.subplots_adjust(hspace=0.8)

   for h, ax in zip(hs, axs.ravel()):
      viz.plot_attribute_dynamics(h, attr_name, func, func_params, rolling, ax=ax)
      ax.grid(alpha=.2)

   plt.suptitle('Average hyperedge purity (by label)', fontsize=32)
   plt.show()


Reading and Writing
--------------------

``ash`` allows to read/write hypernetworks in two ways, either:

 - from/to separate files for interactions and node profiles
 - from/to a single file


The former format allows to read/write interactions and node profiles separately. For the hyperedges:


.. code:: python

	from ash import readwrite as io
    io.write_sh_to_csv(h, filename)
    h = io.read_sh_from_csv(filename)

where each row is a tab-separated list of the form ``[nodes -> start, end]``, with ``nodes`` being the members of each interactions, and ``start, end`` the appearance-disappearance times of the interactions.
Likewise, for the profiles:

.. code:: python

    io.write_profiles_to_csv(h, filename)
    profiles = io.read_profiles_from_csv(filename)

where each row is a comma-separated list of the form ``[node_id, tid, attr1, attr2...]``. Nodes that appear multiple times are stored as a separate row for each time they are active.
Profiles can also be stored as a jsonl file with similar structure:

.. code:: python

    io.write_profiles_to_jsonl(h, filename)
    profiles = io.read_profiles_from_jsonl(filename)



The latter format encloses the whole ``ASH`` in a ``json`` file

.. code:: python

    io.write_ash_to_json(h, filename)
    h = io.read_ash_from_json(filename)





