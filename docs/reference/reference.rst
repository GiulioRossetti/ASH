Reference
=========

Below is a comprehensive overview of the ``ash_model`` API organized by module.


Classes
~~~~~~~~


This module contains the core classes used in the ASH model. Specifically, it includes the ASH class to represent the model itself, and the NProfile class for node profiles.

.. toctree::
  :maxdepth: 1

  classes




Measures
~~~~~~~~~~~

This section includes various measures that can be computed on ASH models.

.. toctree::
  :maxdepth: 1

  measures_attribute_analysis
  measures_clustering
  measures_hyper_conformity
  measures_hyper_segregation
  measures_s_centralities
  multiego


Generators
~~~~~~~~~~~

This module provides functions to generate synthetic ASH models with specific properties, such as random hypergraphs or hypergraphs with controlled attribute homogeneity.

.. toctree::
  :maxdepth: 1

  generators


Paths and Walks
~~~~~~~~~~~~~~~~

This module includes pathfinding and random walk generation algorithms for ASH models.
Notably, it includes s-walks and time-respecting walks.

.. toctree::
  :maxdepth: 1

  paths_walks
  paths_time_respecting_walks
  paths_randwalks
  

I/O
~~~

This module provides functions to import/export ASH models from/to JSON, CSV, and HIF.

.. toctree::
  :maxdepth: 1

  readwrite

Utils
~~~~~

This module provides various utility functions for working with ASH models, including matrix transformations, projections, networkx conversions, and profile manipulations.

.. toctree::
  :maxdepth: 1

  utils

Visualization
~~~~~~~~~~~~~
.. toctree::
  :maxdepth: 1

  viz

