Read/Write
==========

Summary
-------
The readwrite module provides functions for reading and writing ASH model data in various formats, including CSV, JSON/JSONL, and HIF (Hypergraph Interchange Format). These functions facilitate the import and export of profiles, social hypergraphs, and ASH models, enabling easy data manipulation and sharing.

.. currentmodule:: ash_model.readwrite.io
 
CSV
~~~

.. autosummary::
   :toctree: _autosummary/readwrite
   :nosignatures:

   write_profiles_to_csv
   read_profiles_from_csv
   write_sh_to_csv
   read_sh_from_csv

JSON / JSONL
~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary/readwrite
   :nosignatures:

   write_profiles_to_jsonl
   read_profiles_from_jsonl
   write_ash_to_json
   read_ash_from_json

HIF
~~~
HIF (Hypergraph Interchange Format) is a file format designed for representing hypergraphs. It is particularly useful for storing and exchanging hypergraph data due to its simplicity and human-readability.

Read more about HIF  `here <https://github.com/pszufe/HIF-standard/tree/main>`_ 


.. autosummary::
   :toctree: _autosummary/readwrite
   :nosignatures:

   write_hif
   read_hif

Details
-------

.. automodule:: ash_model.readwrite.io
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :no-index:
