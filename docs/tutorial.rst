***************
Tutorial
***************

This section provides interactive tutorials to help you get started with ASH (Attributed Stream Hypergraph).
Each tutorial is a Jupyter notebook that demonstrates different aspects of the library.

All tutorial notebooks are available in the `tutorial/ <https://github.com/GiulioRossetti/ASH/tree/main/tutorial>`_ folder of the repository.

Tutorial notebooks (links on GitHub)
-----------------------------------

The notebooks are kept in the repository under the `tutorial/` folder and are intended to be opened on GitHub or in your local Jupyter environment. The documentation build no longer renders notebooks into the HTML site; instead use the links below to view or download the notebooks from GitHub:

- `00-basics.ipynb <https://github.com/GiulioRossetti/ASH/blob/main/tutorial/00-basics.ipynb>`_
- `01-attribute_analysis.ipynb <https://github.com/GiulioRossetti/ASH/blob/main/tutorial/01-attribute_analysis.ipynb>`_
- `02-walks.ipynb <https://github.com/GiulioRossetti/ASH/blob/main/tutorial/02-walks.ipynb>`_
- `03-generators.ipynb <https://github.com/GiulioRossetti/ASH/blob/main/tutorial/03-generators.ipynb>`_
- `04-io.ipynb <https://github.com/GiulioRossetti/ASH/blob/main/tutorial/04-io.ipynb>`_
- `05-multiego.ipynb <https://github.com/GiulioRossetti/ASH/blob/main/tutorial/05-multiego.ipynb>`_
- `06-segregation.ipynb <https://github.com/GiulioRossetti/ASH/blob/main/tutorial/06-segregation.ipynb>`_

Getting Started
---------------

The tutorials are organized in a progressive manner:

1. **Basics** (``00-basics.ipynb``): Learn how to create an ASH object, add nodes and hyperedges, and access basic properties and measures.

2. **Attribute Analysis** (``01-attribute_analysis.ipynb``): Explore purity/entropy measures on hyperedge profiles, homogeneity, and temporal consistency of attributes.

3. **Walks** (``02-walks.ipynb``): Understand s-walks, distances, and s-based components in the temporal context.

4. **Generators** (``03-generators.ipynb``): Generate random hypergraphs and transform external structures into ASH objects.

5. **I/O Operations** (``04-io.ipynb``): Read and write ASH structures in various formats including CSV profiles, JSON, and HIF format.

6. **Multi-Ego Networks** (``05-multiego.ipynb``): Work with multi-ego network extraction and analysis.

7. **Segregation Analysis** (``06-segregation.ipynb``): Measure and analyze segregation patterns in attributed hypergraphs.

Running the Notebooks
---------------------

To run the tutorial notebooks locally:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/GiulioRossetti/ASH.git
      cd ASH

2. Install the required dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

3. Launch Jupyter:

   .. code-block:: bash

      jupyter notebook tutorial/

   Or open the notebooks directly in VS Code with the Jupyter extension.
