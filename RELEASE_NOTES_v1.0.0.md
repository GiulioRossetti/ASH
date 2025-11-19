# Release v1.0.0 - Codename: Bulbasaur

## Overview
This is the first major release of the ASH (Attributed Stream Hypergraph) library, version 1.0.0, codenamed "Bulbasaur".

## Version Information
- **Version**: 1.0.0
- **Codename**: Bulbasaur
- **Release Date**: November 2025

## What's Included
This release includes the core functionality of the ASH library:
- Attributed Stream Hypergraph data structures
- Node profile management
- Presence store implementations (Dense, Interval)
- Hypergraph generators (random, homophily-driven)
- Measures and metrics (clustering, centralities, segregation, conformity)
- Visualization tools (static and temporal)
- Path analysis (walks, random walks, time-respecting walks)
- MultiEgo network analysis
- I/O utilities for reading and writing hypergraphs

## Installation
```bash
pip install ash_model
```

or via conda:
```bash
conda config --add channels giuliorossetti
conda config --add channels conda-forge
conda install ash_model
```

## Citation
If you use ASH in your research, please cite:

> Failla, A., Citraro, S. & Rossetti, G. _Attributed Stream Hypergraphs: temporal modeling of node-attributed high-order interactions_. Appl Netw Sci 8, 31 (2023). https://doi.org/10.1007/s41109-023-00555-6

## Documentation
Full documentation is available at: https://ash-model.readthedocs.io/

## GitHub Repository
https://github.com/GiulioRossetti/ASH
