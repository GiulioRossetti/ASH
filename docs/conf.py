# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from ash_model import __version__
except ImportError:
    __version__ = "0.2.0"

html_theme = "furo"

version = __version__
# The full version, including alpha/beta/rc tags.
release = version

html_theme_options = {
    "light_logo": "ash.png",
    "dark_logo": "ash.png",
}

# -- Project information -----------------------------------------------------

project = "ASH"
copyright = "2025, Giulio Rossetti"
author = "Giulio Rossetti, Andrea Failla, Salvatore Citraro"

autodoc_mock_imports = [
    "numpy",
    "matplotlib.pyplot",
    "math",
    "matplotlib",
    "networkx",
    "seaborn",
    "scipy",
    "tqdm",
    "seaborn",
    "pandas",
    "csrgraph",
]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # supporto Google/NumPy style docstrings
    "sphinx.ext.intersphinx",  # cross-link con altra doc
]

# Make autosummary scan pages and create stubs automatically
autosummary_generate = True  # Abilitato per generare tabelle riepilogo API
autosummary_generate_overwrite = False
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Mostra i type hints nella signature e non duplicati nella descrizione
autodoc_typehints = "description"
autodoc_typehints_format = "short"

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
}


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# Intersphinx mapping per link esterni
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

# Opzioni per ordina membri (mantiene ordine sorgente)
autodoc_member_order = "bysource"

# Evita warning se mancano __all__
autodoc_inherit_docstrings = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'

# Logo is configured in html_theme_options for Furo theme

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
