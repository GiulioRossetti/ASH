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

import sphinx_rtd_theme

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from ash_model import __version__
except ImportError:
    __version__ = u'0.1.0'

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

version = __version__
# The full version, including alpha/beta/rc tags.
release = version

html_theme_options = {
    "collapse_navigation": False,
    "display_version": False,
    "navigation_depth": 3,
}

# -- Project information -----------------------------------------------------

project = "ASH"
copyright = "2024, Giulio Rossetti"
author = "Giulio Rossetti, Andrea Failla, Salvatore Citraro"

autodoc_mock_imports = [
     "numpy",
    "halp.undirected_hypergraph",
    "halp.utilities.undirected_matrices",
    "matplotlib.pyplot",
    "ash_model"
    "math",
    "halp",
    "matplotlib",
    "networkx",
    "seaborn",
    "scipy",
    "tqdm",
    "seaborn",
    "pandas",
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
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'

html_logo = 'ash.png'

# The name of an image file (relative to this directory) to use as a favicon of
# the docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
