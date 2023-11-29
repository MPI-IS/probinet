# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import sys
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
from pathlib import Path
from typing import List, Any

root_folder = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(root_folder))
# -- Project information -----------------------------------------------------

project = 'pgm'
copyright = (
    '2022, Max Planck Society / '
    'Software Workshop - Max Planck Institute for Intelligent Systems')
author = 'Diego Baptista Theuerkauf'

from pgm.version import __version__  # noqa: E402 # pylint: disable=wrong-import-position

version = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',  # pull doc from docstrings
    # 'sphinx_autodoc_typehints',  # type hints # TODO: remove this from reqs
    'sphinx.ext.intersphinx',  # link to other projects
    'sphinx.ext.todo',  # support TODOs
    'sphinx.ext.ifconfig',  # include stuff based on configuration
    'sphinx.ext.viewcode',  # add source code
    #'myst_parser',  # add MD files # not needed if myst_nb is used!
    'sphinx.ext.autosummary',  # pull doc from docstrings
    # 'nbsphinx',  # add Jupyter notebooks
    'myst_nb',  # add Jupyter notebooks
    'sphinxcontrib.napoleon'  # Google style doc
]
autodoc_default_options = {
    'members': True,  # when set to True, Sphinx will automatically document all members
    # with docstrings.
    'undoc-members': True,
    # members without docstrings will still be included in the documentation,
    # but without detailed documentation.
}
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: List[Any] = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'piccolo_theme'  # 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
