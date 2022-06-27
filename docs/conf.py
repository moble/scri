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

import sphinx

from recommonmark.parser import CommonMarkParser

# -- Project information -----------------------------------------------------

from scri import __version__ as scri_version

# -- Project information -----------------------------------------------------

project = "scri"
copyright = "2016, Michael Boyle"
author = "Michael Boyle"

# The short X.Y version
version = scri_version
# The full version, including alpha/beta/rc tags
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "recommonmark",
]

autosummary_generate = True

autodoc_docstring_signature = True
if sphinx.version_info < (1, 8):
    autodoc_default_flags = ["members", "undoc-members"]
else:
    autodoc_default_options = {"members": None, "undoc-members": None, "special-members": "__call__"}

# -- Try to auto-generate numba-decorated signatures -----------------

import numba
import inspect


def process_numba_docstring(app, what, name, obj, options, signature, return_annotation):
    if type(obj) is not numba.core.registry.CPUDispatcher:
        return (signature, return_annotation)
    else:
        original = obj.py_func
        orig_sig = inspect.signature(original)

        if (orig_sig.return_annotation) is inspect._empty:
            ret_ann = None
        else:
            ret_ann = orig_sig.return_annotation.__name__

        return (str(orig_sig), ret_ann)


def setup(app):
    app.connect("autodoc-process-signature", process_numba_docstring)


# --------------------------------------------------------------------

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
if sphinx.version_info < (1, 8):
    source_parsers = {
        ".md": CommonMarkParser,
    }
    source_suffix = [".rst", ".md"]
else:
    source_suffix = {
        ".rst": "restructuredtext",
        ".txt": "markdown",
        ".md": "markdown",
    }

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
