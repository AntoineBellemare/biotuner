# -- REQUIREMENTS -----------------------------------------------------
# pip install sphinx-material
# pip install sphinxemoji

import datetime
import os
import re
import sys
import json
import urllib.request

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, os.path.abspath("../"))  # To include your biotuner folder


# sys.path.insert(0, os.path.abspath("../"))  # To include your biotuner folder


# -- Project information -----------------------------------------------------
def find_author():
    """This returns 'The Biotuner's development team'"""
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format("__author__"),
        open("../biotuner/__init__.py").read(),
    )
    return str(result.group(1))


project = "Biotuner"
copyright = f"2023–{datetime.datetime.now().year}"
author = 'Antoine Bellemare & François Lespinasse. This documentation and code are licensed under the <a href="https://opensource.org/licenses/MIT">MIT License</a>.'


def get_latest_version_from_pypi(package_name):
    try:
        url = f"https://pypi.org/pypi/{package_name}/json"
        with urllib.request.urlopen(url) as response:
            data = json.load(response)
            return data["info"]["version"]
    except Exception as e:
        print(f"Error fetching version from PyPI: {e}")
        return "unknown"


# Replace 'biotuner' with your package's name as registered on PyPI
version = "0.0.16"
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx_nbexamples",
    "jupyter_sphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
    "sphinxemoji.sphinxemoji",
    "sphinx_copybutton",
    "myst_nb",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

# Ignore duplicated sections warning
suppress_warnings = ["epub.duplicated_toc_entry"]
nitpicky = False  # Set to True to get all warnings about crosslinks

# Prefix document path to section labels, to use:
# `path/to/file:heading` instead of just `heading`
autosectionlabel_prefix_document = True

# -- Options for autodoc -------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_ivar = False
napoleon_use_rtype = False
add_module_names = False  #  If true, the current module name will be prepended to all description

templates_path = ["_templates"]
exclude_patterns = []

# nbsphinx_execute = 'examples/**/*.ipynb'
# nbsphinx_execute_arguments = [
#     "--ExecutePreprocessor.timeout=600",
#     "--ExecutePreprocessor.kernel_name=biotuner",
#     "--ExecutePreprocessor.allow_errors=True",
#     "--ExecutePreprocessor.interrupt_on_timeout=True",
# ]

# -- Options for myst_nb ---------------------------------------------------
nb_execution_mode = "off"
nb_execution_raise_on_error = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_static_path = ['_static']
# html_theme = 'sphinx_rtd_theme'
html_theme = "sphinx_book_theme"
# html_theme = 'pydata_sphinx_theme'
# -- Options for HTML output -------------------------------------------------

html_favicon = "img/favicon.ico"
html_logo = "img/logo.png"
autoclass_content = "both"


# Add autodoc options
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "special-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    "exclude-members": "__weakref__, __module__",
}

latex_elements = {
    "preamble": r"""
    \usepackage{tocloft}
    \renewcommand{\cftchapleader}{\cftdotfill{\cftdotsep}}
    \renewcommand{\cfttoctitlefont}{\hfill\Large\bfseries}
    \renewcommand{\cftaftertoctitle}{\hfill}
    """,
}
html_static_path = []


# git push origin --delete docs_build
# git branch -D docs_build
# git checkout -b docs_build
# sphinx-build -b html docs docs/_build/html
# git add docs/_build/html
# git commit -m "Update documentation"
# git push --set-upstream origin docs_build
