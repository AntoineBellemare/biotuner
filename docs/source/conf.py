# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'biotuner'
copyright = '2023, Antoine Bellemare'
author = 'Antoine Bellemare'
release = '2023-02-28'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'nbsphinx']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
#html_static_path = ['_static']
#html_theme = 'sphinx_rtd_theme'
html_theme = 'sphinx_book_theme'
#html_theme = 'pydata_sphinx_theme'

# Add autodoc options
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'private-members': False,
    'special-members': False,
    'show-inheritance': True,
    'member-order': 'bysource',
    'exclude-members': '__weakref__, __module__',
}

# Add Napoleon extension
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']

#sphinx-build -b html docs/source/ _build/html/