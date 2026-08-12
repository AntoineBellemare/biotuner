"""The single source of truth for biotuner's version.

Everything else derives from this file:

- ``pyproject.toml`` reads it at build time through
  ``[tool.setuptools.dynamic] version = {attr = "biotuner._version.__version__"}``
- ``biotuner/__init__.py`` re-exports it as ``biotuner.__version__``
- ``docs/conf.py`` parses it out of this file directly

It deliberately contains nothing but the assignment, and imports nothing. That
is what lets setuptools read the value by parsing the syntax tree instead of
importing the package -- an import would pull in numpy, pandas, mne and the
rest, which are not guaranteed to be present in a build environment.

To cut a release, change the number here and nowhere else, then tag. Keep the
assignment a plain string literal on one line: anything cleverer (an f-string,
a computed value, a tuple joined together) defeats the static read and sends
setuptools back to importing.
"""

__version__ = "0.5.0"
