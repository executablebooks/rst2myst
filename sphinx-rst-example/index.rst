.. RST Sphinx Example documentation master file, created by
   sphinx-quickstart on Fri Jun 19 11:25:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RST Sphinx Example's documentation!
==============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   about_py

This is the index page

.. code-block:: python3
   :linenos:
   :lineno-start: 2
   :emphasize-lines: 3,5
   :caption: This is Caption Text
   :name: this-py
   :dedent: 4
   :force:

   import pandas as pd
   df1 = pd.DataFrame()
   # Comment here
   df2 = pd.DataFrame()
   df1.merge(df2)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
