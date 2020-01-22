About
-----

The **astropy-helpers** package includes
many build, installation, and documentation-related tools used by the Astropy
project, but packaged separately for use by other projects that wish to
leverage this work. The motivation behind this package and details of its
implementation are in the accepted
`Astropy Proposal for Enhancement (APE) 4 <https://github.com/astropy/astropy-APEs/blob/master/APE4.rst>`_.

Astropy-helpers is not a traditional package in the sense that it
is not intended to be installed directly by users or developers. Instead, it
is meant to be accessed when the ``setup.py`` command is run - see :doc:`using`
for how to do this.

For a real-life example of how to implement astropy-helpers in a project,
see the ``setup.py`` and ``setup.cfg`` files of the
`Affiliated package template <https://github.com/astropy/package-template>`_.

.. note:: astropy-helpers v3.x requires Python 3.5 or later. If you wish to
          maintain Python 2 support for your package that uses astropy-helpers,
          then do not upgrade astropy-helpers to v3.0 or later. We will still
          provide Python 2.7 compatible v2.0.x releases until the end of 2019.

User/developer guide
--------------------
.. toctree::
   :maxdepth: 1

   basic.rst
   advanced.rst
   using.rst
   updating.rst
   known_issues.rst
   developers.rst
   api.rst
