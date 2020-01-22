Notes for astropy-helpers contributors
======================================

Note about versions
-------------------

As described in `APE4
<https://github.com/astropy/astropy-APEs/blob/master/APE4.rst>`_, the version
numbers for astropy-helpers follow the corresponding major/minor version of the
`astropy core package <http://www.astropy.org/>`_, but with an independent
sequence of micro (bugfix) version numbers. Hence, the initial release is 0.4,
in parallel with Astropy v0.4, which will be the first version  of Astropy to
use astropy-helpers.

Trying out changes
------------------

If you contribute a change to astropy-helpers and want to try it out with a
package that already uses astropy-helpers, install astropy-helpers from your
branch of the repository in editable mode::

    pip install -e .

Then go to your package and add the ``--use-system-astropy-helpers`` for any
``setup.py`` command you want to check, e.g.::

    python setup.py build_docs --use-system-astropy-helpers

This will cause the installed version to be used instead of any local submodule.
