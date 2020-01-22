Known issues
============

If you are building a package with a C extension on old MacOS X systems (e.g.
MacOS X 10.7 Lion) you may run into issues (e.g. segmentation fault) with the
default GCC 4.2 compiler available on those systems. If this is the case, you
can tell setuptools to use the clang compiler (which should work) using e.g.::

    CC=clang python setup.py build

See `astropy/astropy#31 <https://github.com/astropy/astropy/issues/31>`_ for a
discussion of the original problem.
