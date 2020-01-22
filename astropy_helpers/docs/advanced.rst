Advanced functionality
======================

OpenMP helpers
--------------

We provide a helper function
:func:`~astropy_helpers.openmp_helpers.add_openmp_flags_if_available` that can
be used to automatically add OpenMP flags for C/Cython extensions, based on
whether OpenMP is available and produces executable code. To use this, edit the
``setup_package.py`` file where you define a C extension, import the helper
function::

    from astropy_helpers.openmp_helpers import add_openmp_flags_if_available

then once you have defined the extension and before returning it, use it as::

    extension = Extension(...)

    add_openmp_flags_if_available(extension)

    return [extension]
