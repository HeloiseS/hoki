"""
Different implementations of the ``./setup.py test`` command depending on
what's locally available.

If Astropy v1.1 or later is available it should be possible to import
AstropyTest from ``astropy.tests.command``. Otherwise there is a skeleton
implementation that allows users to at least discover the ``./setup.py test``
command and learn that they need Astropy to run it.
"""

import os
from ..utils import import_file

# Previously these except statements caught only ImportErrors, but there are
# some other obscure exceptional conditions that can occur when importing
# astropy.tests (at least on older versions) that can cause these imports to
# fail

try:

    # If we are testing astropy itself, we need to use import_file to avoid
    # actually importing astropy (just the file we need).
    command_file = os.path.join('astropy', 'tests', 'command.py')
    if os.path.exists(command_file):
        AstropyTest = import_file(command_file, 'astropy_tests_command').AstropyTest
    else:
        import astropy  # noqa
        from astropy.tests.command import AstropyTest

except Exception:

    # No astropy at all--provide the dummy implementation
    from ._dummy import _DummyCommand

    class AstropyTest(_DummyCommand):
        command_name = 'test'
        description = 'Run the tests for this package'
        error_msg = (
                "The 'test' command requires the astropy package to be "
                "installed and importable.")
