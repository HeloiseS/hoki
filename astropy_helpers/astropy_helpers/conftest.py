# This file contains settings for pytest that are specific to astropy-helpers.
# Since we run many of the tests in sub-processes, we need to collect coverage
# data inside each subprocess and then combine it into a single .coverage file.
# To do this we set up a list which run_setup appends coverage objects to.
# This is not intended to be used by packages other than astropy-helpers.

import os
import glob

try:
    from coverage import CoverageData
except ImportError:
    HAS_COVERAGE = False
else:
    HAS_COVERAGE = True

if HAS_COVERAGE:
    SUBPROCESS_COVERAGE = []


def pytest_configure(config):
    if HAS_COVERAGE:
        SUBPROCESS_COVERAGE.clear()


def pytest_unconfigure(config):

    if HAS_COVERAGE:

        # We create an empty coverage data object
        combined_cdata = CoverageData()

        # Add all files from astropy_helpers to make sure we compute the total
        # coverage, not just the coverage of the files that have non-zero
        # coverage.

        lines = {}
        for filename in glob.glob(os.path.join('astropy_helpers', '**', '*.py'), recursive=True):
            lines[os.path.abspath(filename)] = []

        for cdata in SUBPROCESS_COVERAGE:
            # For each CoverageData object, we go through all the files and
            # change the filename from one which might be a temporary path
            # to the local filename. We then only keep files that actually
            # exist.
            for filename in cdata.measured_files():
                try:
                    pos = filename.rindex('astropy_helpers')
                except ValueError:
                    continue
                short_filename = filename[pos:]
                if os.path.exists(short_filename):
                    lines[os.path.abspath(short_filename)].extend(cdata.lines(filename))

        combined_cdata.add_lines(lines)

        combined_cdata.write_file('.coverage.subprocess')
