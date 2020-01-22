# -*- coding: utf-8 -*-

import glob
import os
import json
import textwrap

from distutils.version import LooseVersion

import setuptools

import pytest

from . import reset_setup_helpers, reset_distutils_log  # noqa
from . import run_cmd, run_setup, testpackage, create_testpackage
from ..utils import silence


TEST_SETUP_PY = """\
#!/usr/bin/env python
from __future__ import print_function

import os
import sys

# This import is not the real run of ah_bootstrap for the purposes of the test,
# so we need to preserve the command-line arguments otherwise these get eaten
# up by this import
args = sys.argv[:]
import ah_bootstrap
sys.argv = args

{extra}

# reset the name of the package installed by ah_boostrap to
# _astropy_helpers_test_--this will prevent any confusion by pkg_resources with
# any already installed packages named astropy_helpers
# We also disable auto-upgrade by default
ah_bootstrap.DIST_NAME = 'astropy-helpers-test'
ah_bootstrap.PACKAGE_NAME = '_astropy_helpers_test_'
ah_bootstrap.AUTO_UPGRADE = False
ah_bootstrap.DOWNLOAD_IF_NEEDED = False
try:
    ah_bootstrap.BOOTSTRAPPER = ah_bootstrap._Bootstrapper.main()
    ah_bootstrap.use_astropy_helpers({args})
finally:
    ah_bootstrap.DIST_NAME = 'astropy-helpers'
    ah_bootstrap.PACKAGE_NAME = 'astropy_helpers'
    ah_bootstrap.AUTO_UPGRADE = True
    ah_bootstrap.DOWNLOAD_IF_NEEDED = True

# Kind of a hacky way to do this, but this assertion is specifically
# for test_check_submodule_no_git
# TODO: Rework the tests in this module so that it's easier to test specific
# behaviors of ah_bootstrap for each test
assert '--no-git' not in sys.argv

import _astropy_helpers_test_
filename = os.path.abspath(_astropy_helpers_test_.__file__)
filename = filename.replace('.pyc', '.py')  # More consistent this way

# We print out variables that are needed in tests below in JSON
import json
data = {{}}
data['filename'] = filename
data['ah_bootstrap.BOOTSTRAPPER.use_git'] = ah_bootstrap.BOOTSTRAPPER.use_git
print(json.dumps(data))
"""

AH_BOOTSTRAP_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'ah_bootstrap.py')

with open(AH_BOOTSTRAP_FILE) as f:
    AH_BOOTSTRAP = f.read()

# The behavior checked in some of the tests depends on the version of
# setuptools
try:
    # We need to use LooseVersion here instead of StrictVersion since developer
    # versions of setuptools ('35.0.2.post20170530') don't satisfy the
    # StrictVersion criteria even though they satisfy PEP440
    SETUPTOOLS_VERSION = LooseVersion(setuptools.__version__).version
except:
    # Broken setuptools? ¯\_(ツ)_/¯
    SETUPTOOLS_VERSION = (0, 0, 0)


def test_bootstrap_from_submodule(tmpdir, testpackage, capsys):
    """
    Tests importing _astropy_helpers_test_ from a submodule in a git
    repository.  This tests actually performing a fresh clone of the repository
    without the submodule initialized, and that importing astropy_helpers in
    that context works transparently after calling
    `ah_boostrap.use_astropy_helpers`.
    """

    orig_repo = tmpdir.mkdir('orig')

    with orig_repo.as_cwd():

        run_cmd('git', ['init'])

        orig_repo.join('ah_bootstrap.py').write(AH_BOOTSTRAP)
        run_cmd('git', ['add', 'ah_bootstrap.py'])

        # Write a test setup.py that uses ah_bootstrap; it also ensures that
        # any previous reference to astropy_helpers is first wiped from
        # sys.modules
        orig_repo.join('setup.py').write(TEST_SETUP_PY.format(args='', extra=''))
        run_cmd('git', ['add', 'setup.py'])

        # Add our own clone of the astropy_helpers repo as a submodule named
        # astropy_helpers
        run_cmd('git', ['submodule', 'add', str(testpackage),
                        '_astropy_helpers_test_'])

        run_cmd('git', ['commit', '-m', 'test repository'])

        os.chdir(str(tmpdir))

        # Creates a clone of our test repo in the directory 'clone'
        run_cmd('git', ['clone', 'orig', 'clone'])

        os.chdir('clone')

        run_setup('setup.py', [])

        stdout, stderr = capsys.readouterr()
        path = json.loads(stdout.strip())['filename']

        # Ensure that the astropy_helpers used by the setup.py is the one that
        # was imported from git submodule
        a = os.path.normcase(path)
        b = os.path.normcase(str(tmpdir.join('clone', '_astropy_helpers_test_',
                                             '_astropy_helpers_test_',
                                             '__init__.py')))
        assert a == b


def test_bootstrap_from_submodule_no_locale(tmpdir, testpackage, capsys,
                                            monkeypatch):
    """
    Regression test for https://github.com/astropy/astropy/issues/2749

    Runs test_bootstrap_from_submodule but with missing locale/language
    settings.
    """

    for varname in ('LC_ALL', 'LC_CTYPE', 'LANG', 'LANGUAGE'):
        monkeypatch.delenv(varname, raising=False)

    test_bootstrap_from_submodule(tmpdir, testpackage, capsys)


def test_bootstrap_from_submodule_bad_locale(tmpdir, testpackage, capsys,
                                             monkeypatch):
    """
    Additional regression test for
    https://github.com/astropy/astropy/issues/2749
    """

    for varname in ('LC_ALL', 'LC_CTYPE', 'LANG', 'LANGUAGE'):
        monkeypatch.delenv(varname, raising=False)

    # Test also with bad LC_CTYPE a la http://bugs.python.org/issue18378
    monkeypatch.setenv('LC_CTYPE', 'UTF-8')

    test_bootstrap_from_submodule(tmpdir, testpackage, capsys)


UPDATE_ERROR_PATCH = """
class UpgradeError(Exception):
    pass

def _do_upgrade(*args, **kwargs):
    raise UpgradeError()

ah_bootstrap._Bootstrapper._do_upgrade = _do_upgrade
"""


def test_check_submodule_no_git(capsys, tmpdir, testpackage):
    """
    Tests that when importing astropy_helpers from a submodule, it is still
    recognized as a submodule even when using the --no-git option.

    In particular this ensures that the auto-upgrade feature is not activated.
    """

    orig_repo = tmpdir.mkdir('orig')

    with orig_repo.as_cwd():

        orig_repo.join('ah_bootstrap.py').write(AH_BOOTSTRAP)

        run_cmd('git', ['init'])

        # Write a test setup.py that uses ah_bootstrap; it also ensures that
        # any previous reference to astropy_helpers is first wiped from
        # sys.modules
        args = 'auto_upgrade=True'
        orig_repo.join('setup.py').write(TEST_SETUP_PY.format(args=args, extra=UPDATE_ERROR_PATCH))
        run_cmd('git', ['add', 'setup.py'])

        # Add our own clone of the astropy_helpers repo as a submodule named
        # astropy_helpers
        run_cmd('git', ['submodule', 'add', str(testpackage),
                        '_astropy_helpers_test_'])

        run_cmd('git', ['commit', '-m', 'test repository'])

        run_setup('setup.py', ['--no-git'])

        stdout, stderr = capsys.readouterr()

        use_git = bool(json.loads(stdout.strip())['ah_bootstrap.BOOTSTRAPPER.use_git'])

        if 'UpgradeError' in stderr:
            pytest.fail('Attempted to run auto-upgrade despite importing '
                        '_astropy_helpers_test_ from a git submodule')

        # Ensure that the no-git option was in fact set
        assert not use_git


def test_bootstrap_from_directory(tmpdir, testpackage, capsys):
    """
    Tests simply bundling a copy of the astropy_helpers source code in its
    entirety bundled directly in the source package and not in an archive.
    """

    source = tmpdir.mkdir('source')
    testpackage.copy(source.join('_astropy_helpers_test_'))

    with source.as_cwd():

        source.join('ah_bootstrap.py').write(AH_BOOTSTRAP)

        source.join('setup.py').write(TEST_SETUP_PY.format(args='', extra=''))
        run_setup('setup.py', [])
        stdout, stderr = capsys.readouterr()
        path = json.loads(stdout.strip())['filename']

        # Ensure that the astropy_helpers used by the setup.py is the one that
        # was imported from git submodule
        a = os.path.normcase(path)
        b = os.path.normcase(str(source.join('_astropy_helpers_test_',
                                             '_astropy_helpers_test_',
                                             '__init__.py')))
        assert a == b


def test_bootstrap_from_archive(tmpdir, testpackage, capsys):
    """
    Tests importing _astropy_helpers_test_ from a .tar.gz source archive
    shipped alongside the package that uses it.
    """

    orig_repo = tmpdir.mkdir('orig')

    # Make a source distribution of the test package
    with silence():
        run_setup(str(testpackage.join('setup.py')),
                  ['sdist', '--dist-dir=dist', '--formats=gztar'])

    dist_dir = testpackage.join('dist')
    for dist_file in dist_dir.visit('*.tar.gz'):
        dist_file.copy(orig_repo)

    with orig_repo.as_cwd():

        orig_repo.join('ah_bootstrap.py').write(AH_BOOTSTRAP)

        # Write a test setup.py that uses ah_bootstrap; it also ensures that
        # any previous reference to astropy_helpers is first wiped from
        # sys.modules
        args = 'path={0!r}'.format(os.path.basename(str(dist_file)))
        orig_repo.join('setup.py').write(TEST_SETUP_PY.format(args=args, extra=''))

        run_setup('setup.py', [])

        stdout, stderr = capsys.readouterr()
        path = json.loads(stdout.strip())['filename']

        # Installation from the .tar.gz should have resulted in a .egg
        # directory that the _astropy_helpers_test_ package was imported from
        eggs = _get_local_eggs()
        assert eggs
        egg = orig_repo.join(eggs[0])
        assert os.path.isdir(str(egg))

        a = os.path.normcase(path)
        b = os.path.normcase(str(egg.join('_astropy_helpers_test_',
                                          '__init__.py')))

        assert a == b


def test_download_if_needed(tmpdir, testpackage, capsys):
    """
    Tests the case where astropy_helpers was not actually included in a
    package, or is otherwise missing, and we need to "download" it.

    This does not test actually downloading from the internet--this is normally
    done through setuptools' easy_install command which can also install from a
    source archive.  From the point of view of ah_boostrap the two actions are
    equivalent, so we can just as easily simulate this by providing a setup.cfg
    giving the path to a source archive to "download" (as though it were a
    URL).
    """

    source = tmpdir.mkdir('source')

    # Ensure ah_bootstrap is imported from the local directory
    import ah_bootstrap  # noqa

    # Make a source distribution of the test package
    with silence():
        run_setup(str(testpackage.join('setup.py')),
                  ['sdist', '--dist-dir=dist', '--formats=gztar'])

    dist_dir = testpackage.join('dist')

    with source.as_cwd():

        source.join('ah_bootstrap.py').write(AH_BOOTSTRAP)

        source.join('setup.py').write(TEST_SETUP_PY.format(
            args='download_if_needed=True', extra=''))
        source.join('setup.cfg').write(textwrap.dedent("""\
            [easy_install]
            find_links = {find_links}
        """.format(find_links=str(dist_dir))))

        run_setup('setup.py', [])

        stdout, stderr = capsys.readouterr()
        path = json.loads(stdout.strip())['filename']

        # easy_install should have worked by 'installing' astropy_helpers as a
        # .egg in the current directory
        eggs = _get_local_eggs()
        assert eggs
        egg = source.join(eggs[0])
        assert os.path.isdir(str(egg))

        a = os.path.normcase(path)
        b = os.path.normcase(str(egg.join('_astropy_helpers_test_',
                                          '__init__.py')))
        assert a == b


EXTRA_PACKAGE_INDEX = """
from setuptools.package_index import PackageIndex

class FakePackageIndex(PackageIndex):
    def __init__(self, *args, **kwargs):
        PackageIndex.__init__(self, *args, **kwargs)
        self.to_scan = {dists}

    def find_packages(self, requirement):
        # no-op
        pass

ah_bootstrap.PackageIndex = FakePackageIndex
"""


def test_upgrade(tmpdir, capsys):
    orig_dir = create_testpackage(tmpdir.mkdir('orig'))

    # Make a test package that uses _astropy_helpers_test_
    source = tmpdir.mkdir('source')
    dist_dir = source.mkdir('dists')
    orig_dir.copy(source.join('_astropy_helpers_test_'))

    with source.as_cwd():

        source.join('ah_bootstrap.py').write(AH_BOOTSTRAP)

        setup_py = TEST_SETUP_PY.format(args='auto_upgrade=True', extra='')
        source.join('setup.py').write(setup_py)

        # This will be used to later to fake downloading the upgrade package
        source.join('setup.cfg').write(textwrap.dedent("""\
            [easy_install]
            find_links = {find_links}
        """.format(find_links=str(dist_dir))))

    # Make additional "upgrade" versions of the _astropy_helpers_test_
    # package--one of them is version 0.2 and the other is version 0.1.1.  The
    # auto-upgrade should ignore version 0.2 but use version 0.1.1.
    upgrade_dir_1 = create_testpackage(tmpdir.mkdir('upgrade_1'), version='0.2')
    upgrade_dir_2 = create_testpackage(tmpdir.mkdir('upgrade_2'), version='0.1.1')

    dists = []
    # For each upgrade package go ahead and build a source distribution of it
    # and copy that source distribution to a dist directory we'll use later to
    # simulate a 'download'
    for upgrade_dir in [upgrade_dir_1, upgrade_dir_2]:
        with silence():
            run_setup(str(upgrade_dir.join('setup.py')),
                      ['sdist', '--dist-dir=dist', '--formats=gztar'])
        dists.append(str(upgrade_dir.join('dist')))
        for dist_file in upgrade_dir.visit('*.tar.gz'):
            dist_file.copy(source.join('dists'))

    with source.as_cwd():

        setup_py = TEST_SETUP_PY.format(args='auto_upgrade=True',
                                        extra=EXTRA_PACKAGE_INDEX.format(dists=dists))
        source.join('setup.py').write(setup_py)

        # Now run the source setup.py; this test is similar to
        # test_download_if_needed, but we explicitly check that the correct
        # *version* of _astropy_helpers_test_ was used
        run_setup('setup.py', [])

        stdout, stderr = capsys.readouterr()
        path = json.loads(stdout.strip())['filename']
        eggs = _get_local_eggs()
        assert eggs

        egg = source.join(eggs[0])
        assert os.path.isdir(str(egg))
        a = os.path.normcase(path)
        b = os.path.normcase(str(egg.join('_astropy_helpers_test_',
                                          '__init__.py')))
        assert a == b
        assert 'astropy_helpers_test-0.1.1-' in str(egg)


def _get_local_eggs(path='.'):
    """
    Helper utility used by some tests to get the list of egg archive files
    in a local directory.
    """

    if SETUPTOOLS_VERSION[0] >= 7:
        eggs = glob.glob(os.path.join(path, '.eggs', '*.egg'))
    else:
        eggs = glob.glob('*.egg')

    return eggs
