import glob
import imp
import os
import pkgutil
import re
import sys
import tarfile

import pytest
from warnings import catch_warnings

from . import reset_setup_helpers, reset_distutils_log  # noqa
from . import run_cmd, run_setup, cleanup_import
from astropy_helpers.git_helpers import get_git_devstr

_DEV_VERSION_RE = re.compile(r'\d+\.\d+(?:\.\d+)?\.dev(\d+)')

ASTROPY_HELPERS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

TEST_VERSION_SETUP_PY_OLDSTYLE = """\
#!/usr/bin/env python

import sys
from setuptools import setup

NAME = 'apyhtest_eva'
VERSION = {version!r}
RELEASE = 'dev' not in VERSION

sys.path.insert(0, r'{astropy_helpers_path}')

from astropy_helpers.git_helpers import get_git_devstr
from astropy_helpers.version_helpers import generate_version_py

if not RELEASE:
    VERSION += get_git_devstr(False)

generate_version_py(NAME, VERSION, RELEASE, False, uses_git=not RELEASE)

setup(name=NAME, version=VERSION, packages=['apyhtest_eva'])
"""

TEST_VERSION_SETUP_CFG = """\
[metadata]
name = apyhtest_eva
version = {version}
"""

TEST_VERSION_SETUP_PY_NEWSTYLE = """\
#!/usr/bin/env python

import sys
sys.path.insert(0, r'{astropy_helpers_path}')

from astropy_helpers.setup_helpers import setup
setup()
"""


TEST_VERSION_INIT = """\
try:
    from .version import version as __version__
    from .version import githash as __githash__
except ImportError:
    __version__ = __githash__ = ''
"""


@pytest.fixture(params=["oldstyle", "newstyle"])
def version_test_package(tmpdir, request):

    # We test both the old-style syntax of deermining VERSION, RELEASE, etc.
    # inside the setup.py, and the new style of getting these from the setup.cfg
    # file.

    def make_test_package_oldstyle(version='42.42.dev'):
        test_package = tmpdir.mkdir('test_package')
        test_package.join('setup.py').write(
            TEST_VERSION_SETUP_PY_OLDSTYLE.format(version=version,
                                                  astropy_helpers_path=ASTROPY_HELPERS_PATH))
        test_package.mkdir('apyhtest_eva').join('__init__.py').write(TEST_VERSION_INIT)
        with test_package.as_cwd():
            run_cmd('git', ['init'])
            run_cmd('git', ['add', '--all'])
            run_cmd('git', ['commit', '-m', 'test package'])

        if '' in sys.path:
            sys.path.remove('')

        sys.path.insert(0, '')

        def finalize():
            cleanup_import('apyhtest_eva')

        request.addfinalizer(finalize)

        return test_package

    def make_test_package_newstyle(version='42.42.dev'):
        test_package = tmpdir.mkdir('test_package')
        test_package.join('setup.cfg').write(
            TEST_VERSION_SETUP_CFG.format(version=version))

        test_package.join('setup.py').write(
            TEST_VERSION_SETUP_PY_NEWSTYLE.format(astropy_helpers_path=ASTROPY_HELPERS_PATH))

        test_package.mkdir('apyhtest_eva').join('__init__.py').write(TEST_VERSION_INIT)
        with test_package.as_cwd():
            run_cmd('git', ['init'])
            run_cmd('git', ['add', '--all'])
            run_cmd('git', ['commit', '-m', 'test package'])

        if '' in sys.path:
            sys.path.remove('')

        sys.path.insert(0, '')

        def finalize():
            cleanup_import('apyhtest_eva')

        request.addfinalizer(finalize)

        return test_package

    if request.param == 'oldstyle':
        return make_test_package_oldstyle
    else:
        return make_test_package_newstyle


def test_update_git_devstr(version_test_package, capsys):
    """Tests that the commit number in the package's version string updates
    after git commits even without re-running setup.py.
    """

    # We have to call version_test_package to actually create the package
    test_pkg = version_test_package()

    with test_pkg.as_cwd():
        run_setup('setup.py', ['--version'])

        stdout, stderr = capsys.readouterr()
        version = stdout.strip()

        m = _DEV_VERSION_RE.match(version)
        assert m, (
            "Stdout did not match the version string pattern:"
            "\n\n{0}\n\nStderr:\n\n{1}".format(stdout, stderr))
        revcount = int(m.group(1))

        import apyhtest_eva
        assert apyhtest_eva.__version__ == version

        # Make a silly git commit
        with open('.test', 'w'):
            pass

        run_cmd('git', ['add', '.test'])
        run_cmd('git', ['commit', '-m', 'test'])

        import apyhtest_eva.version
        imp.reload(apyhtest_eva.version)

    # Previously this checked packagename.__version__, but in order for that to
    # be updated we also have to re-import _astropy_init which could be tricky.
    # Checking directly that the packagename.version module was updated is
    # sufficient:
    m = _DEV_VERSION_RE.match(apyhtest_eva.version.version)
    assert m
    assert int(m.group(1)) == revcount + 1

    # This doesn't test astropy_helpers.get_helpers.update_git_devstr directly
    # since a copy of that function is made in packagename.version (so that it
    # can work without astropy_helpers installed).  In order to get test
    # coverage on the actual astropy_helpers copy of that function just call it
    # directly and compare to the value in packagename
    from astropy_helpers.git_helpers import update_git_devstr

    newversion = update_git_devstr(version, path=str(test_pkg))
    assert newversion == apyhtest_eva.version.version


def test_version_update_in_other_repos(version_test_package, tmpdir):
    """
    Regression test for https://github.com/astropy/astropy-helpers/issues/114
    and for https://github.com/astropy/astropy-helpers/issues/107
    """

    test_pkg = version_test_package()

    with test_pkg.as_cwd():
        run_setup('setup.py', ['build'])

    # Add the path to the test package to sys.path for now
    sys.path.insert(0, str(test_pkg))
    try:
        import apyhtest_eva
        m = _DEV_VERSION_RE.match(apyhtest_eva.__version__)
        assert m
        correct_revcount = int(m.group(1))

        with tmpdir.as_cwd():
            testrepo = tmpdir.mkdir('testrepo')
            testrepo.chdir()
            # Create an empty git repo
            run_cmd('git', ['init'])

            import apyhtest_eva.version
            imp.reload(apyhtest_eva.version)
            m = _DEV_VERSION_RE.match(apyhtest_eva.version.version)
            assert m
            assert int(m.group(1)) == correct_revcount
            correct_revcount = int(m.group(1))

            # Add several commits--more than the revcount for the apyhtest_eva package
            for idx in range(correct_revcount + 5):
                test_filename = '.test' + str(idx)
                testrepo.ensure(test_filename)
                run_cmd('git', ['add', test_filename])
                run_cmd('git', ['commit', '-m', 'A message'])

            import apyhtest_eva.version
            imp.reload(apyhtest_eva.version)
            m = _DEV_VERSION_RE.match(apyhtest_eva.version.version)
            assert m
            assert int(m.group(1)) == correct_revcount
            correct_revcount = int(m.group(1))
    finally:
        sys.path.remove(str(test_pkg))


@pytest.mark.parametrize('version', ['1.0.dev', '1.0'])
def test_installed_git_version(version_test_package, version, tmpdir, capsys):
    """
    Test for https://github.com/astropy/astropy-helpers/issues/87

    Ensures that packages installed with astropy_helpers have a correct copy
    of the git hash of the installed commit.
    """

    # To test this, it should suffice to build a source dist, unpack it
    # somewhere outside the git repository, and then do a build and import
    # from the build directory--no need to "install" as such

    test_pkg = version_test_package(version)

    with test_pkg.as_cwd():
        run_setup('setup.py', ['build'])

        try:
            import apyhtest_eva
            githash = apyhtest_eva.__githash__
            assert githash and isinstance(githash, str)
            # Ensure that it does in fact look like a git hash and not some
            # other arbitrary string
            assert re.match(r'[0-9a-f]{40}', githash)
        finally:
            cleanup_import('apyhtest_eva')

        run_setup('setup.py', ['sdist', '--dist-dir=dist', '--formats=gztar'])

        tgzs = glob.glob(os.path.join('dist', '*.tar.gz'))
        assert len(tgzs) == 1

        tgz = test_pkg.join(tgzs[0])

    build_dir = tmpdir.mkdir('build_dir')
    tf = tarfile.open(str(tgz), mode='r:gz')
    tf.extractall(str(build_dir))

    with build_dir.as_cwd():
        pkg_dir = glob.glob('apyhtest_eva-*')[0]
        os.chdir(pkg_dir)

        with catch_warnings(record=True) as w:
            run_setup('setup.py', ['build'])

        try:
            import apyhtest_eva
            loader = pkgutil.get_loader('apyhtest_eva')
            # Ensure we are importing the 'packagename' that was just unpacked
            # into the build_dir
            assert loader.get_filename().startswith(str(build_dir))
            assert apyhtest_eva.__githash__ == githash
        finally:
            cleanup_import('apyhtest_eva')


def test_get_git_devstr(tmpdir):
    dirpath = str(tmpdir)
    warn_msg = "No git repository present at"
    # Verify as much as possible, but avoid dealing with paths on windows
    if not sys.platform.startswith('win'):
        warn_msg += " '{}'".format(dirpath)

    with catch_warnings(record=True) as w:
        devstr = get_git_devstr(path=dirpath)
        assert devstr == '0'
        assert len(w) == 1
        assert str(w[0].message).startswith(warn_msg)
