import os
import sys
import types
from copy import deepcopy
from importlib import machinery
from distutils.core import Extension

from ..openmp_helpers import add_openmp_flags_if_available, generate_openmp_enabled_py
from ..setup_helpers import _module_state, register_commands

IS_TRAVIS_LINUX = os.environ.get('TRAVIS_OS_NAME', None) == 'linux'
IS_TRAVIS_OSX = os.environ.get('TRAVIS_OS_NAME', None) == 'osx'
IS_APPVEYOR = os.environ.get('APPVEYOR', None) == 'True'
OPENMP_EXPECTED = os.environ.get('OPENMP_EXPECTED', False) == 'True'
PY3_LT_35 = sys.version_info[0] == 3 and sys.version_info[1] < 5

_state = None


def setup_function(function):
    global state
    state = deepcopy(_module_state)


def teardown_function(function):
    _module_state.clear()
    _module_state.update(state)


def test_add_openmp_flags_if_available():

    register_commands('openmp_testing', '0.0', False)

    using_openmp = add_openmp_flags_if_available(Extension('test', []))

    # Make sure that on Travis (Linux) and AppVeyor OpenMP does get used (for
    # MacOS X usually it will not work but this will depend on the compiler).
    # Having this is useful because we'll find out if OpenMP no longer works
    # for any reason on platforms on which it does work at the time of writing.
    if OPENMP_EXPECTED:
        assert using_openmp
    else:
        assert not using_openmp


def test_generate_openmp_enabled_py():

    register_commands('openmp_autogeneration_testing', '0.0', False)

    # Test file generation
    generate_openmp_enabled_py('')
    assert os.path.isfile('openmp_enabled.py')

    # Load openmp_enabled file as a module to check the result
    loader = machinery.SourceFileLoader('openmp_enabled', 'openmp_enabled.py')
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)

    is_openmp_enabled = mod.is_openmp_enabled()

    # Test is_openmp_enabled()
    assert isinstance(is_openmp_enabled, bool)

    if OPENMP_EXPECTED:
        assert is_openmp_enabled
    else:
        assert not is_openmp_enabled
