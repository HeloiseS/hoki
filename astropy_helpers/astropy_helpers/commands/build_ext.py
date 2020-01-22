import errno
import os
import shutil

from distutils.core import Extension
from distutils.ccompiler import get_default_compiler
from distutils.command.build_ext import build_ext as DistutilsBuildExt

from ..distutils_helpers import get_main_package_directory
from ..utils import get_numpy_include_path, import_file

__all__ = ['AstropyHelpersBuildExt']


def should_build_with_cython(previous_cython_version, is_release):
    """
    Returns the previously used Cython version (or 'unknown' if not
    previously built) if Cython should be used to build extension modules from
    pyx files.
    """

    # Only build with Cython if, of course, Cython is installed, we're in a
    # development version (i.e. not release) or the Cython-generated source
    # files haven't been created yet (cython_version == 'unknown'). The latter
    # case can happen even when release is True if checking out a release tag
    # from the repository
    have_cython = False
    try:
        from Cython import __version__ as cython_version  # noqa
        have_cython = True
    except ImportError:
        pass

    if have_cython and (not is_release or previous_cython_version == 'unknown'):
        return cython_version
    else:
        return False


class AstropyHelpersBuildExt(DistutilsBuildExt):
    """
    A custom 'build_ext' command that allows for manipulating some of the C
    extension options at build time.
    """

    _uses_cython = False
    _force_rebuild = False

    def __new__(cls, value, **kwargs):

        # NOTE: we need to wait until AstropyHelpersBuildExt is initialized to
        # import setuptools.command.build_ext because when that package is
        # imported, setuptools tries to import Cython - and if it's not found
        # it will affect the rest of the build process. This is an issue because
        # if we import that module at the top of this one, setup_requires won't
        # have been honored yet, so Cython may not yet be available - and if we
        # import build_ext too soon, it will think Cython is not available even
        # if it is then intalled when setup_requires is processed. To get around
        # this we dynamically create a new class that inherits from the
        # setuptools build_ext, and by this point setup_requires has been
        # processed.

        from setuptools.command.build_ext import build_ext as SetuptoolsBuildExt

        class FinalBuildExt(AstropyHelpersBuildExt, SetuptoolsBuildExt):
            pass

        new_type = type(cls.__name__, (FinalBuildExt,), dict(cls.__dict__))
        obj = SetuptoolsBuildExt.__new__(new_type)
        obj.__init__(value)

        return obj

    def finalize_options(self):

        # First let's find the package folder, then we can check if the
        # version and cython_version are accessible
        self.package_dir = get_main_package_directory(self.distribution)

        version = import_file(os.path.join(self.package_dir, 'version.py'),
                              name='version').version
        self.is_release = 'dev' not in version

        try:
            self.previous_cython_version = import_file(os.path.join(self.package_dir,
                                                                    'cython_version.py'),
                                                       name='cython_version').cython_version
        except (FileNotFoundError, ImportError):
            self.previous_cython_version = 'unknown'

        self._uses_cython = should_build_with_cython(self.previous_cython_version, self.is_release)

        # Add a copy of the _compiler.so module as well, but only if there
        # are in fact C modules to compile (otherwise there's no reason to
        # include a record of the compiler used). Note that self.extensions
        # may not be set yet, but self.distribution.ext_modules is where any
        # extension modules passed to setup() can be found
        extensions = self.distribution.ext_modules
        if extensions:
            build_py = self.get_finalized_command('build_py')
            package_dir = build_py.get_package_dir(self.package_dir)
            src_path = os.path.relpath(
                os.path.join(os.path.dirname(__file__), 'src'))
            shutil.copy(os.path.join(src_path, 'compiler.c'),
                        os.path.join(package_dir, '_compiler.c'))
            ext = Extension(self.package_dir + '.compiler_version',
                            [os.path.join(package_dir, '_compiler.c')])
            extensions.insert(0, ext)

        super().finalize_options()

        # If we are using Cython, then make sure we re-build if the version
        # of Cython that is installed is different from the version last
        # used to generate the C files.
        if self._uses_cython and self._uses_cython != self.previous_cython_version:
            self._force_rebuild = True

        # Regardless of the value of the '--force' option, force a rebuild
        # if the debug flag changed from the last build
        if self._force_rebuild:
            self.force = True

    def run(self):

        # For extensions that require 'numpy' in their include dirs,
        # replace 'numpy' with the actual paths
        np_include = None
        for extension in self.extensions:
            if 'numpy' in extension.include_dirs:
                if np_include is None:
                    np_include = get_numpy_include_path()
                idx = extension.include_dirs.index('numpy')
                extension.include_dirs.insert(idx, np_include)
                extension.include_dirs.remove('numpy')

            self._check_cython_sources(extension)

        # Note that setuptools automatically uses Cython to discover and
        # build extensions if available, so we don't have to explicitly call
        # e.g. cythonize.

        super().run()

        # Update cython_version.py if building with Cython

        if self._uses_cython and self._uses_cython != self.previous_cython_version:
            build_py = self.get_finalized_command('build_py')
            package_dir = build_py.get_package_dir(self.package_dir)
            cython_py = os.path.join(package_dir, 'cython_version.py')
            with open(cython_py, 'w') as f:
                f.write('# Generated file; do not modify\n')
                f.write('cython_version = {0!r}\n'.format(self._uses_cython))

            if os.path.isdir(self.build_lib):
                # The build/lib directory may not exist if the build_py
                # command was not previously run, which may sometimes be
                # the case
                self.copy_file(cython_py,
                               os.path.join(self.build_lib, cython_py),
                               preserve_mode=False)

    def _check_cython_sources(self, extension):
        """
        Where relevant, make sure that the .c files associated with .pyx
        modules are present (if building without Cython installed).
        """

        # Determine the compiler we'll be using
        if self.compiler is None:
            compiler = get_default_compiler()
        else:
            compiler = self.compiler

        # Replace .pyx with C-equivalents, unless c files are missing
        for jdx, src in enumerate(extension.sources):
            base, ext = os.path.splitext(src)
            pyxfn = base + '.pyx'
            cfn = base + '.c'
            cppfn = base + '.cpp'

            if not os.path.isfile(pyxfn):
                continue

            if self._uses_cython:
                extension.sources[jdx] = pyxfn
            else:
                if os.path.isfile(cfn):
                    extension.sources[jdx] = cfn
                elif os.path.isfile(cppfn):
                    extension.sources[jdx] = cppfn
                else:
                    msg = (
                        'Could not find C/C++ file {0}.(c/cpp) for Cython '
                        'file {1} when building extension {2}. Cython '
                        'must be installed to build from a git '
                        'checkout.'.format(base, pyxfn, extension.name))
                    raise IOError(errno.ENOENT, msg, cfn)

            # Cython (at least as of 0.29.2) uses deprecated Numpy API features
            # the use of which produces a few warnings when compiling.
            # These additional flags should squelch those warnings.
            # TODO: Feel free to remove this if/when a Cython update
            # removes use of the deprecated Numpy API
            if compiler == 'unix':
                extension.extra_compile_args.extend([
                    '-Wp,-w', '-Wno-unused-function'])
