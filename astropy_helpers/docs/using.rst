Using astropy-helpers in a package
==================================

astropy-helpers includes a special "bootstrap" module called ``ah_bootstrap.py``
which is intended to be used by a project's setup.py in order to ensure that the
astropy-helpers package is available for build/installation.


The easiest way to get set up with astropy-helpers in a new package is to use
the `package-template <http://docs.astropy.org/projects/package-template>`_
that we provide. This template is specifically designed for use with the helpers,
so using it avoids some of the tedium of setting up the helpers.

However, we now go through the steps of adding astropy-helpers
as a submodule to a package in case you wish to do so manually. First, add
astropy-helpers as a submodule at the root of your repository::

    git submodule add git://github.com/astropy/astropy-helpers astropy_helpers

Then go inside the submodule and check out a stable version of astropy-helpers.
You can see the available versions by running::

    $ cd astropy_helpers
    $ git tag
    ...
    v2.0.6
    v2.0.7
    ...
    v3.0.1
    v3.0.2

If you want to support Python 2, pick the latest v2.0.x version (in the above
case ``v2.0.7``) and if you don't need to support Python 2, just pick the latest
stable version (in the above case ``v3.0.2``). Check out this version with e.g.::

    $ git checkout v3.0.2

Then go back up to the root of your repository and copy the ``ah_bootstrap.py``
file from the submodule to the root of your repository::

    $ cd ..
    $ cp astropy_helpers/ah_bootstrap.py .

Finally, add::

    import ah_bootstrap

at the top of your ``setup.py`` file. This will ensure that ``astropy_helpers``
is now available to use in your ``setup.py`` file. Finally, add then commit your
changes::

    git add astropy_helpers ah_bootstrap.py setup.py
    git commit -m "Added astropy-helpers"
