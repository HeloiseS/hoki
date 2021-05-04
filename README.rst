.. image:: black_text.png
    :height: 35px

   
.. image:: https://img.shields.io/pypi/v/hoki?style=flat-square   
    :alt: PyPI
    
.. image:: https://zenodo.org/badge/197853216.svg
    :target: https://zenodo.org/badge/latestdoi/197853216
   
.. image:: https://github.com/HeloiseS/hoki/workflows/CI/badge.svg   
    :alt: tests
    
.. image:: https://img.shields.io/pypi/dm/hoki
    
Bridging the Gap Between Observation and Theory
=================================================


In order to facilitate the application of **BPASS** to a wide range of scientific investigations, we have developped the tools necessary for observers to take full advantage of our models in an intuitive manner. 

Hoki isn't only versatile, it also takes care of the nitty gritty pre-processing!

**Spend less time on coding and more time on the science!**

**WHAT IS BPASS?**

   *The Binary Populations And Spectral Synthesis (BPASS) code simulates stellar populations and follows their evolution until death. Including binary evolution is crucial to correctly interpreting astronomical observations. The detailed follow-up of the stellar evolution within the code allows the retreival of important information such as supernova and gravitational wave event rates, giving us the ability to understand the properties of the stellar populations that are the precursors of these events.*

----
   
Installing hoki
^^^^^^^^^^^^^^^^^

You can pip install the **most recent stable release** from pip:

.. code-block:: none

   pip3 install --user hoki
   
You can also download the **github dev version** with the following command:

.. code-block:: none

   pip3 install . git+https://github.com/HeloiseS/hoki

If you install the development version of hoki from github, we recommend you do so in a `conda environment <https://www.anaconda.com>`_ 
To check that hoki is up and running you can run the unit tests in the test folder. I like using `pytest` (which you'll have to download) and run 

.. code-block:: none

   pytest --verbose

This way it'll show you each test as they pass or FAIL. In the pip and github version, they should all pass on your machine as data is provided within the package to test the functionalities.


**Requirements:** The following packages are required by `hoki`. If you pip install the stable version from pypi it will all be done automatically.

.. code-block:: none

   `astropy`, `numpy`, `pandas`, `matplotlib`, `pyyaml`, `wheel`, `emcee`, `corner`, `numba`


**Note:** Python 2 is not supported

----

Read the docs
^^^^^^^^^^^^^^^


`Click Here! Click Here! Click Here! <https://heloises.github.io/hoki/intro.html>`_

----

Download Tutorials
^^^^^^^^^^^^^^^^^^^^

Check out these Jupyter notebooks I made - you can find them on `this repo! <https://github.com/HeloiseS/hoki_tutorials>`__

---- 

Paper and how to cite us!
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. image:: https://joss.theoj.org/papers/10.21105/joss.01987/status.svg
   :target: https://doi.org/10.21105/joss.01987
   
Our paper *Hoki: Making BPASS Accessible Through Python* has now been published in the Journal of Open Source Software (JOSS). (See DOI above)

The paper is `available on ArXiv <https://arxiv.org/abs/2001.11069>`_ as published by JOSS

**Please if you use `hoki` for your science, include us in your publications!** As you can imagine developing a tool and maintaining it for the community is very time consuming, and unfortunatly citations remain the most important metric. 

If you use the following tools, please acknowledge the follwing publications:

**AgeWizard**:

.. image:: https://img.shields.io/badge/arxiv-2004.02883-red
   :target: https://arxiv.org/abs/2004.02883

**UnderlyingCountRatio**:

.. image:: https://img.shields.io/badge/arxiv-2004.13040-red
   :target: https://arxiv.org/abs/2004.13040

**BIBTEX**

.. code-block::

   @ARTICLE{2020JOSS....5.1987S,
       author = {{Stevance}, Heloise and {Eldridge}, J. and {Stanway}, Elizabeth},
        title = "{Hoki: Making BPASS accessible through Python}",
      journal = {The Journal of Open Source Software},
     keywords = {Python, galaxies, Batchfile, SED, astronomy, binary stars, Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Astrophysics of Galaxies, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = "2020",
        month = "Jan",
       volume = {5},
       number = {45},
          eid = {1987},
        pages = {1987},
          doi = {10.21105/joss.01987},
     archivePrefix = {arXiv},
       eprint = {2001.11069},
     primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020JOSS....5.1987S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
     


---- 

License
^^^^^^^^^^^

This project is Copyright (c) H. F. Stevance and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause licence. 

----

Contributing
^^^^^^^^^^^^^^

If ANYTHING comes to mind, whether it be something in the tutorials, features you would like us to consider, BUGS, etc.. 
**Please just drop it in an issue! Don't let your imposter syndrome talk you out of it ;)**


.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge
    
