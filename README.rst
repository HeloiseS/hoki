.. image:: black_text.png
    :height: 35px


.. image:: https://zenodo.org/badge/197853216.svg
   :target: https://zenodo.org/badge/latestdoi/197853216
.. image:: https://img.shields.io/pypi/v/hoki?style=flat-square   :alt: PyPI

.. image:: https://travis-ci.org/HeloiseS/hoki.svg?branch=master
    :target: https://travis-ci.org/HeloiseS/hoki
    
.. image:: https://codecov.io/gh/HeloiseS/hoki/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/HeloiseS/hoki
    
.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge
    
Bridging the Gap Between Observation and Theory
==============================================


In order to facilitate the application of **BPASS** to a wide range of scientific investigations, we have developped the tools necessary for observers to take full advantage of our models in an intuitive manner. 

Hoki isn't only versatile, it also takes care of the nitty gritty pre-processing!

**Spend less time on coding and more time on the science!**

**WHAT IS BPASS?**

   *The Binary Populations And Spectral Synthesis (BPASS) code simulates stellar populations and follows their evolution until death. Including binary evolution is crucial to correctly interpreting astronomical observations. The detailed follow-up of the stellar evolution within the code allows the retreival of important information such as supernova and gravitational wave event rates, giving us the ability to understand the properties of the stellar populations that are the precursors of these events.*

----
   
Installing hoki
^^^^^^^^^^^

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

   `astropy`, `numpy`, `pandas`, `matplotlib`, `pyyaml`, `wheel`

**Note:** Python 2 is not supported

----

Read the docs
^^^^^^^^^^^

`Click Here! Click Here! Click Here! <https://heloises.github.io/hoki/intro.html>`_

----

Download Tutorials
^^^^^^^^^^^^^^^
Check out these Jupyter notebooks I made - you can find them on `this repo! <https://github.com/HeloiseS/hoki_tutorials>`__

---- 

Citation
^^^^^^^^^
A peer-reviewed journal article about `hoki` will appear in due time but for now you can use the following bibtex entry:

.. code-block:: none

   @Misc{hoki_citation,
     author =   {Heloise Stevance},
     title =    {Hoki},
     howpublished = {\url{https://github.com/HeloiseS/hoki}},
     doi = {10.5281/zenodo.3445659},
     year = {2019}
     } 
     
**Please if you use `hoki` for your science, include us in your publications!** As you can imagine developing a tool and maintaining it for the community is very time consuming, and unfortunatly citations remain the most important metric. 

---- 

License
^^^^^^^^^^^

This project is Copyright (c) H. F. Stevance and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause licence. 

----

Contributing
^^^^^^^^^^^

If ANYTHING comes to mind, whether it be something in the tutorials, features you would like us to consider, BUGS, etc.. 
**Please just drop it in an issue! Don't let your imposter syndrome talk you out of it ;)**


