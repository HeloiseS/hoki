#############
Quick Start
#############


**************
Set-up
**************

Install Hoki
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the most recent stable release of hoki you can use:

.. code-block:: none

   sudo pip3 install --user hoki
   
If you are feeling adventurous and want the most recent (in development) version of hoki, you can clone our `GitHub repository <https://github.com/HeloiseS/hoki>`__
   

Download the BPASS models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BPASS models can be downloaded from the BPASS:

.. list-table::
   :widths: 20 5 20 50
   :header-rows: 1
   
   * - Download
     - Hoki Compatible
     - Release Date
     - Reference
   * - `BPASSv2.2 <https://bpass.auckland.ac.nz/9.html>`__
     - Yes
     - July 2018
     - `Stanway & Eldridge (2018) <https://ui.adsabs.harvard.edu/abs/2018MNRAS.479...75S/abstract>`__
   * - `BPASSv2.1 <https://bpass.auckland.ac.nz/8.html>`__
     - ?
     - October 2017
     - `Eldridge, Stanway, Xiao, et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017PASA...34...58E/abstract>`__
   * - `BPASSv2.0 <https://bpass.auckland.ac.nz/2.html>`__
     - ?
     - --
     - -- 
   * - `BPASSv1 <https://bpass.auckland.ac.nz/1.html>`__
     - ?
     - --
     - --

.. note::
   
   ``hoki`` is dedicated to being an interface with the BPASS models, but given the substancial size of the entire set of models, they are not downloaded upon installation of ``hoki``, and you should download the models you want to work on.


***************
Loading in Data
***************

Stellar Model Outputs
^^^^^^^^^^^^^^^^^^^^^^
W.I.P

Stellar Population Outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^

A stellar output file can be loaded in using the ``population_output()`` function in the ``load`` module. 

.. code-block:: python
   :linenos:
   
   from hoki import load
   
   output = load.population_output('path')
   

The function will figure out based on the file name what data is being loaded in and will return the appropriate data format: ``pandas.DataFrames`` in most cases, apart from HR diagrams, which have their own ``HRDiagrams`` class -- because they're quite a complex data structure. 

.. tip::
  The full details of the he stellar population outputs can be found in the `BPASS manual <https://bpass.auckland.ac.nz/8/files/bpassv2_1_manual_accessible_version.pdf>`__. 

Here we summarise the shape of the outputs (51 time bins) for a given metalicity and IMF. 

.. list-table::
   :header-rows: 1
   
   * - Output
     - File Name Root
     - Shape
   * - Massive star type numbers
     - numbers
     - 51 x 21
   * - Supernova Rates
     - supernova
     - 51 x 18
   * - Energy and elemental yields
     - yields
     - 51 x 9
   * - Stellar mass remaining at the end
     - starmass
     - 51 x 3
   * - HR diagrams
     - hrs
     - 51 x 100 x 100 x 3 x 3
     
.. note:: These models are calculated for 9 IMFs and 13 metalicities. 


Spectral Synthesis Outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^


-----------------------------

.. seealso:: 
   For dedicated tutorials about specific aspecs of hoki and BPASS, check our Cook Book section in the side bar!


