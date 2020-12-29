.. examples:

***********************************************************************
Compare the accuracy and speed of different load aggregation algorithms
***********************************************************************

This example compares the simulation times and the accuracy of borehole wall
temperature predictions of different load aggregation algorithms implemented
into the :doc:`load agggregation <load_aggregation>` module.

The g-function of a single borehole is first calculated. Then, the borehole wall
temperature variations are calculated using the load aggregation schemes of
Bernier et al. [1]_, Liu [2]_, and Claesson and Javed [3]_,. The time-variation
of heat extraction rates is given by the synthetic load profile of
Bernier et al. [1]_.

The following script validates the load aggregation schemes with the exact
solution obtained from convolution in the Fourier domain (see ref. [4]_).

The script is located in: 
`pygfunction/examples/comparison_load_aggregation.py`

.. literalinclude:: ../../pygfunction/examples/comparison_load_aggregation.py
   :language: python
   :linenos:

.. rubric:: References
.. [1] Bernier, M., Pinel, P., Labib, R. and Paillot, R. (2004). A multiple load
   aggregation algorithm for annual hourly simulations of GCHP systems. HVAC&R
   Research 10 (4): 471–487.
.. [2] Liu, X. (2005). Development and experimental validation
   of simulation of hydronic snow melting systems for bridges. Ph.D.
   Thesis. Oklahoma State University.
.. [3] Claesson, J., & Javed, S. (2011). A load-aggregation method to calculate
   extraction temperatures of borehole heat exchangers. ASHRAE Transactions,
   118 (1): 530–539.
.. [4] Marcotte, D., & Pasquier, P. (2008). Fast fluid and ground temperature
   computation for geothermal ground-loop heat exchanger systems. Geothermics,
   37 (6) : 651-665.
