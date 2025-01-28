.. examples:

***********************************************
Simulation of a borehole using load aggregation
***********************************************

This example demonstrates the use of the
:doc:`load aggregation <load_aggregation>` module to predict the borehole
wall temperature of a single temperature with known heat extraction rates.

The g-function of a single borehole is first calculated. Then, the borehole wall
temperature variations are calculated using the load aggregation scheme of
Claesson and Javed [1]_. The time-variation of heat extraction rates is given by
the synthetic load profile of Bernier et al. [2]_.

The following script validates the load aggregation scheme with the exact
solution obtained from convolution in the Fourier domain (see ref. [3]_).

The script is located in:
`pygfunction/examples/load_aggregation.py`

.. literalinclude:: ../../../examples/load_aggregation.py
   :language: python
   :linenos:

.. rubric:: References
.. [1] Claesson, J., & Javed, S. (2011). A load-aggregation method to calculate
   extraction temperatures of borehole heat exchangers. ASHRAE Transactions,
   118 (1): 530–539.
.. [2] Bernier, M., Pinel, P., Labib, R. and Paillot, R. (2004). A multiple load
   aggregation algorithm for annual hourly simulations of GCHP systems. HVAC&R
   Research 10 (4): 471–487.
.. [3] Marcotte, D., & Pasquier, P. (2008). Fast fluid and ground temperature
   computation for geothermal ground-loop heat exchanger systems. Geothermics,
   37 (6) : 651-665.
