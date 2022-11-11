.. examples:

**********************************************
Simulation of fluid temperatures in a borehole
**********************************************

This example demonstrates the use of the
:doc:`pipes <../modules/pipes>` module to predict the fluid temperature variations
in a borehole with known heat extraction rates.

The g-function of a single borehole is first calculated. Then, the borehole wall
temperature variations are calculated using the load aggregation scheme of
Claesson and Javed [1]_. The time-variation of heat extraction rates is given by
the synthetic load profile of Bernier et al. [2]_. Three pipe configurations are
compared: (1) a single U-tube, using the model of Eskilson and Claesson
[3]_, (2) a double U-tube in parallel, using the model of Cimmino [4]_, and (3)
a double U-tube in series, using the model of Cimmino [4]_.

The script is located in:
`pygfunction/examples/fluid_temperature.py`

.. literalinclude:: ../../../examples/fluid_temperature.py
   :language: python
   :linenos:

.. rubric:: References
.. [1] Claesson, J., & Javed, S. (2011). A load-aggregation method to calculate
   extraction temperatures of borehole heat exchangers. ASHRAE Transactions,
   118 (1): 530–539.
.. [2] Bernier, M., Pinel, P., Labib, R. and Paillot, R. (2004). A multiple load
   aggregation algorithm for annual hourly simulations of GCHP systems. HVAC&R
   Research 10 (4): 471–487.
.. [3] Eskilson, P., & Claesson, J. (1988). Simulation model for thermally
   interacting heat extraction boreholes. Numerical Heat Transfer 13 : 149-165.
.. [4] Cimmino, M. (2016). Fluid and borehole wall temperature profiles in
   vertical geothermal boreholes with multiple U-tubes. Renewable Energy 96 :
   137-147.
