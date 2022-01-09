.. examples:

*****************************************************************
Simulation of fluid temperatures in a field of multiple boreholes
*****************************************************************

This example demonstrates the use of the
:doc:`networks <../modules/networks>` module to predict the fluid temperature variations
in a bore field with known heat extraction rates.

The g-function of a bore field is first calculated using the equal inlet fluid
temperature boundary condition [1]_. Then, the borehole wall temperature
variations are calculated using the load aggregation scheme of Claesson and
Javed [2]_. The time-variation of heat extraction rates is given by the
synthetic load profile of Bernier et al. [3]_. Predicted inlet and outlet fluid
temperatures of double U-tube boreholes are calculated using the model of
Cimmino [4]_.

The script is located in: 
`pygfunction/examples/fluid_temperature.py`

.. literalinclude:: ../../../examples/fluid_temperature_multiple_boreholes.py
   :language: python
   :linenos:

.. rubric:: References
.. [1] Cimmino, M. (2015). The effects of borehole thermal
   resistances and fluid flow rate on the g-functions of geothermal bore
   fields. International Journal of Heat and Mass Transfer, 91, 1119-1127.
.. [2] Claesson, J., & Javed, S. (2011). A load-aggregation method to calculate
   extraction temperatures of borehole heat exchangers. ASHRAE Transactions,
   118 (1): 530–539.
.. [3] Bernier, M., Pinel, P., Labib, R. and Paillot, R. (2004). A multiple load
   aggregation algorithm for annual hourly simulations of GCHP systems. HVAC&R
   Research 10 (4): 471–487.
.. [4] Cimmino, M. (2016). Fluid and borehole wall temperature profiles in
   vertical geothermal boreholes with multiple U-tubes. Renewable Energy 96 :
   137-147.
