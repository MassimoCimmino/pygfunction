.. examples:

*******************************************************************************************************
Simulation of fluid temperatures in a field of series-connected boreholes and reversible flow direction
*******************************************************************************************************

This example demonstrates the use of the
:doc:`networks <../modules/networks>` module to predict the fluid temperature variations
in a bore field with series-connected boreholes and reversible flow direction.

The variable fluid mass flow rates g-functions of a bore field are first calculated
using the mixed inlet fluid temperature boundary condition [1]_. Then, the effective borehole
wall temperature variations are calculated using the load aggregation scheme of Claesson and
Javed [2]_. g-Functions used in the temporal superposition are interpolated with regards to
the fluid mass flow rate at the moment of heat extraction.

The script is located in:
`pygfunction/examples/fluid_temperature_reversible_flow_direction.py`

.. literalinclude:: ../../../examples/fluid_temperature_reversible_flow_direction.py
   :language: python
   :linenos:

.. rubric:: References
.. [1] Cimmino, M. (2024). g-Functions for fields of
   series- and parallel-connected boreholes with variable fluid mass flow
   rate and reversible flow direction. Renewable Energy 228: 120661.
.. [2] Claesson, J., & Javed, S. (2011). A load-aggregation method to calculate
   extraction temperatures of borehole heat exchangers. ASHRAE Transactions,
   118 (1): 530–539.
