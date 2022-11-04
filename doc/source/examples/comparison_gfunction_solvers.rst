.. examples:

**************************************************************
Compare the accuracy and speed of different g-function solvers
**************************************************************

This example compares the simulation times and the accuracy of different
solvers for the evaluation of g-functions.

The g-function of a field of 6 by4 boreholes is first calculated for a boundary
condition of uniform borehole wall temperature along the boreholes, equal
for all boreholes. Three different solvers are compared : 'detailed',
'similarities' [1]_ and 'equivalent' [2]_. Their accuracy and calculation time
are compared using the 'detailed' solver as a reference. This shows that the
'similarities' solver can evaluate g-functions with high accuracy.

The g-function of a field of 12 by 10 boreholes is then calculated for a boundary
condition of uniform borehole wall temperature along the boreholes, equal
for all boreholes. Two different solvers are compared : 'similarities' and
'equivalent'. The accuracy and calculation time of the 'equivalent' is
compared using the 'similarities' solver as a reference. This shows that
the 'equivalent' solver evaluates g-functions at a very high calculation
speed while maintaining reasonable accuracy.

The script is located in:
`pygfunction/examples/comparison_gfunction_solvers.py`

.. literalinclude:: ../../../examples/comparison_gfunction_solvers.py
   :language: python
   :linenos:

.. rubric:: References
.. [1] Cimmino, M. (2018). Fast calculation of the
   g-functions of geothermal borehole fields using similarities in the
   evaluation of the finite line source solution. Journal of Building
   Performance Simulation, 11 (6), 655-668.
.. [2] Prieto, C., & Cimmino, M.
   (2021). Thermal interactions in large irregular fields of geothermal
   boreholes: the method of equivalent borehole. Journal of Building
   Performance Simulation, 14 (4), 446-460.
