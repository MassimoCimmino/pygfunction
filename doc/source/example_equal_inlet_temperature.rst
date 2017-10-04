.. examples:

*************************************************************
Calculation of g-functions with equal inlet fluid temperature
*************************************************************

This example demonstrates the use of the :doc:`g-function <gfunction>` module
and the :doc:`pipes <pipes>` module to calculate *g*-functions using a boundary
condition of equal inlet fluid temperature into all boreholes, based on the
method of Cimmino [1]_. The total rate of heat extraction in the bore field is
constant.

The following script generates the *g*-functions of a rectangular field of
6 x 4 boreholes. The *g*-function using a boundary condition of equal inlet
fluid temperature is compared to the *g*-functions obtained using boundary
conditions of uniform heat extraction rate and of uniform borehole wall
temperature.

The script is located in: 
`./examples/equal_inlet_temperature.py`

.. literalinclude:: ../../examples/equal_inlet_temperature.py
   :language: python
   :linenos:

.. rubric:: References
.. [1] Cimmino, M. (2015). The effects of borehole thermal
   resistances and fluid flow rate on the g-functions of geothermal bore
   fields. International Journal of Heat and Mass Transfer, 91, 1119-1127.
