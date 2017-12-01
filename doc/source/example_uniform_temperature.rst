.. examples:

*****************************************************************
Calculation of g-functions with uniform borehole wall temperature
*****************************************************************

This example demonstrates the use of the :doc:`g-function <gfunction>` module
to calculate *g*-functions using a boundary condition of uniform and equal 
borehole wall temperature for all boreholes. The total rate of heat extraction
in the bore field is constant.

The following script generates the *g*-functions of rectangular fields of
3 x 2, 6 x 4 and 10 x 10 boreholes. *g*-Functions are verified against the
*g*-functions presented by Cimmino and Bernier [1]_.

The script is located in: 
`pygfunction/examples/uniform_temperature.py`

.. literalinclude:: ../../pygfunction/examples/uniform_temperature.py
   :language: python
   :linenos:

.. rubric:: References
.. [1] Cimmino, M., & Bernier, M. (2014). A
   semi-analytical method to generate g-functions for geothermal bore
   fields. International Journal of Heat and Mass Transfer, 70, 641-650.