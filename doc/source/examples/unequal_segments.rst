.. examples:

***********************************************************
Calculation of g-functions with unequal numbers of segments
***********************************************************

This example demonstrates the use of the :doc:`g-function <../modules/gfunction>` module
to calculate *g*-functions using a boundary condition of uniform and equal 
borehole wall temperature for all boreholes. The total rate of heat extraction
in the bore field is constant. The discretization along three of the boreholes is refined
for the calculation of the *g*-function and to draw the heat extraction rate profiles
along their lengths.

The following script generates the *g*-functions of a rectangular field of
6 x 4. *g*-Functions using equal and unequal numbers of segments are compared.

The script is located in: 
`pygfunction/examples/unequal_segments.py`

.. literalinclude:: ../../../examples/unequal_segments.py
   :language: python
   :linenos:
