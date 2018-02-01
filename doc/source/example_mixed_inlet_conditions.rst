.. examples:

*********************************************************************
Calculation of g-functions with mixed parallel and series connections
*********************************************************************

This example demonstrates the use of the :doc:`g-function <gfunction>` module
and the :doc:`pipes <pipes>` module to calculate *g*-functions considering the
piping connections between the boreholes, based on the method of Cimmino [1]_.
For boreholes connected in series, it is considered the outlet fluid temperature
of the upstream borehole is equal to the inlet fluid temperature of the
downstream borehole. The total rate of heat extraction in the bore field is
constant.

The following script generates the *g*-functions of a field of 5 equally spaced
borehole on a straight line and connected in series. The boreholes have
different lengths. The *g*-function considering piping conections is compared to
the *g*-function obtained using a boundary condition of uniform borehole wall
temperature.

The script is located in: 
`pygfunction/examples/mixed_inlet_conditions.py`

.. literalinclude:: ../../pygfunction/examples/mixed_inlet_conditions.py
   :language: python
   :linenos:

.. rubric:: References
.. [1] Cimmino, M. (2018). g-Functions for bore fields with mixed parallel and
   series connections considering the axial fluid temperature variations.
   IGSHPA Research Track, Stockholm. In review.
