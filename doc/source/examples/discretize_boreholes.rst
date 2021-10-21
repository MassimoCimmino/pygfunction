.. examples:

********************************************************************
Calculation of g-Functions computed with non-uniform segment lengths
********************************************************************

This example demonstrates the use of the :func:`utilities <../modules/utilities>`
module to determine discretized segment ratios along a borehole.

The following script computes g-Functions for a field of 6x4 boreholes
utilizing the MIFT and UBWT boundary conditions with a 48 segments per borehole
and equal segment lengths. The MIFT and UBWT g-functions are computed with only
8 segments per borehole and non-uniform segment lengths. RMSE values are
compared. It is shown that g-functions can be calculated accurately using a
small number of segments.

The script is located in:
`pygfunction/examples/discretize_boreholes.py`

.. literalinclude:: ../../../pygfunction/examples/discretize_boreholes.py
   :language: python
   :linenos: