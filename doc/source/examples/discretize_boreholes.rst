.. examples:

***********************************************************
g-Functions computed with discretized segment lengths
***********************************************************

This example demonstrates the use of the :func:`utilities <../modules/utilities>`
module to determine discretized segment ratios along a borehole.

The following script computes g-Functions for a field of 6x4 boreholes utilizing
the MIFT and UT boundary conditions with a 24 segments per borehole and equal
segment lengths. The MIFT and UT g-functions are computed with less segments
per borehole, making use of a discretize function. RMSE values are compared.

The script is located in:
`pygfunction/examples/discretize_boreholes.py`

.. literalinclude:: ../../../pygfunction/examples/discretize_boreholes.py
   :language: python
   :linenos: