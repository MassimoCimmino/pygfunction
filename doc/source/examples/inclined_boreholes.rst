.. examples:

**************************************************
Calculation of g-functions with inclined boreholes
**************************************************

This example demonstrates the use of the :doc:`g-function <../modules/gfunction>` module
to calculate *g*-functions of fields of inclined boreholes using a boundary condition of
uniform and equal borehole wall temperature for all boreholes. The total rate of heat
extraction in the bore field is constant.

The following script generates the *g*-functions of two bore fields. The first field
corresponds to the optimal configuration presented by Claesson and Eskilson [1]_. The
second field corresponds to the configuration comprised of 8 boreholes in a circle
presented by the same authors.

The script is located in:
`pygfunction/examples/inclined_boreholes.py`

.. literalinclude:: ../../../examples/inclined_boreholes.py
   :language: python
   :linenos:

.. rubric:: References
.. [1] Claesson J, and Eskilson, P. (1987). Conductive heat extraction by
   thermally interacting deep boreholes, in "Thermal analysis of heat
   extraction boreholes". Ph.D. Thesis, University of Lund, Lund, Sweden.
