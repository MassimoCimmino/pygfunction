.. examples:

******************************************************
Calculation of effective bore field thermal resistance
******************************************************

This example demonstrates the use of the
:py:func:`.pipes.field_thermal_resistance` function to evaluate the bore field
thermal resistance. The concept effective bore field thermal is detailed by
Cimmino [1]_

The following script evaluates the effective bore field thermal resistance
for fields of 1 to 5 series-connected boreholes with fluid flow rates ranging
from 0.01 kg/s ti 1.00 kg/s.

The script is located in:
`pygfunction/examples/bore_field_thermal_resistance.py`

.. literalinclude:: ../../../examples/bore_field_thermal_resistance.py
   :language: python
   :linenos:

.. rubric:: References
.. [1] Cimmino, M. (2018). g-Functions for bore fields with
   mixed parallel and series connections considering the axial fluid
   temperature variations. Proceedings of the IGSHPA Sweden Research Track
   2018. Stockholm, Sweden. pp. 262-270.
