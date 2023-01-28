.. examples:

************************************************************
Evaluation of thermal resistances using the multipole method
************************************************************

This example demonstrates the use of the
:py:func:`.pipes.thermal_resistances` function to
evaluate internal thermal resistances in a borehole. The example also covers the
use of the :py:func:`.pipes.multipole` function to evaluate the
2D temperature field in and around a borehole.

The thermal resistances of a borehole with two pipes are evaluated using the
multipole method of Claesson and Hellstrom [1]_. Based on the calculated
thermal resistances, the heat flows from the pipes required to obtain pipe
temperatures of 1 degC are evaluated. The temperatures in and around the
borehole with 2 pipes are then calculated. Results are verified against the
results of Claesson and Hellstrom [1]_.

The script is located in:
`pygfunction/examples/multipole_temperature.py`

.. literalinclude:: ../../../examples/multipole_temperature.py
   :language: python
   :linenos:

.. rubric:: References
.. [1] Claesson, J., & Hellstrom, G. (2011). Multipole method to calculate
   borehole thermal resistances in a borehole heat exchanger. HVAC&R Research,
   17(6), 895-911.
