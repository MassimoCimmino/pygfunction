.. examples:

********************************************************************************
Calculation of fluid temperature profiles in a borehole with independent U-tubes
********************************************************************************

This example demonstrates the use of the :doc:`pipes <../modules/pipes>` module to
calculate the fluid temperature profiles in a borehole with independent U-tubes,
based on the method of Cimmino [1]_. The borehole wall temperature is uniform
in this example.

The following script evaluates the fluid temperatures in a borehole with 4
independent U-tubes with different inlet fluid temperatures and different inlet
fluid mass flow rates. The resulting fluid temperature profiles are verified
against the fluid temeprature profiles presented by Cimmino [1]_.

The script is located in:
`pygfunction/examples/multiple_independent_Utubes.py`

.. literalinclude:: ../../../examples/multiple_independent_Utubes.py
   :language: python
   :linenos:

.. rubric:: References
.. [1] Cimmino, M. (2016). Fluid and borehole wall temperature profiles in
   vertical geothermal boreholes with multiple U-tubes. Renewable Energy 96 :
   137-147.
