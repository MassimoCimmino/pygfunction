.. examples:

**********************************************************
Definition of a bore field using custom borehole positions
**********************************************************

This example demonstrates the use of the :doc:`borehole <../modules/boreholes>` module
to define the positions of the boreholes within a bore field from a list of
borehole positions.

Two borehole positions (1 and 2) are intentionally added as duplicates and
are removed by calling the :func:`pygfunction.boreholes.remove_duplicates`
function.

The following script generates a bore field with 5 boreholes. The field is
then plotted on a figure.

The script is located in:
`pygfunction/examples/custom_bore_field.py`

.. literalinclude:: ../../../examples/custom_bore_field.py
   :language: python
   :linenos:
