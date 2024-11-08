from pygfunction.gfunction import gFunction
from pygfunction.boreholes import Borehole
from typing import Dict, Optional, Any, List


class PYG(object):

    def __init__(self, borehole_config_list: list[list[float]],
                 alpha: float,
                 time: list[float],
                 options=None,
                 solver_method="equivalent",
                 boundary_condition="UHTR"):

        """
        Borehole config parameters are defined in the Borehole class
        All other parameters are defined in the PyGFunction class
        """

        borefield = []  # List of borehole objects

        if options is None:
            options = {}

        for borehole_config_items in borehole_config_list:
            _borehole = Borehole(*borehole_config_items)
            borefield.append(_borehole)

        self.gfunc = gFunction(
            borefield,
            alpha,
            time=time,

            boundary_condition=boundary_condition,

            options=options,
            method=solver_method,
        )

    def to_list(self):
        return self.gfunc.gFunc.tolist()
