import numpy as np
from pygfunction.gfunction import gFunction
from pygfunction.boreholes import Borehole

class PYG(object):
    def __init__(self, borehole_config_list: list[list[float]], alpha: float, time: list[int], options=None, solver_method="equivalent"):
        """
        Borehole config parameters are defined in the Borehole class
        All other parameters are defined in the PyGFunction class
        """

        # default options:
        # disTol=0.01, tol=1.0e-6, dtype=np.double, disp=False,

        # self.H = H
        # self.D = D
        # self.r_b = r_b
        # self.x = x
        # self.y = y
        # self.tilt = tilt
        # self.orientation = orientation

        # Check if borehole is inclined
        # self._is_tilted = np.abs(self.tilt) > 1.0e-6
        # self.alpha = alpha  # Soil thermal diffusivity (in m2/s).
        # alpha = soil.k / soil.rhoCp
        # self.time = time  # Values of time (in seconds) for which the g-function is evaluated.
        # self.solver_method: str = solver_method  # Solver method: "equivalent", "similarities", "detailed"
        # Select the correct solver method:
        # https://github.com/BETSRG/GHEDesigner/blob/main/ghedesigner/gfunction.py#L30
        # if use_similarities:
        #     self.solver_method='similarities'
        # else:
        #     self.solver_method='detailed'
        # self.options: dict = options  # Additional options for the g-function solver
        # if self.options is None:
        #     # Build options dict
        #     options = {
        #         'nSegments': 1,
        #         'segment_ratios': None,
        #         'disTol': 0.01,
        #         'tol': 1.0e-6,
        #         'dtype': np.double,
        #         'disp': False,
        #         'profiles': True
        #     }

        boundary = "UHTR"  # "UHTR" or "UBWT" or "MIFT"

        bore_field = []  # List of borehole objects
        # # bhe_objects = []
        for borehole_config_items in borehole_config_list:
            _borehole = Borehole(*borehole_config_items)
            bore_field.append(_borehole)
        # Initialize pipe model
        # if boundary == "MIFT":
        #     bhe = get_bhe_object(bhe_type, m_flow_borehole, fluid, _borehole, pipe, grout, soil)
        #     bhe_objects.append(bhe)

        # if boundary in ("UHTR", "UBWT"):
        self.gfunc = gFunction(
            bore_field,
            alpha,
            time=time,
            boundary_condition=boundary,
            # options=options,
            method=solver_method,
        )
    def to_list(self):
        return self.gfunc.gFunc.tolist()
