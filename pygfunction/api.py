from pygfunction.gfunction import gFunction
from pygfunction.boreholes import Borehole

from typing import Union, List, Tuple, Dict


class BHFieldParams(object):
    def __init__(self, xy_coord_pairs: Union[List[Tuple[float, float]], List[List[float]]],
                 height: Union[float, List[float]], depth: Union[float, List[float]],
                 borehole_radius: Union[float, List[float]], tilt_angle: Union[float, List[float]] = 0,
                 orientation_angle: Union[float, List[float]] = 0):

        self.xy_coords = xy_coord_pairs
        self.x_coords = [xy_pair[0] for xy_pair in xy_coord_pairs]
        self.y_coords = [xy_pair[1] for xy_pair in xy_coord_pairs]
        self.num_bh = len(xy_coord_pairs)
        self.height = height
        self.height_list = []
        self.depth = depth
        self.depth_list = []
        self.bh_radius = borehole_radius
        self.bh_radius_list = []
        self.tilt_angle = tilt_angle
        self.tilt_angle_list = []
        self.orientation_angle = orientation_angle
        self.orientation_angle_list = []

    def setup_list(self, user_value):
        if type(user_value) == list:
            if len(user_value) != self.num_bh:
                assert False
            return user_value
        else:
            return [user_value] * self.num_bh

    def as_config(self):
        self.height_list = self.setup_list(self.height)
        self.depth_list = self.setup_list(self.depth)
        self.bh_radius_list = self.setup_list(self.bh_radius)
        self.tilt_angle_list = self.setup_list(self.tilt_angle)
        self.orientation_angle_list = self.setup_list(self.orientation_angle)

        all_bh_configs = zip(self.height_list,
                             self.depth_list,
                             self.bh_radius_list,
                             self.x_coords,
                             self.y_coords,
                             self.tilt_angle_list,
                             self.orientation_angle_list)

        return [Borehole(*cfg) for cfg in all_bh_configs]


class GFunctionGenerator(object):

    def __init__(self, borehole_field: BHFieldParams,
                 alpha: float,
                 time: List[float],
                 options: Union[Dict[str, str], None] = None,
                 solver_method: str = "equivalent",
                 boundary_condition: str = "UHTR"):
        """
        Borehole config parameters are defined in the Borehole class
        All other parameters are defined in the PyGFunction class
        """

        if options is None:
            options = {}

        self.gfunc = gFunction(
            borehole_field.as_config(),
            alpha,
            time=time,
            boundary_condition=boundary_condition,
            options=options,
            method=solver_method,
        )

    def to_list(self):
        return self.gfunc.gFunc.tolist()
