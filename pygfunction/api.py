from pygfunction.gfunction import gFunction
from pygfunction.boreholes import Borehole

from typing import Union, List, Tuple, Dict, Iterable

FloatOrList = Union[float, List[float]]


class BoreholeFieldParameters:
    """This class represents the borehole field and can generate inputs for pygfunction"""

    def __init__(self):
        self.height_list: List[float] = []  # TODO: Document the units
        self.depth_list: List[float] = []
        self.bh_radius_list: List[float] = []
        self.tilt_angle_list: List[float] = []  # TODO: Is this degrees? And I assume from vertical?
        self.orientation_angle_list: List[float] = []  # TODO: Same question
        self.x_coords: List[float] = []  # TODO: What are the units here, I'm assuming meters
        self.y_coords: List[float] = []  # TODO: Same question

    def initialize_borehole_field_generic(
            self, xy_coord_pairs: List[Tuple[float, float]],
            height: FloatOrList, depth: FloatOrList,
            borehole_radius: FloatOrList, tilt_angle: FloatOrList = 0,
            orientation_angle: FloatOrList = 0):
        """
        Generates a set of borehole field parameters from lists of values _or_ scalar values
        Any scalar arguments are extended to the size of the coordinates list to cover each borehole.
        List arguments must be sized to match the size of the coordinate list.

        Parameters
        ----------
        xy_coord_pairs
        height
        depth
        borehole_radius
        tilt_angle
        orientation_angle

        Returns
        -------

        """
        self.x_coords = [xy_pair[0] for xy_pair in xy_coord_pairs]
        self.y_coords = [xy_pair[1] for xy_pair in xy_coord_pairs]
        num_bh = len(xy_coord_pairs)
        self.height_list = self.setup_list(height, num_bh)
        self.depth_list = self.setup_list(depth, num_bh)
        self.bh_radius_list = self.setup_list(borehole_radius, num_bh)
        self.tilt_angle_list = self.setup_list(tilt_angle, num_bh)
        self.orientation_angle_list = self.setup_list(orientation_angle, num_bh)

    @staticmethod
    def setup_list(check_data: FloatOrList, expected_size: int):
        """
        If the check_data argument is a float, this simply returns a list of expected size, filled with that value.
        If the check_data argument is a list, this validates the size of the list and returns it.

        Parameters
        ----------
        check_data
        expected_size

        Returns
        -------

        """
        if isinstance(check_data, list):
            if len(check_data) != expected_size:
                raise ValueError(f"Expected list of length {expected_size}, but got {len(check_data)}.")
            return check_data
        return [check_data] * expected_size

    def as_config(self):
        all_bh_configs = zip(self.height_list,
                             self.depth_list,
                             self.bh_radius_list,
                             self.x_coords,
                             self.y_coords,
                             self.tilt_angle_list,
                             self.orientation_angle_list)
        return [Borehole(*cfg) for cfg in all_bh_configs]


class GFunctionGenerator(object):

    def __init__(self, borehole_field: BoreholeFieldParameters,
                 alpha: float,
                 time: List[float],
                 options: Union[Dict[str, str], None] = None,
                 solver_method: str = "equivalent",
                 boundary_condition: str = "UHTR"):
        """
        Borehole config parameters are defined in the Borehole class
        All other parameters are defined in the PyGFunction class

        Parameters
        ----------
        borehole_field
        alpha
        time
        options
        solver_method
        boundary_condition
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
