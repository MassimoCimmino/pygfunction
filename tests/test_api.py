import numpy as np

from unittest import TestCase
from pygfunction.api import GFunctionGenerator, BHFieldParams


class TestAPI(TestCase):

    def setUp(self):
        self.height = 150
        self.depth = 4
        self.bh_radius = 0.075

        # g-function config
        self.alpha = 1e-6  # Ground thermal diffusivity [m2/s]
        ts = self.height ** 2 / (9 * self.alpha)
        self.time = np.array([0.1, 1., 10.]) * ts

    def test_compute_vertical_g_functions_from_api(self):
        # bh locations
        xy_coords = [(0, 5), (0, 10), (0, 15), (0, 20)]

        # compute g-functions
        bh_field_params = BHFieldParams(xy_coords, self.height, self.depth, self.bh_radius)
        GFunctionGenerator(bh_field_params, self.alpha, self.time)

    def test_compute_inclined_g_functions_from_api(self):
        # borehole config
        self.height = 150.  # Borehole length [m]
        self.depth = 4.  # Borehole buried self.depth [m]
        self.bh_radius = 0.075  # Borehole radius [m]

        # bh locations
        xy_coords = [(0, 5), (0, 10), (0, 15), (0, 20)]

        # compute g-functions
        bh_field_params = BHFieldParams(xy_coords, self.height, self.depth, self.bh_radius, tilt_angle=20,
                                        orientation_angle=20)
        GFunctionGenerator(bh_field_params, self.alpha, self.time, solver_method="detailed")

    # # Set up initial borehole config values
    # complete_borehole_config = [self.height, self.depth, self.bh_radius, x, y, tilt, orientation]
    # # print(f"{complete_borehole_config=}=")
    # simple_borehole_config = [self.height, self.depth, self.bh_radius, x, y]
    # # print(f"{simple_borehole_config=}")
    #
    # number_of_boreholes = 6
    # # only change x, y coordinates between boreholes
    # increment_indices = [3, 4]
    # # Generate list of lists of borehole configs. Increment each value by 1 to avoid duplicates
    # complete_borefield_config = [
    #     [x + i if idx in increment_indices else x for idx, x in enumerate(complete_borehole_config)] for i in
    #     range(number_of_boreholes)]
    # # pprint(f"{complete_borefield_config=}=")
    # simple_borefield_config = [
    #     [x + i if idx in increment_indices else x for idx, x in enumerate(simple_borehole_config)] for i in
    #     range(number_of_boreholes)]
    # # pprint(f"{simple_borefield_config=}=")
    #
    # # -- Act
    # # TODO: submit various options and test the output
    # pyg_1 = GFunctionGenerator(complete_borefield_config, alpha, time, solver_method="detailed")
    # pyg_2 = GFunctionGenerator(simple_borefield_config, alpha, time)

    # -- Assert
    # assert isinstance(pyg_1.to_list(), list)
    # assert isinstance(pyg_2.to_list(), list)
