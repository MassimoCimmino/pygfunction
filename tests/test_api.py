import numpy as np

from unittest import TestCase
from pygfunction.api import BoreholeField


class TestAPI(TestCase):

    def setUp(self):
        self.height = 192
        self.depth = 2
        self.bh_radius = 0.08

        # g-function config
        self.alpha = 1e-6  # Ground thermal diffusivity [m2/s]
        ts = self.height ** 2 / (9 * self.alpha)

        # test values from:
        # Spitler et al. 2021, "G-Function Library for Modeling Vertical Bore Ground Heat Exchanger"
        # https://gdr.openei.org/submissions/1325
        # L-shaped, 3x3, B=5, H=192, r_b=0.08
        self.time = np.exp(
            [-8.5, -7.8, -7.2, -6.5, -5.9, -5.2, -4.5, -3.963, -3.27, -2.864,
             -2.577, -2.171, -1.884, -1.191, -0.497, -0.274, -0.051, 0.196,
             0.419, 0.642, 0.873, 1.112, 1.335, 1.679, 2.028, 2.275, 3.003]) * ts

        self.expected_g_values = [
            2.835159810088887, 3.186553817896705, 3.516527838262404, 4.004329837473649,
            4.563116154725879, 5.422730649791384, 6.515803491910567, 7.482956710803077,
            8.821023313877978, 9.6182191663194, 10.175664703095364, 10.941363073004217,
            11.461472146417226, 12.608257361815083, 13.568342463717922, 13.830179312701365,
            14.065319930740815, 14.294348922588275, 14.472150360525662, 14.62411713920069,
            14.755089916843, 14.86479421584671, 14.946616058533072, 15.040137216786091,
            15.103960182529738, 15.135243749367413, 15.185795927543893
        ]

        self.deg_to_rad = np.pi / 180

    def test_compute_vertical_g_functions_from_api(self):
        def run_test(xy):
            # compute g-functions
            bh_field = BoreholeField()
            bh_field.initialize_borehole_field_generic(xy, self.height, self.depth, self.bh_radius)
            g_vals = bh_field.get_g_functions(self.alpha, self.time, boundary_condition="UBWT")

            # tolerance values are not as tight as one might expect
            for idx, test_val in enumerate(g_vals):
                self.assertAlmostEqual(test_val, self.expected_g_values[idx], delta=2e-1)

        # bh locations
        xy_coords = [(0, 0), (5, 0), (10, 0), (0, 10), (0, 5)]

        # run test with list of tuples
        run_test(xy_coords)

        # run test with list of lists
        run_test([list(z) for z in xy_coords])

    def test_compute_inclined_g_functions_from_api(self):
        # borehole config
        self.height = 150.  # Borehole length [m]
        self.depth = 4.  # Borehole buried self.depth [m]
        self.bh_radius = 0.075  # Borehole radius [m]

        # bh locations
        xy_coords = [(0, 5), (0, 10), (0, 15), (0, 20)]

        # compute g-functions
        bh_field = BoreholeField()
        bh_field.initialize_borehole_field_generic(
            xy_coords, self.height, self.depth, self.bh_radius,
            tilt_angle=20 * self.deg_to_rad, orientation_angle=20 * self.deg_to_rad
        )
        g = bh_field.get_g_functions(self.alpha, self.time, solver_method="detailed")

        # we don't have any other reference to compare these to currently
        self.assertIsInstance(g.tolist(), list)
