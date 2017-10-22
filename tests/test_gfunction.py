# -*- coding: utf-8 -*-
""" Test suite for gfunction module.
"""
from __future__ import division, print_function, absolute_import

import unittest

import numpy as np


class TestUniformHeatExtractionRate(unittest.TestCase):
    """ Test cases for calculation of g-functions using uniform heat extraction
        rate boundary condition.
    """

    def setUp(self):
        self.H = 150.           # Borehole length [m]
        self.D = 4.             # Borehole buried depth [m]
        self.r_b = 0.075        # Borehole radius [m]
        self.B = 7.5            # Borehole spacing [m]
        self.alpha = 1.0e-6     # Ground thermal diffusivity [m2/s]

    def test_one_borehole(self, rel_tol=1.0e-4):
        """ Tests the value of the g-function of one borehole.
        """
        from pygfunction.gfunction import uniform_heat_extraction
        from pygfunction.boreholes import rectangle_field

        # Results of Cimmino and Bernier (2014)
        time = np.array([1070, 131118, 33554478])*3600.
        g_ref = np.array([3.65501492640735,
                          5.87704770495107,
                          6.68675948788837])

        # Calculation of the g-function at the same time values
        N_1 = 1
        N_2 = 1
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)
        g = uniform_heat_extraction(boreField, time, self.alpha)

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-10),
                        msg='Incorrect values of the g-function of one '
                            'borehole for uniform heat extraction rate.')

    def test_three_by_two(self, rel_tol=1.0e-4):
        """ Tests the value of the g-function of a 3 by 2 bore field.
        """
        from pygfunction.gfunction import uniform_heat_extraction
        from pygfunction.boreholes import rectangle_field

        # Results of Cimmino and Bernier (2014)
        time = np.array([1070, 131118, 33554478])*3600.
        g_ref = np.array([3.66173047944222,
                          11.2627341238639,
                          16.0243474955568])

        # Calculation of the g-function at the same time values
        N_1 = 3
        N_2 = 2
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)
        g = uniform_heat_extraction(boreField, time, self.alpha)

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-10),
                        msg='Incorrect values of the g-function of three by '
                            'two field for uniform heat extraction rate.')


class TestUniformTemperature(unittest.TestCase):
    """ Test cases for calculation of g-functions using uniform borehole wall
        temperature boundary condition.
    """

    def setUp(self):
        self.H = 150.           # Borehole length [m]
        self.D = 4.             # Borehole buried depth [m]
        self.r_b = 0.075        # Borehole radius [m]
        self.B = 7.5            # Borehole spacing [m]
        self.alpha = 1.0e-6     # Ground thermal diffusivity [m2/s]

    def test_one_borehole_one_segment(self, rel_tol=1.0e-3):
        """ Tests the value of the g-function of one borehole.
        """
        from pygfunction.gfunction import uniform_temperature
        from pygfunction.boreholes import rectangle_field

        # Results of Cimmino and Bernier (2014)
        time = np.array([1070, 131118, 33554478])*3600.
        g_ref = np.array([3.65502692098609,
                          5.87704904232202,
                          6.68675948788828])

        # Calculation of the g-function at the same time values
        N_1 = 1
        N_2 = 1
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)
        g = uniform_temperature(boreField, time, self.alpha, nSegments=1)

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-10),
                        msg='Incorrect values of the g-function of one '
                            'borehole for uniform temperature (1 segment).')

    def test_three_by_two_one_segment(self, rel_tol=1.0e-3):
        """ Tests the value of the g-function of a 3 by 2 bore field.
        """
        from pygfunction.gfunction import uniform_temperature
        from pygfunction.boreholes import rectangle_field

        # Results of Cimmino and Bernier (2014)
        time = np.array([1070, 131118, 33554478])*3600.
        g_ref = np.array([3.6617432932388,
                          11.2153482087534,
                          15.9710298219412])

        # Calculation of the g-function at the same time values
        N_1 = 3
        N_2 = 2
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)
        g = uniform_temperature(boreField, time, self.alpha, nSegments=1)

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-10),
                        msg='Incorrect values of the g-function of three by '
                            'two field for uniform temperature (1 segment).')

    def test_one_borehole_twelve_segments(self, rel_tol=1.0e-3):
        """ Tests the value of the g-function of one borehole.
        """
        from pygfunction.gfunction import uniform_temperature
        from pygfunction.boreholes import rectangle_field

        # Results of Cimmino and Bernier (2014)
        time = np.array([1070, 131118, 33554478])*3600.
        g_ref = np.array([3.65476902065229,
                          5.85967970721258,
                          6.6329923757352])

        # Calculation of the g-function at the same time values
        N_1 = 1
        N_2 = 1
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)
        g = uniform_temperature(boreField, time, self.alpha, nSegments=12)

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-10),
                        msg='Incorrect values of the g-function of one '
                            'borehole for uniform temperature (12 segments).')

    def test_three_by_two_twelve_segments(self, rel_tol=1.0e-3):
        """ Tests the value of the g-function of a 3 by 2 bore field.
        """
        from pygfunction.gfunction import uniform_temperature
        from pygfunction.boreholes import rectangle_field

        # Results of Cimmino and Bernier (2014)
        time = np.array([1070, 131118, 33554478])*3600.
        g_ref = np.array([3.66148345464598,
                          11.0406206357965,
                          15.1697321426028])

        # Calculation of the g-function at the same time values
        N_1 = 3
        N_2 = 2
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)
        g = uniform_temperature(boreField, time, self.alpha, nSegments=12)

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-10),
                        msg='Incorrect values of the g-function of three by '
                            'two field for uniform temperature (12 segments).')
        
        
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    unittest.main()
