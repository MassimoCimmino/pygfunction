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
        time = np.array([1070, 33554478])*3600.
        g_ref = np.array([3.65501492640735,
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
        time = np.array([1070, 33554478])*3600.
        g_ref = np.array([3.66173047944222,
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

    def test_one_borehole_one_segment(self, rel_tol=1.0e-4):
        """ Tests the value of the g-function of one borehole.
        """
        from pygfunction.gfunction import uniform_temperature
        from pygfunction.boreholes import rectangle_field

        # Results of Cimmino and Bernier (2014)
        time = np.array([1070, 33554478])*3600.
        g_ref = np.array([3.65502692098609,
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

    def test_three_by_two_one_segment(self, rel_tol=1.0e-4):
        """ Tests the value of the g-function of a 3 by 2 bore field.
        """
        from pygfunction.gfunction import uniform_temperature
        from pygfunction.boreholes import rectangle_field

        # Results of Cimmino and Bernier (2014)
        time = np.array([1070, 33554478])*3600.
        g_ref = np.array([3.6617432932388,
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

    def test_one_borehole_twelve_segments(self, rel_tol=1.0e-4):
        """ Tests the value of the g-function of one borehole.
        """
        from pygfunction.gfunction import uniform_temperature
        from pygfunction.boreholes import rectangle_field

        # Results of Cimmino and Bernier (2014)
        time = np.array([1070, 33554478])*3600.
        g_ref = np.array([3.65476902065229,
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

    def test_three_by_two_twelve_segments(self, rel_tol=1.0e-4):
        """ Tests the value of the g-function of a 3 by 2 bore field.
        """
        from pygfunction.gfunction import uniform_temperature
        from pygfunction.boreholes import rectangle_field

        # Results of Cimmino and Bernier (2014)
        time = 33554478*3600.
        g_ref = 15.1697321426028

        # Calculation of the g-function at the same time values
        N_1 = 3
        N_2 = 2
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)
        g = uniform_temperature(boreField, time, self.alpha, nSegments=12)

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-10),
                        msg='Incorrect values of the g-function of three by '
                            'two field for uniform temperature (12 segments).')


class TestEqualInletTemperature(unittest.TestCase):
    """ Test cases for calculation of g-functions using equal inlet fluid
        temperature boundary condition.
    """

    def setUp(self):
        self.H = 150.           # Borehole length [m]
        self.D = 4.             # Borehole buried depth [m]
        self.r_b = 0.075        # Borehole radius [m]
        self.B = 7.5            # Borehole spacing [m]
        self.alpha = 1.0e-6     # Ground thermal diffusivity [m2/s]
        self.k_s = 2.0          # Ground thermal conductivity [W/m.K]
        self.m_flow = 0.5       # Fluid mass flow rate (per borehole) [kg/s]
        self.cp = 4000.         # Fluid specific heat capacity [J/kg.K]

    def test_one_borehole_twenty_four_segments(self, rel_tol=1.0e-4):
        """ Tests the value of the g-function of one borehole.
        """
        from pygfunction.pipes import SingleUTube
        from pygfunction.gfunction import equal_inlet_temperature
        from pygfunction.boreholes import rectangle_field

        # Pipe characteristics (will be overwritten by thermal resistances)
        r_out = 0.01
        r_in = 0.0075
        Ds = 0.06
        pos = [(-Ds, 0.), (Ds, 0.)]
        k_g = 1.5

        # Results of Cimmino (2015)
        # Number of line source segments per borehole
        nSegments = 24
        # Very large time values to approximate steady-state
        time = 1.0e20
        # Borehole thermal resistances [m.K/W]
        Rb = np.array([0.1, 0.3])
        # Borehole short-circuit resistance [m.K/W]
        Ra = 0.25
        g_ref = np.array([6.63770426518980,
                          6.65263629558422])

        # Calculation of the g-function at the same time values
        N_1 = 1
        N_2 = 1
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)
        g = np.zeros_like(g_ref)
        for i in range(len(Rb)):
            UTubes = [SingleUTube(
                    pos, r_in, r_out, borehole, self.k_s, k_g, 0., J=0)
                      for borehole in boreField]
            # Overwrite borehole resistances
            Rd_00 = 2*Rb[i]
            Rd_01 = 2*Ra*Rd_00/(2*Rd_00 - Ra)
            for UTube in UTubes:
                UTube._Rd[0,0] = Rd_00
                UTube._Rd[1,1] = Rd_00
                UTube._Rd[0,1] = Rd_01
                UTube._Rd[1,0] = Rd_01
                UTube._update_model_variables(self.m_flow, self.cp, nSegments)
            g[i] = equal_inlet_temperature(boreField, UTubes, self.m_flow,
                                           self.cp, time, self.alpha,
                                           nSegments=nSegments)

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-10),
                        msg='Incorrect values of the g-function of one '
                            'borehole for equal inlet fluid temperature '
                            '(24 segments).')

    def test_four_by_four_twenty_four_segments(self, rel_tol=1.0e-4):
        """ Tests the value of the g-function of 4 by 4 bore field.
        """
        from pygfunction.pipes import SingleUTube
        from pygfunction.gfunction import equal_inlet_temperature
        from pygfunction.boreholes import rectangle_field

        # Pipe characteristics (will be overwritten by thermal resistances)
        r_out = 0.01
        r_in = 0.0075
        Ds = 0.06
        pos = [(-Ds, 0.), (Ds, 0.)]
        k_g = 1.5

        # Results of Cimmino (2015)
        # Number of line source segments per borehole
        nSegments = 24
        # Very large time values to approximate steady-state
        time = 1.0e20
        # Borehole thermal resistances [m.K/W]
        Rb = 1.0
        # Borehole short-circuit resistance [m.K/W]
        Ra = 0.25
        g_ref = 28.0185515095651

        # Calculation of the g-function at the same time values
        N_1 = 4
        N_2 = 4
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)
        g = np.zeros_like(g_ref)
        UTubes = [SingleUTube(
                pos, r_in, r_out, borehole, self.k_s, k_g, 0., J=0)
                  for borehole in boreField]
        # Overwrite borehole resistances
        Rd_00 = 2*Rb
        Rd_01 = 2*Ra*Rd_00/(2*Rd_00 - Ra)
        for UTube in UTubes:
            UTube._Rd[0,0] = Rd_00
            UTube._Rd[1,1] = Rd_00
            UTube._Rd[0,1] = Rd_01
            UTube._Rd[1,0] = Rd_01
            UTube._update_model_variables(self.m_flow, self.cp, nSegments)
        g = equal_inlet_temperature(boreField, UTubes, self.m_flow,
                                       self.cp, time, self.alpha,
                                       nSegments=nSegments)

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-10),
                        msg='Incorrect values of the g-function of four by '
                            'four field for equal inlet fluid temperature '
                            '(24 segments).')


class TestThermalResponseFactors(unittest.TestCase):
    """ Test cases for the evaluation of segment to segment thermal response
        factors.
    """

    def setUp(self):
        self.H = 150.           # Borehole length [m]
        self.D = 4.             # Borehole buried depth [m]
        self.r_b = 0.075        # Borehole radius [m]
        self.B = 7.5            # Borehole spacing [m]
        self.alpha = 1.0e-6     # Ground thermal diffusivity [m2/s]

    def test_one_borehole_four_segments(self, rel_tol=1.0e-6):
        """ Tests the value of the thermal response factor matrix for one
            borehole with and without similarities.
        """
        from pygfunction.gfunction import thermal_response_factors
        from pygfunction.gfunction import _borehole_segments
        from pygfunction.boreholes import rectangle_field

        N_1 = 1
        N_2 = 1
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)
        boreSegments = _borehole_segments(boreField, nSegments=4)
        time = np.array([33554478])*3600.

        # Calculation of thermal response factor matrix using similarities
        h = thermal_response_factors(boreSegments, time, self.alpha,
                                     use_similarities=True)
        # Calculation of thermal response factor matrix without similarities
        h_none = thermal_response_factors(boreSegments, time, self.alpha,
                                          use_similarities=False)

        self.assertTrue(np.allclose(h, h_none, rtol=rel_tol, atol=1e-10),
                        msg='Incorrect values of the thermal response factors '
                            'for one borehole (4 segments).')

    def test_three_by_two_four_segments(self, rel_tol=1.0e-6):
        """ Tests the value of the thermal response factor matrix for three by
            two field with and without similarities.
        """
        from pygfunction.gfunction import thermal_response_factors
        from pygfunction.gfunction import _borehole_segments
        from pygfunction.boreholes import rectangle_field

        N_1 = 3
        N_2 = 2
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)
        boreSegments = _borehole_segments(boreField, nSegments=4)
        time = np.array([33554478])*3600.

        # Calculation of thermal response factor matrix using similarities
        h = thermal_response_factors(boreSegments, time, self.alpha,
                                     use_similarities=True)
        # Calculation of thermal response factor matrix without similarities
        h_none = thermal_response_factors(boreSegments, time, self.alpha,
                                          use_similarities=False)

        self.assertTrue(np.allclose(h, h_none, rtol=rel_tol, atol=1e-10),
                        msg='Incorrect values of the thermal response factors '
                            'for three by two field (4 segments).')

    def test_two_unequal_boreholes_four_segments(self, rel_tol=1.0e-6):
        """ Tests the value of the thermal response factor matrix for two
            boreholes of unequal lengths with and without similarities.
        """
        from pygfunction.gfunction import thermal_response_factors
        from pygfunction.gfunction import _borehole_segments
        from pygfunction.boreholes import Borehole

        borehole1 = Borehole(self.H, self.D, self.r_b, 0., 0.)
        borehole2 = Borehole(self.H*1.432, self.D, self.r_b, self.B, 0.)
        boreField = [borehole1, borehole2]
        boreSegments = _borehole_segments(boreField, nSegments=4)
        time = np.array([33554478])*3600.

        # Calculation of thermal response factor matrix using similarities
        h = thermal_response_factors(boreSegments, time, self.alpha,
                                     use_similarities=True)
        # Calculation of thermal response factor matrix without similarities
        h_none = thermal_response_factors(boreSegments, time, self.alpha,
                                          use_similarities=False)

        self.assertTrue(np.allclose(h, h_none, rtol=rel_tol, atol=1e-10),
                        msg='Incorrect values of the thermal response factors '
                            'two unequal boreholes (4 segments).')
        
        
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    unittest.main()
