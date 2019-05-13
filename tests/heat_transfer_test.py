# -*- coding: utf-8 -*-
""" Test suite for heat_transfer module.
"""
from __future__ import absolute_import, division, print_function

import unittest

import numpy as np
from scipy.integrate import dblquad
from scipy.special import erfc


class TestFiniteLineSource(unittest.TestCase):
    """ Test cases for finite_line_source function.
    """

    def setUp(self):
        self.t = 1. * 8760. * 3600.     # Time is 1 year
        self.alpha = 1.0e-6             # Thermal diffusivity
        self.D1 = 4.0                   # Buried depth of source
        self.D2 = 16.0                  # Buried depth of target
        self.H1 = 10.0                  # Length of source
        self.H2 = 7.0                   # Length of target
        self.dis = 12.0                 # Distance of target

    def test_finite_line_source(self, rel_tol=1.0e-6):
        """ Tests the value of the FLS solution.
        """
        from pygfunction.boreholes import Borehole
        from pygfunction.heat_transfer import finite_line_source
        # Evaluate the double integral
        reference = dblquad(fls_double,
                            self.D1, self.D1+self.H1,
                            lambda x: self.D2, lambda x: self.D2+self.H2,
                            args=(self.t, self.dis, self.alpha))[0]/self.H2
        # Evaluate using heat_transfer.finite_line_source
        borehole1 = Borehole(self.H1, self.D1, 0.05, 0., 0.)
        borehole2 = Borehole(self.H2, self.D2, 0.05, self.dis, 0.)
        calculated = finite_line_source(self.t, self.alpha,
                                        borehole1, borehole2)
        self.assertAlmostEqual(calculated, reference,
                               delta=rel_tol*reference,
                               msg='Incorrect value of finite line source '
                                   'solution.')

    def test_finite_line_source_real_part(self, rel_tol=1.0e-6):
        """ Tests the value of the real part of the FLS solution.
        """
        from pygfunction.boreholes import Borehole
        from pygfunction.heat_transfer import finite_line_source
        # Evaluate the double integral
        reference = dblquad(fls_double,
                            self.D1, self.D1+self.H1,
                            lambda x: self.D2, lambda x: self.D2+self.H2,
                            args=(self.t,
                                  self.dis,
                                  self.alpha,
                                  True,
                                  False))[0]/self.H2
        # Evaluate using heat_transfer.finite_line_source
        borehole1 = Borehole(self.H1, self.D1, 0.05, 0., 0.)
        borehole2 = Borehole(self.H2, self.D2, 0.05, self.dis, 0.)
        calculated = finite_line_source(self.t, self.alpha,
                                        borehole1, borehole2,
                                        reaSource=True, imgSource=False)
        self.assertAlmostEqual(calculated, reference,
                               delta=rel_tol*reference,
                               msg='Incorrect value of the real part of the '
                                   'finite line source solution.')

    def test_finite_line_source_image_part(self, rel_tol=1.0e-6):
        """ Tests the value of the image part of the FLS solution.
        """
        from pygfunction.boreholes import Borehole
        from pygfunction.heat_transfer import finite_line_source
        # Evaluate the double integral
        reference = dblquad(fls_double,
                            self.D1, self.D1+self.H1,
                            lambda x: self.D2, lambda x: self.D2+self.H2,
                            args=(self.t,
                                  self.dis,
                                  self.alpha,
                                  False,
                                  True))[0]/self.H2
        # Evaluate using heat_transfer.finite_line_source
        borehole1 = Borehole(self.H1, self.D1, 0.05, 0., 0.)
        borehole2 = Borehole(self.H2, self.D2, 0.05, self.dis, 0.)
        calculated = finite_line_source(self.t, self.alpha,
                                        borehole1, borehole2,
                                        reaSource=False, imgSource=True)
        self.assertAlmostEqual(calculated, reference,
                               delta=np.abs(rel_tol*reference),
                               msg='Incorrect value of the image part of the '
                                   'finite line source solution.')

    def test_finite_line_source_no_part(self, rel_tol=1.0e-6):
        """ Tests the value of the FLS solution when considering no source.
        """
        from pygfunction.boreholes import Borehole
        from pygfunction.heat_transfer import finite_line_source
        # Evaluate using heat_transfer.finite_line_source
        borehole1 = Borehole(self.H1, self.D1, 0.05, 0., 0.)
        borehole2 = Borehole(self.H2, self.D2, 0.05, self.dis, 0.)
        calculated = finite_line_source(self.t, self.alpha,
                                        borehole1, borehole2,
                                        reaSource=False, imgSource=False)
        self.assertEqual(calculated, 0.,
                               msg='Incorrect value of no part of the '
                                   'finite line source solution.')


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
        from pygfunction.heat_transfer import thermal_response_factors
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
        from pygfunction.heat_transfer import thermal_response_factors
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
        from pygfunction.heat_transfer import thermal_response_factors
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


def fls_double(z2, z1, t, dis, alpha, reaSource=True, imgSource=True):
    """ FLS expression for double integral solution.
    """
    r_pos = np.sqrt(dis**2 + (z2 - z1)**2)
    r_neg = np.sqrt(dis**2 + (z2 + z1)**2)
    fls = 0.
    if reaSource:
        fls += 0.5*erfc(r_pos/np.sqrt(4*alpha*t))/r_pos
    if imgSource:
        fls += -0.5*erfc(r_neg/np.sqrt(4*alpha*t))/r_neg
    return fls


if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    unittest.main()
