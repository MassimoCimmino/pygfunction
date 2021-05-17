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
