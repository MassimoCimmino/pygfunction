# -*- coding: utf-8 -*-
""" Test suite for boreholes module.
"""
from __future__ import absolute_import, division, print_function

import unittest

import numpy as np
from scipy.constants import pi


class TestBorehole(unittest.TestCase):
    """ Test cases for Borehole() class.
    """

    def setUp(self):
        from pygfunction import boreholes
        self.H = 150.
        self.D = 2.5
        self.r_b = 0.075
        self.tilt = pi/6.
        self.orientation = pi
        self.x = 2.
        self.y = 4.
        self.bore = boreholes.Borehole(self.H, self.D, self.r_b,
                                       self.x, self.y, tilt=self.tilt,
                                       orientation=self.orientation)
        self.x2 = 5.
        self.y2 = 8.
        self.bore2 = boreholes.Borehole(self.H, self.D, self.r_b,
                                        self.x2, self.y2, tilt=self.tilt,
                                        orientation=self.orientation)

    def test_inputs(self):
        """ Tests that all inputs return their value.
        """
        self.assertEqual(self.bore.H, self.H,
                         msg='borehole.H returns incorrect value.')
        self.assertEqual(self.bore.D, self.D,
                         msg='borehole.D returns incorrect value.')
        self.assertEqual(self.bore.r_b, self.r_b,
                         msg='borehole.r_b returns incorrect value.')
        self.assertEqual(self.bore.x, self.x,
                         msg='borehole.x returns incorrect value.')
        self.assertEqual(self.bore.y, self.y,
                         msg='borehole.y returns incorrect value.')
        self.assertEqual(self.bore.tilt, self.tilt,
                         msg='borehole.tilt returns incorrect value.')
        self.assertEqual(self.bore.orientation, self.orientation,
                         msg='borehole.orientation returns incorrect value.')

    def test_position(self):
        """ Tests position() class method.
        """
        pos = self.bore.position()
        self.assertEqual(pos, (self.x, self.y),
                         msg='borehole.position() returns incorrect value.')

    def test_distance_with_self(self, abs_tol=1.0e-6):
        """ Tests distance(borehole) class method returns radius when distance
            is less than the borehole radius.
        """
        dis = self.bore.distance(self.bore)
        self.assertAlmostEqual(dis, self.r_b, delta=abs_tol,
                         msg=('borehole.distance() should return the borehole '
                              'radius when the distance is less than r_b.'))

    def test_distance_with_another(self, abs_tol=1.0e-6):
        """ Tests distance(borehole2) class method returns separating distance.
        """
        dis = self.bore.distance(self.bore2)
        correct_dis = np.sqrt((self.x2 - self.x)**2 + (self.y2 - self.y)**2)
        self.assertAlmostEqual(dis, correct_dis, delta=abs_tol,
                         msg=('borehole.distance() does not evaluate correct '
                              'separating distance with target borehole.'))


class TestBoreFields_RectangleField(unittest.TestCase):
    """ Test cases for boreholes.rectangle_field().
    """

    def setUp(self):
        self.H = 150.
        self.D = 2.5
        self.r_b = 0.075
        self.B_1 = 5.
        self.B_2 = 6.

    def test_rectangular_1_borehole(self):
        """ Tests construction of rectangular field with one borehole.
        """
        from pygfunction import boreholes
        N_1 = 1
        N_2 = 1
        boreField = boreholes.rectangle_field(N_1, N_2, self.B_1, self.B_2,
                                              self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), 1,
                         msg=('Incorrect number of boreholes in '
                              'rectangle_field with N1=1 and N2=1.'))

    def test_rectangular_1_row(self):
        """ Tests construction of rectangular field with one row.
        """
        from pygfunction import boreholes
        N_1 = 4
        N_2 = 1
        boreField = boreholes.rectangle_field(N_1, N_2, self.B_1, self.B_2,
                                              self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), N_1,
                         msg=('Incorrect number of boreholes in '
                              'rectangle_field with N2=1.'))

    def test_rectangular_1_column(self):
        """ Tests construction of rectangular field with one column.
        """
        from pygfunction import boreholes
        N_1 = 1
        N_2 = 4
        boreField = boreholes.rectangle_field(N_1, N_2, self.B_1, self.B_2,
                                              self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), N_2,
                         msg=('Incorrect number of boreholes in '
                              'rectangle_field with N1=1.'))

    def test_rectangular(self):
        """ Tests construction of rectangular field with multiple rows/columns.
        """
        from pygfunction import boreholes
        N_1 = 4
        N_2 = 5
        boreField = boreholes.rectangle_field(N_1, N_2, self.B_1, self.B_2,
                                              self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), N_1*N_2,
                         msg=('Incorrect number of boreholes in '
                              'rectangle_field.'))


class TestBoreFields_LShapedField(unittest.TestCase):
    """ Test cases for boreholes.L_shaped_field().
    """

    def setUp(self):
        self.H = 150.
        self.D = 2.5
        self.r_b = 0.075
        self.B_1 = 5.
        self.B_2 = 6.

    def test_L_shaped_1_borehole(self):
        """ Tests construction of L-shaped field with one borehole.
        """
        from pygfunction import boreholes
        N_1 = 1
        N_2 = 1
        boreField = boreholes.L_shaped_field(N_1, N_2, self.B_1, self.B_2,
                                             self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), 1,
                         msg=('Incorrect number of boreholes in '
                              'L_shaped_field with N1=1 and N2=1.'))

    def test_L_shaped_1_row(self):
        """ Tests construction of L-shaped field with one borehole.
        """
        from pygfunction import boreholes
        N_1 = 5
        N_2 = 1
        boreField = boreholes.L_shaped_field(N_1, N_2, self.B_1, self.B_2,
                                             self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), N_1,
                         msg=('Incorrect number of boreholes in '
                              'L_shaped_field with N2=1.'))

    def test_L_shaped_1_column(self):
        """ Tests construction of L-shaped field with one borehole.
        """
        from pygfunction import boreholes
        N_1 = 1
        N_2 = 5
        boreField = boreholes.L_shaped_field(N_1, N_2, self.B_1, self.B_2,
                                             self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), N_2,
                         msg=('Incorrect number of boreholes in '
                              'L_shaped_field with N1=1.'))

    def test_L_shaped(self):
        """ Tests construction of L-shaped field with with multiple
            rows/columns.
        """
        from pygfunction import boreholes
        N_1 = 3
        N_2 = 5
        boreField = boreholes.L_shaped_field(N_1, N_2, self.B_1, self.B_2,
                                             self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), N_1+N_2-1,
                         msg=('Incorrect number of boreholes in '
                              'L_shaped_field.'))


class TestBoreFields_UShapedField(unittest.TestCase):
    """ Test cases for boreholes.U_shaped_field().
    """

    def setUp(self):
        self.H = 150.
        self.D = 2.5
        self.r_b = 0.075
        self.B_1 = 5.
        self.B_2 = 6.

    def test_U_shaped_1_borehole(self):
        """ Tests construction of L-shaped field with one borehole.
        """
        from pygfunction import boreholes
        N_1 = 1
        N_2 = 1
        boreField = boreholes.U_shaped_field(N_1, N_2, self.B_1, self.B_2,
                                             self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), 1,
                         msg=('Incorrect number of boreholes in '
                              'U_shaped_field with N1=1 and N2=1.'))

    def test_U_shaped_1_row(self):
        """ Tests construction of L-shaped field with one borehole.
        """
        from pygfunction import boreholes
        N_1 = 5
        N_2 = 1
        boreField = boreholes.U_shaped_field(N_1, N_2, self.B_1, self.B_2,
                                             self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), N_1,
                         msg=('Incorrect number of boreholes in '
                              'U_shaped_field with N2=1.'))

    def test_U_shaped_1_column(self):
        """ Tests construction of U-shaped field with one borehole.
        """
        from pygfunction import boreholes
        N_1 = 1
        N_2 = 5
        boreField = boreholes.U_shaped_field(N_1, N_2, self.B_1, self.B_2,
                                             self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), N_2,
                         msg=('Incorrect number of boreholes in '
                              'U_shaped_field with N1=1.'))

    def test_U_shaped(self):
        """ Tests construction of U-shaped field with with multiple
            rows/columns.
        """
        from pygfunction import boreholes
        N_1 = 3
        N_2 = 5
        boreField = boreholes.U_shaped_field(N_1, N_2, self.B_1, self.B_2,
                                             self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), N_1+2*N_2-2,
                         msg=('Incorrect number of boreholes in '
                              'U_shaped_field.'))


class TestBoreFields_BoxShapedField(unittest.TestCase):
    """ Test cases for boreholes.box_shaped_field().
    """

    def setUp(self):
        self.H = 150.
        self.D = 2.5
        self.r_b = 0.075
        self.B_1 = 5.
        self.B_2 = 6.

    def test_box_shaped_1_borehole(self):
        """ Tests construction of box-shaped field with one borehole.
        """
        from pygfunction import boreholes
        N_1 = 1
        N_2 = 1
        boreField = boreholes.box_shaped_field(N_1, N_2, self.B_1, self.B_2,
                                               self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), 1,
                         msg=('Incorrect number of boreholes in '
                              'box_shaped_field with N1=1 and N2=1.'))

    def test_box_shaped_1_row(self):
        """ Tests construction of L-shaped field with one borehole.
        """
        from pygfunction import boreholes
        N_1 = 5
        N_2 = 1
        boreField = boreholes.box_shaped_field(N_1, N_2, self.B_1, self.B_2,
                                             self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), N_1,
                         msg=('Incorrect number of boreholes in '
                              'box_shaped_field with N2=1.'))

    def test_box_shaped_1_column(self):
        """ Tests construction of box-shaped field with one borehole.
        """
        from pygfunction import boreholes
        N_1 = 1
        N_2 = 5
        boreField = boreholes.box_shaped_field(N_1, N_2, self.B_1, self.B_2,
                                               self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), N_2,
                         msg=('Incorrect number of boreholes in '
                              'box_shaped_field with N1=1.'))

    def test_box_shaped(self):
        """ Tests construction of box-shaped field with with multiple
            rows/columns.
        """
        from pygfunction import boreholes
        N_1 = 3
        N_2 = 5
        boreField = boreholes.box_shaped_field(N_1, N_2, self.B_1, self.B_2,
                                             self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), 2*N_1+2*N_2-4,
                         msg=('Incorrect number of boreholes in '
                              'box_shaped_field.'))


class TestBoreFields_CircleField(unittest.TestCase):
    """ Test cases for boreholes.circle_field().
    """

    def setUp(self):
        self.H = 150.
        self.D = 2.5
        self.r_b = 0.075
        self.R = 5.

    def test_circle_1_borehole(self):
        """ Tests construction of circle field with one borehole.
        """
        from pygfunction import boreholes
        N = 1
        boreField = boreholes.circle_field(N, self.R, self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), 1,
                         msg=('Incorrect number of boreholes in '
                              'circle_field with N1=1 and N2=1.'))

    def test_circle(self):
        """ Tests construction of circle field with with multiple boreholes.
        """
        from pygfunction import boreholes
        N = 8
        boreField = boreholes.circle_field(N, self.R, self.H, self.D, self.r_b)
        self.assertEqual(len(boreField), N,
                         msg=('Incorrect number of boreholes in '
                              'circle_field.'))


if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    unittest.main()
