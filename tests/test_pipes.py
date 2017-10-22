# -*- coding: utf-8 -*-
""" Test suite for pipes module.
"""
from __future__ import division, print_function, absolute_import

import unittest


class TestFluidFrictionFactor(unittest.TestCase):
    """ Test cases for calculation of friction factor in circular pipes.
    """

    def setUp(self):
        self.r_in = 0.01            # Inner radius [m]
        self.epsilon = 1.5e-6       # Pipe surface roughness [m]
        self.visc = 0.00203008      # Fluid viscosity [kg/m.s]
        self.den = 1014.78          # Fluid density

    def test_laminar(self, abs_tol=3.0e-5):
        """ Tests the value of the Darcy friction factor in laminar flow.
        """
        from pygfunction.pipes import fluid_friction_factor_circular_pipe
        m_flow = 0.05               # Fluid mass flow rate

        # Result from EES
        reference = 0.04082
        # Calculate using heat_transfer.fluid_friction_factor_circular_pipe()
        f = fluid_friction_factor_circular_pipe(m_flow, self.r_in, self.visc,
                                                self.den, self.epsilon)
        self.assertAlmostEqual(f, reference, delta=abs_tol,
                               msg='Incorrect value of Darcy friction '
                                   'factor for laminar flow.')

    def test_turbulent_rough(self, abs_tol=3.0e-5):
        """ Tests the value of the Darcy friction factor in turbulent flow
            in rough pipes.
        """
        from pygfunction.pipes import fluid_friction_factor_circular_pipe
        m_flow = 0.50               # Fluid mass flow rate

        # Result from EES
        reference = 0.02766
        # Calculate using heat_transfer.fluid_friction_factor_circular_pipe()
        f = fluid_friction_factor_circular_pipe(m_flow, self.r_in, self.visc,
                                                self.den, self.epsilon,
                                                tol=1.0e-8)
        self.assertAlmostEqual(f, reference, delta=abs_tol,
                               msg='Incorrect value of Darcy friction '
                                   'factor for turbulent flow in rough pipes.')


if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    unittest.main()
