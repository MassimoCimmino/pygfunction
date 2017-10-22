# -*- coding: utf-8 -*-
""" Test suite for pipes module.
"""
from __future__ import division, print_function, absolute_import

import unittest

import numpy as np
from scipy.constants import pi


class TestFluidFrictionFactor(unittest.TestCase):
    """ Test cases for calculation of friction factor in circular pipes.
    """

    def setUp(self):
        self.r_in = 0.01            # Inner radius [m]
        self.epsilon = 1.5e-6       # Pipe surface roughness [m]
        self.visc = 0.00203008      # Fluid viscosity [kg/m.s]
        self.den = 1014.78          # Fluid density

    def test_laminar(self, rel_tol=1.0e-3):
        """ Tests the value of the Darcy friction factor in laminar flow.
        """
        from pygfunction.pipes import fluid_friction_factor_circular_pipe
        m_flow = 0.05               # Fluid mass flow rate

        # Result from EES
        reference = 0.04082
        # Calculate using heat_transfer.fluid_friction_factor_circular_pipe()
        f = fluid_friction_factor_circular_pipe(m_flow, self.r_in, self.visc,
                                                self.den, self.epsilon)
        self.assertAlmostEqual(f, reference, delta=rel_tol*reference,
                               msg='Incorrect value of Darcy friction '
                                   'factor for laminar flow.')

    def test_turbulent_rough(self, rel_tol=1.0e-3):
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
        self.assertAlmostEqual(f, reference, delta=rel_tol*reference,
                               msg='Incorrect value of Darcy friction '
                                   'factor for turbulent flow in rough pipes.')


class TestConvectiveHeatTransferCoefficient(unittest.TestCase):
    """ Test cases for calculation of convection coefficient in circular pipes.
    """

    def setUp(self):
        self.r_in = 0.01            # Inner radius [m]
        self.epsilon = 1.5e-6       # Pipe surface roughness [m]
        self.visc = 0.00203008      # Fluid viscosity [kg/m.s]
        self.den = 1014.78          # Fluid density [kg/m3]
        self.cp = 3977.             # Fluid specific heat capacity [J/kg.K]
        self.k = 0.4922             # Fluid thermal conductivity [W/m.K]

    def test_turbulent(self, rel_tol=1.0e-3):
        """ Tests the value of the convection coefficient in laminar flow.
        """
        from pygfunction.pipes import convective_heat_transfer_coefficient_circular_pipe
        m_flow = 0.50               # Fluid mass flow rate

        # Result from EES
        reference = 4037.58769
        # Calculate using heat_transfer.fluid_friction_factor_circular_pipe()
        h = convective_heat_transfer_coefficient_circular_pipe(m_flow,
                                                               self.r_in,
                                                               self.visc,
                                                               self.den,
                                                               self.k,
                                                               self.cp,
                                                               self.epsilon)
        self.assertAlmostEqual(h, reference, delta=rel_tol*reference,
                               msg='Incorrect value of the convection coeff. '
                                   'for turbulent flow.')


class TestPipeThermalResistance(unittest.TestCase):
    """ Test cases for calculation of the thermal resistance through the pipe
        wall.
    """

    def setUp(self):
        self.r_in = 0.01            # Inner radius [m]
        self.r_out = 0.02           # Outer radius [m]
        self.k = 0.6                # Fluid thermal conductivity [W/m.K]

    def test_turbulent(self, rel_tol=1.0e-6):
        """ Tests the value of the convection coefficient in laminar flow.
        """
        from pygfunction.pipes import conduction_thermal_resistance_circular_pipe

        # Exact solution
        reference = np.log(self.r_out/self.r_in)/(2*pi*self.k)
        # Calculate using heat_transfer.fluid_friction_factor_circular_pipe()
        R = conduction_thermal_resistance_circular_pipe(self.r_in, self.r_out,
                                                        self.k)
        self.assertAlmostEqual(R, reference, delta=rel_tol*reference,
                               msg='Incorrect value of the thermal resistance '
                                   'through pipe wall.')


if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    unittest.main()
