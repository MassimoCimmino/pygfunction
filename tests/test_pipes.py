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
        # Calculate using pipes.fluid_friction_factor_circular_pipe()
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
        # Calculate using pipes.fluid_friction_factor_circular_pipe()
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
        # Calculate using pipes.convective_heat_transfer_coefficient_circular_pipe()
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
        # Calculate using pipes.conduction_thermal_resistance_circular_pipe()
        R = conduction_thermal_resistance_circular_pipe(self.r_in, self.r_out,
                                                        self.k)
        self.assertAlmostEqual(R, reference, delta=rel_tol*reference,
                               msg='Incorrect value of the thermal resistance '
                                   'through pipe wall.')


class TestBoreholeThermalResistances(unittest.TestCase):
    """ Test cases for calculation of the internal thermal resistances of a
        borehole.
    """

    def setUp(self):
        # Pipe positions [m]
        self.pos_2pipes = [(0.03, 0.00), (-0.03, 0.02)]
        self.r_out = 0.02       # Pipe outer radius [m]
        self.r_b = 0.07         # Borehole radius [m]
        self.k_s = 2.5          # Ground thermal conductivity [W/m.K]
        self.k_g = 1.5          # Grout thermal conductivity [W/m.K]
        beta = 1.2
        # Fluid to outer pipe wall thermal resistance [m.K/W]
        self.Rfp = beta/(2*pi*self.k_g)

    def test_line_source_approximation(self):
        """ Tests the value of the internal resistances and conductances.
        """
        from pygfunction.pipes import thermal_resistances

        # Exact solution (Claesson and Hellstrom, 2011)
        R_ref = np.array([[25.486e-2, 01.538e-2],
                          [01.538e-2, 25.207e-2]])
        S_ref = np.array([[3.698, 0.240],
                          [0.240, 3.742]])
        # Calculate using heat_transfer.fluid_friction_factor_circular_pipe()
        R, Rd = thermal_resistances(self.pos_2pipes, self.r_out, self.r_b,
                                    self.k_s, self.k_g, self.Rfp, J=0)
        S = 1/Rd
        self.assertTrue(np.allclose(R, R_ref, rtol=1e-8, atol=1e-5),
                        msg='Incorrect value of the internal thermal '
                            'resistances.')
        self.assertTrue(np.allclose(S, S_ref, rtol=1e-8, atol=1e-3),
                        msg='Incorrect value of the delta-circuit thermal '
                            'resistances.')

    def test_first_order_multipole(self):
        """ Tests the value of the internal resistances and conductances.
        """
        from pygfunction.pipes import thermal_resistances

        # Exact solution (Claesson and Hellstrom, 2011)
        R_ref = np.array([[25.569e-2, 01.562e-2],
                          [01.562e-2, 25.288e-2]])
        S_ref = np.array([[3.683, 0.243],
                          [0.243, 3.727]])
        # Calculate using heat_transfer.fluid_friction_factor_circular_pipe()
        R, Rd = thermal_resistances(self.pos_2pipes, self.r_out, self.r_b,
                                    self.k_s, self.k_g, self.Rfp, J=1)
        S = 1/Rd
        self.assertTrue(np.allclose(R, R_ref, rtol=1e-8, atol=1e-5),
                        msg='Incorrect value of the internal thermal '
                            'resistances.')
        self.assertTrue(np.allclose(S, S_ref, rtol=1e-8, atol=1e-3),
                        msg='Incorrect value of the delta-circuit thermal '
                            'resistances.')

    def test_second_order_multipole(self):
        """ Tests the value of the internal resistances and conductances.
        """
        from pygfunction.pipes import thermal_resistances

        # Exact solution (Claesson and Hellstrom, 2011)
        R_ref = np.array([[25.590e-2, 01.561e-2],
                          [01.561e-2, 25.309e-2]])
        S_ref = np.array([[3.681, 0.242],
                          [0.242, 3.724]])
        # Calculate using heat_transfer.fluid_friction_factor_circular_pipe()
        R, Rd = thermal_resistances(self.pos_2pipes, self.r_out, self.r_b,
                                    self.k_s, self.k_g, self.Rfp, J=2)
        S = 1/Rd
        self.assertTrue(np.allclose(R, R_ref, rtol=1e-8, atol=1e-5),
                        msg='Incorrect value of the internal thermal '
                            'resistances.')
        self.assertTrue(np.allclose(S, S_ref, rtol=1e-8, atol=1e-3),
                        msg='Incorrect value of the delta-circuit thermal '
                            'resistances.')


if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    unittest.main()
