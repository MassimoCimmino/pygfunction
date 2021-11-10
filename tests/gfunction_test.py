# -*- coding: utf-8 -*-
""" Test suite for gfunction module.
"""
import unittest

import numpy as np

import pygfunction.utilities


class TestSolvers(unittest.TestCase):
    """
    Test cases for calculation of g-functions using various solver methods
    available.
    """

    def setUp(self) -> None:
        self.H = 150.  # Borehole length [m]
        self.D = 4.  # Borehole buried depth [m]
        self.r_b = 0.075  # Borehole radius [m]
        self.B = 7.5  # Borehole spacing [m]
        self.alpha = 1.0e-6  # Ground thermal diffusivity [m2/s]

    def test_detailed(self):
        from pygfunction.gfunction import gFunction
        from pygfunction.boreholes import rectangle_field
        from pygfunction.utilities import time_geometric, segment_ratios
        # Calculation of the g-function at the same time values
        N_1 = 3
        N_2 = 2
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)

        # Thermal properties
        alpha = 1.0e-6  # Ground thermal diffusivity (m2/s)

        # Geometrically expanding time vector.
        dt = 100 * 3600.  # Time step
        tmax = 3000. * 8760. * 3600.  # Maximum time
        Nt = 10  # Number of time steps
        time = time_geometric(dt, tmax, Nt)

        # g-Function calculation options
        options_uniform = {'nSegments': 24,
                          'segment_ratios': None,
                          'disp': False}
        ratios = segment_ratios(8)
        options_unequal = {'nSegments': 8,
                          'segment_ratios': ratios,
                          'disp': False}
        method = 'detailed'

        gfunc_UBWT_uniform = gFunction(
            boreField, alpha, time=time, options=options_uniform,
            method=method).gFunc

        gfunc_UBWT_unequal = gFunction(
            boreField, alpha, time=time, options=options_unequal,
            method=method).gFunc

        self.assertTrue(np.allclose(gfunc_UBWT_uniform, gfunc_UBWT_unequal,
                                    rtol=1e-2, atol=1e-6),
                        msg='Incorrect g-function for the detailed solver test.'
                            'A converged solution is compared to a discretized'
                            'solution using the UBWT boundary condition.')

    def test_similarities(self):
        from pygfunction.gfunction import gFunction
        from pygfunction.boreholes import rectangle_field
        from pygfunction.utilities import time_geometric, segment_ratios
        # Calculation of the g-function at the same time values
        N_1 = 3
        N_2 = 2
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)

        # Thermal properties
        alpha = 1.0e-6  # Ground thermal diffusivity (m2/s)

        # Geometrically expanding time vector.
        dt = 100 * 3600.  # Time step
        tmax = 3000. * 8760. * 3600.  # Maximum time
        Nt = 10  # Number of time steps
        time = time_geometric(dt, tmax, Nt)

        # g-Function calculation options
        options_uniform = {'nSegments': 24,
                          'segment_ratios': None,
                          'disp': False}
        ratios = segment_ratios(8)
        options_unequal = {'nSegments': 8,
                          'segment_ratios': ratios,
                          'disp': False}
        method = 'similarities'

        gfunc_UBWT_uniform = gFunction(
            boreField, alpha, time=time, options=options_uniform,
            method=method).gFunc

        gfunc_UBWT_unequal = gFunction(
            boreField, alpha, time=time, options=options_unequal,
            method=method).gFunc

        self.assertTrue(np.allclose(gfunc_UBWT_uniform, gfunc_UBWT_unequal,
                                    rtol=1e-2, atol=1e-6),
                        msg='Incorrect g-function for the similarities solver '
                            'test.'
                            'A converged solution is compared to a discretized'
                            'solution using the UT boundary condition.')

    def test_equivalent(self):
        from pygfunction.gfunction import gFunction
        from pygfunction.boreholes import rectangle_field
        from pygfunction.utilities import time_geometric, segment_ratios
        # Calculation of the g-function at the same time values
        N_1 = 3
        N_2 = 2
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)

        # Thermal properties
        alpha = 1.0e-6  # Ground thermal diffusivity (m2/s)

        # Geometrically expanding time vector.
        dt = 100 * 3600.  # Time step
        tmax = 3000. * 8760. * 3600.  # Maximum time
        Nt = 10  # Number of time steps
        time = time_geometric(dt, tmax, Nt)

        # g-Function calculation options
        options_uniform = {'nSegments': 24,
                          'segment_ratios': None,
                          'disp': False}
        ratios = segment_ratios(8)
        options_unequal = {'nSegments': 8,
                          'segment_ratios': ratios,
                          'disp': False}
        method = 'equivalent'

        gfunc_UBWT_uniform = gFunction(
            boreField, alpha, time=time, options=options_uniform,
            method=method).gFunc

        gfunc_UBWT_unequal = gFunction(
            boreField, alpha, time=time, options=options_unequal,
            method=method).gFunc

        self.assertTrue(np.allclose(gfunc_UBWT_uniform, gfunc_UBWT_unequal,
                                    rtol=1e-2, atol=1e-6),
                        msg='Incorrect g-function for the equivalent solver '
                            'test.'
                            'A converged solution is compared to a discretized'
                            'solution using the UT boundary condition.')


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

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-6),
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

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-6),
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

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-6),
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

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-6),
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
        g = uniform_temperature(
            boreField, time, self.alpha, nSegments=12, segment_ratios=None)

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-6),
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
        g = uniform_temperature(
            boreField, time, self.alpha, nSegments=12, segment_ratios=None)

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-6),
                        msg='Incorrect values of the g-function of three by '
                            'two field for uniform temperature (12 segments).')

    def test_unequal_segments(self, rel_tol=1.0e-3):
        from pygfunction.gfunction import gFunction
        from pygfunction.boreholes import rectangle_field
        from pygfunction.utilities import time_geometric
        # Calculation of the g-function at the same time values
        N_1 = 3
        N_2 = 2
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)

        # Geometrically expanding time vector.
        dt = 100 * 3600.  # Time step
        tmax = 3000. * 8760. * 3600.  # Maximum time
        Nt = 10  # Number of time steps
        time = time_geometric(dt, tmax, Nt)

        nSegments = [12, 11, 13, 12, 11, 13]

        # g-Function calculation option for uniform borehole segment lengths
        # in the field by defining nSegments as an integer >= 1
        options = {'nSegments': nSegments[0],
                   'segment_ratios': None}
        method = 'similarities'
        gfunc = gFunction(
            boreField, self.alpha, time=time, options=options, method=method)
        g_ref = gfunc.gFunc

        # g-Function calculation with nSegments passed as list, where the list
        # is of the same length as boreField and each borehole is defined to
        # have >= 1 segment
        # Note: nSegments[i] pertains to boreField[i]
        options = {'nSegments': nSegments,
                   'segment_ratios': None}
        method = 'similarities'
        gfunc = gFunction(
            boreField, self.alpha, time=time, options=options, method=method)
        g = gfunc.gFunc

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-6),
                        msg='Incorrect values of the g-function of six '
                            'boreholes for uniform temperature and '
                            'unequal numbers of segments.')


class TestMixedInletTemperature(unittest.TestCase):
    """ Test cases for calculation of g-functions using mixed inlet fluid
        temperature boundary condition.
    """

    def setUp(self):
        self.H = 150.           # Borehole length [m]
        self.D = 4.             # Borehole buried depth [m]
        self.r_b = 0.075        # Borehole radius [m]
        self.B = 7.5            # Borehole spacing [m]
        self.alpha = 1.0e-6     # Ground thermal diffusivity [m2/s]
        self.k_s = 2.0          # Ground thermal conductivity [W/m.K]
        self.k_g = 1.0          # Grout thermal conductivity [W/m.K]
        self.k_p = 0.4          # Pipe thermal conductivity [W/m.K]
        self.r_out = 0.02       # Pipe outer radius [m]
        self.r_in = 0.015       # Pipe inner radius [m]
        self.D_s = 0.05         # Shank spacing [m]
        self.epsilon = 1.0e-06  # Pipe roughness [m]
        self.m_flow_network = 0.25  # Fluid mass flor rate in network [kg/s]

    def test_unequal_segments(self, rel_tol=1.0e-2):
        """ Tests the value of the g-function of six boreholes in series with
            unequal numbers of segments.
        """
        from pygfunction.gfunction import gFunction
        from pygfunction.boreholes import rectangle_field
        from pygfunction.utilities import time_geometric, segment_ratios
        from pygfunction.pipes import \
            conduction_thermal_resistance_circular_pipe, \
            convective_heat_transfer_coefficient_circular_pipe, \
            SingleUTube
        from pygfunction.media import Fluid
        from pygfunction.networks import Network
        from numpy import pi
        # Calculation of the g-function at the same time values
        N_1 = 3
        N_2 = 2
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)

        # Geometrically expanding time vector.
        dt = 100 * 3600.  # Time step
        tmax = 3000. * 8760. * 3600.  # Maximum time
        Nt = 10  # Number of time steps
        time = time_geometric(dt, tmax, Nt)

        nSegments = [12, 11, 13, 12, 11, 13]

        # Fluid is propylene-glycol (20 %) at 20 degC
        fluid = Fluid('MPG', 20.)

        bore_connectivity = list(range(-1, 5))
        # Pipe thermal resistance
        R_p = conduction_thermal_resistance_circular_pipe(
            self.r_in, self.r_out, self.k_p)
        m_flow_pipe = self.m_flow_network  # all boreholes in series
        h_f = convective_heat_transfer_coefficient_circular_pipe(
            m_flow_pipe, self.r_in, fluid.mu, fluid.rho, fluid.k, fluid.cp,
            self.epsilon)
        R_f = 1.0/(h_f*2*pi*self.r_in)

        pos_pipes = [(-self.D_s, 0), (self.D_s, 0)]

        UTubes = []
        for borebole in boreField:
            UTube = SingleUTube(pos_pipes, self.r_in, self.r_out, borebole,
                                self.k_s, self.k_g, R_f + R_p)
            UTubes.append(UTube)
        network = Network(
            boreField, UTubes, bore_connectivity=bore_connectivity,
            m_flow_network=m_flow_pipe, cp_f=fluid.cp)

        # g-Function calculation option for uniform borehole segment lengths
        # in the field by defining nSegments as an integer >= 1
        options = {'nSegments': nSegments[0],
                   'segment_ratios': None}
        method = 'similarities'
        gfunc = gFunction(
            network, self.alpha, time=time, options=options, method=method)
        g_reference = gfunc.gFunc

        # g-Function calculation with nSegments passed as list, where the list
        # is of the same length as boreField and each borehole is defined to
        # have >= 1 segment
        # Note: nSegments[i] pertains to boreField[i]
        options = {'nSegments': nSegments}
        method = 'similarities'
        gfunc = gFunction(
            network, self.alpha, time=time, options=options, method=method)
        g_unequal_nSegments = gfunc.gFunc

        self.assertTrue(
            np.allclose(
                g_unequal_nSegments, g_reference, rtol=rel_tol, atol=1e-6),
            msg='Incorrect values of the g-function of six '
                'boreholes for mixed inlet temperature and '
                'unequal numbers of segments.')

        # Compute g-function with predefined segment lengths
        nSegments = 8
        # Define the segment ratios for each borehole in each segment
        # the segment lengths are defined top to bottom left to right
        ratios = np.array(
            [0.05, 0.10, 0.10, 0.25, 0.25, 0.10, 0.10, 0.05])
        options = {'nSegments': nSegments,
                   'segment_ratios': ratios,
                   'disp': False}
        method = 'similarities'

        gfunc = gFunction(
            network, self.alpha, time=time, options=options, method=method)
        g_predefined_ratios = gfunc.gFunc

        self.assertTrue(
            np.allclose(
                g_predefined_ratios, g_reference, rtol=rel_tol, atol=1e-6),
            msg='Incorrect values of the g-function of six '
                'boreholes for mixed inlet temperature and '
                'unequal numbers of segments.')

        # Compute g-function with discretized segment lengths
        ratios = segment_ratios(8)
        options = {'nSegments': nSegments,
                   'segment_ratios': ratios,
                   'disp': False}
        method = 'similarities'

        gfunc = gFunction(
            network, self.alpha, time=time, options=options, method=method)
        g_generated_ratios = gfunc.gFunc

        self.assertTrue(
            np.allclose(
                g_generated_ratios, g_reference, rtol=rel_tol, atol=1e-6),
            msg='Incorrect values of the g-function of six '
                'boreholes for mixed inlet temperature and '
                'unequal numbers of segments.')


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

        # Reference results
        # Number of line source segments per borehole
        nSegments = 24
        # Very large time values to approximate steady-state
        time = 1.0e20
        # Borehole thermal resistances [m.K/W]
        Rb = np.array([0.1, 0.3])
        # Borehole short-circuit resistance [m.K/W]
        Ra = 0.25
        g_ref = np.array([6.63668744, 6.65139651])

        # Calculation of the g-function at the same time values
        N_1 = 1
        N_2 = 1
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)
        g = np.zeros_like(g_ref)
        for i in range(len(Rb)):
            UTubes = [SingleUTube(
                    pos, r_in, r_out, borehole, self.k_s, k_g, 1e-30, J=0)
                      for borehole in boreField]
            # Overwrite borehole resistances
            Rd_00 = 2*Rb[i]
            Rd_01 = 2*Ra*Rd_00/(2*Rd_00 - Ra)
            for UTube in UTubes:
                UTube._Rd[0,0] = Rd_00
                UTube._Rd[1,1] = Rd_00
                UTube._Rd[0,1] = Rd_01
                UTube._Rd[1,0] = Rd_01
                UTube._update_model_variables(self.m_flow, self.cp, nSegments, None)
            g[i] = equal_inlet_temperature(
                boreField, UTubes, self.m_flow, self.cp, time, self.alpha,
                nSegments=nSegments, segment_ratios=None)

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-6),
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

        # Reference results
        # Number of line source segments per borehole
        nSegments = 24
        # Very large time values to approximate steady-state
        time = 1.0e20
        # Borehole thermal resistances [m.K/W]
        Rb = 1.0
        # Borehole short-circuit resistance [m.K/W]
        Ra = 0.25
        g_ref = 28.01257172

        # Calculation of the g-function at the same time values
        N_1 = 4
        N_2 = 4
        boreField = rectangle_field(N_1, N_2, self.B, self.B,
                                    self.H, self.D, self.r_b)
        g = np.zeros_like(g_ref)
        UTubes = [SingleUTube(
                pos, r_in, r_out, borehole, self.k_s, k_g, 1e-30, J=0)
                  for borehole in boreField]
        # Overwrite borehole resistances
        Rd_00 = 2*Rb
        Rd_01 = 2*Ra*Rd_00/(2*Rd_00 - Ra)
        for UTube in UTubes:
            UTube._Rd[0,0] = Rd_00
            UTube._Rd[1,1] = Rd_00
            UTube._Rd[0,1] = Rd_01
            UTube._Rd[1,0] = Rd_01
            UTube._update_model_variables(self.m_flow, self.cp, nSegments, None)
        g = equal_inlet_temperature(
            boreField, UTubes, self.m_flow, self.cp, time, self.alpha,
            nSegments=nSegments, segment_ratios=None)

        self.assertTrue(np.allclose(g, g_ref, rtol=rel_tol, atol=1e-6),
                        msg='Incorrect values of the g-function of four by '
                            'four field for equal inlet fluid temperature '
                            '(24 segments).')
        
        
if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    unittest.main()
