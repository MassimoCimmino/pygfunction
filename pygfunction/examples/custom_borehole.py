# -*- coding: utf-8 -*-
""" Example definition of a borehole. A top-view plot of the borehole is
    created and the borehole resistance is computed.

"""
import numpy as np
from scipy.constants import pi

import pygfunction as gt


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    D = 5.          # Borehole buried depth (m)
    H = 400.        # Borehole length (m)
    r_b = 0.0875    # Borehole radius (m)

    # Pipe dimensions (all configurations)
    epsilon = 1.0e-6    # Pipe roughness (m)

    # Pipe dimensions (single U-tube and double U-tube)
    r_out = 0.0211      # Pipe outer radius (m)
    r_in = 0.0147       # Pipe inner radius (m)
    D_s = 0.052         # Shank spacing (m)

    # Pipe dimensions (coaxial)
    r_in_in = 0.0221    # Inside pipe inner radius (m)
    r_in_out = 0.025    # Inside pipe outer radius (m)
    r_out_in = 0.0487   # Outer pipe inside radius (m)
    r_out_out = 0.055   # Outer pipe outside radius (m)
    # Vectors of inner and outer pipe radii
    # Note : The dimensions of the inlet pipe are the first elements of
    #        the vectors. In this example, the inlet pipe is the inside pipe.
    r_inner = np.array([r_in_in, r_out_in])     # Inner pipe radii (m)
    r_outer = np.array([r_in_out, r_out_out])   # Outer pip radii (m)

    # Ground properties
    k_s = 2.0           # Ground thermal conductivity (W/m.K)

    # Grout properties
    k_g = 1.0           # Grout thermal conductivity (W/m.K)

    # Pipe properties
    k_p = 0.4           # Pipe thermal conductivity (W/m.K)

    # Fluid properties
    # Total fluid mass flow rate per borehole (kg/s)
    m_flow_borehole = 1.0
    # The fluid is propylene-glycol (20 %) at 20 degC
    fluid = gt.media.Fluid('MPG', 20.)
    cp_f = fluid.cp     # Fluid specific isobaric heat capacity (J/kg.K)
    rho_f = fluid.rho   # Fluid density (kg/m3)
    mu_f = fluid.mu     # Fluid dynamic viscosity (kg/m.s)
    k_f = fluid.k       # Fluid thermal conductivity (W/m.K)

    # -------------------------------------------------------------------------
    # Initialize borehole model
    # -------------------------------------------------------------------------

    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

    # -------------------------------------------------------------------------
    # Define a single U-tube borehole
    # -------------------------------------------------------------------------

    # Pipe positions
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos_single = [(-D_s, 0.), (D_s, 0.)]

    # Pipe thermal resistance
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
        r_in, r_out, k_p)

    # Fluid to inner pipe wall thermal resistance
    m_flow_pipe = m_flow_borehole
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow_pipe, r_in, mu_f, rho_f, k_f, cp_f, epsilon)
    R_f = 1.0 / (h_f * 2 * pi * r_in)

    # Single U-tube GHE in borehole
    SingleUTube = gt.pipes.SingleUTube(
        pos_single, r_in, r_out, borehole, k_s, k_g, R_f + R_p)

    # Check the geometry to make sure it is physically possible

    # This class method is automatically called at the instanciation of the
    # pipe object and raises an error if the pipe geometry is invalid. It is
    # manually called here for demonstration.
    check_single = SingleUTube._check_geometry()
    print('The geometry of the borehole is valid (realistic/possible): '
          + str(check_single))

    # Evaluate and print the effective borehole thermal resistance
    R_b = SingleUTube.effective_borehole_thermal_resistance(
        m_flow_borehole, fluid.cp)
    print('Single U-tube Borehole thermal resistance: '
          '{0:.4f} m.K/W'.format(R_b))

    # Visualize the borehole geometry and save the figure
    fig_single = SingleUTube.visualize_pipes()
    fig_single.savefig('single-u-tube-borehole.png')

    # -------------------------------------------------------------------------
    # Define a double U-tube borehole
    # -------------------------------------------------------------------------

    # Pipe positions
    # Double U-tube [(x_in1, y_in1), (x_in2, y_in2),
    #                (x_out1, y_out1), (x_out2, y_out2)]
    # Note: in series configuration, fluid enters pipe (in,1), exits (out,1),
    # then enters (in,2) and finally exits (out,2)
    # (if you view visualize_pipe, series is 1->3->2->4)
    pos_double = [(-D_s, 0.), (0., -D_s), (D_s, 0.), (0., D_s)]

    # Pipe thermal resistance
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
            r_in, r_out, k_p)

    # Fluid to inner pipe wall thermal resistance
    # Double U-tube in series
    m_flow_pipe_series = m_flow_borehole
    h_f_series = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow_pipe_series, r_in, mu_f, rho_f, k_f, cp_f, epsilon)
    R_f_series = 1.0 / (h_f_series * 2 * pi * r_in)
    # Double U-tube in parallel
    m_flow_pipe_parallel = m_flow_borehole / 2
    h_f_parallel = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow_pipe_parallel, r_in, mu_f, rho_f, k_f, cp_f, epsilon)
    R_f_parallel = 1.0 / (h_f_parallel * 2 * pi * r_in)

    # Double U-tube GHE in borehole
    # Double U-tube in series
    DoubleUTube_series = gt.pipes.MultipleUTube(
        pos_double, r_in, r_out, borehole, k_s, k_g, R_p + R_f_series, 2,
        config='series')
    # Double U-tube in parallel
    DoubleUTube_parallel = gt.pipes.MultipleUTube(
        pos_double, r_in, r_out, borehole, k_s, k_g, R_p + R_f_parallel, 2,
        config='parallel')

    # Evaluate and print the effective borehole thermal resistance
    R_b_series = DoubleUTube_series.effective_borehole_thermal_resistance(
        m_flow_borehole, fluid.cp)
    print('Double U-tube (series) Borehole thermal resistance: {0:.4f} m.K/W'.
          format(R_b_series))
    R_b_parallel = DoubleUTube_parallel.effective_borehole_thermal_resistance(
        m_flow_borehole, fluid.cp)
    print('Double U-tube (parallel) Borehole thermal resistance: {0:.4f} m.K/W'.
          format(R_b_parallel))

    # Visualize the borehole geometry and save the figure
    fig_double = DoubleUTube_series.visualize_pipes()
    fig_double.savefig('double-u-tube-borehole.png')

    # -------------------------------------------------------------------------
    # Define a coaxial borehole
    # -------------------------------------------------------------------------

    # Pipe positions
    # Coaxial pipe (x, y)
    pos = (0., 0.)

    # Pipe thermal resistance
    # (the two pipes have the same thermal conductivity, k_p)
    # Inner pipe
    R_p_in = gt.pipes.conduction_thermal_resistance_circular_pipe(
        r_in_in, r_in_out, k_p)
    # Outer pipe
    R_p_out = gt.pipes.conduction_thermal_resistance_circular_pipe(
        r_out_in, r_out_out, k_p)

    # Fluid-to-fluid thermal resistance
    # Inner pipe
    h_f_in = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow_borehole, r_in_in, mu_f, rho_f, k_f, cp_f, epsilon)
    R_f_in = 1.0 / (h_f_in * 2 * pi * r_in_in)
    # Outer pipe
    h_f_a_in, h_f_a_out = \
        gt.pipes.convective_heat_transfer_coefficient_concentric_annulus(
            m_flow_borehole, r_in_out, r_out_in, mu_f, rho_f, k_f, cp_f,
            epsilon)
    R_f_out_in = 1.0 / (h_f_a_in * 2 * pi * r_in_out)
    R_ff = R_f_in + R_p_in + R_f_out_in

    # Coaxial GHE in borehole
    R_f_out_out = 1.0 / (h_f_a_out * 2 * pi * r_out_in)
    R_fp = R_p_out + R_f_out_out
    Coaxial = gt.pipes.Coaxial(
        pos, r_inner, r_outer, borehole, k_s, k_g, R_ff, R_fp, J=2)

    # Evaluate and print the effective borehole thermal resistance
    R_b = Coaxial.effective_borehole_thermal_resistance(
        m_flow_borehole, fluid.cp)
    print('Coaxial tube Borehole thermal resistance: {0:.4f} m.K/W'.
          format(R_b))

    # Visualize the borehole geometry and save the figure
    fig_coaxial = Coaxial.visualize_pipes()
    fig_coaxial.savefig('coaxial-borehole.png')


if __name__ == '__main__':
    main()
