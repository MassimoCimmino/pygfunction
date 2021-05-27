# -*- coding: utf-8 -*-
""" Example definition of a borehole. A top-view plot of the borehole is
    created and the borehole resistance is computed.

"""
from __future__ import absolute_import, division, print_function

import pygfunction as gt
from numpy import pi


def main():
    # Borehole dimensions
    H = 400.        # Borehole length (m)
    D = 5.          # Borehole buried depth (m)
    r_b = 0.0875    # Borehole radius (m)

    # Pipe dimensions
    rp_out = 0.0133     # Pipe outer radius (m)
    rp_in = 0.0108      # Pipe inner radius (m)
    D_s = 0.029445      # Shank spacing (m)
    epsilon = 1.0e-6  # Pipe roughness (m)

    # Pipe positions
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos_single = [(-D_s, 0.), (D_s, 0.)]
    # Double U-tube [(x_in1, y_in1), (x_in2, y_in2),
    #                (x_out1, y_out1), (x_out2, y_out2)]
    # Note: in series configuration, fluid enters pipe (in,1), exits (out,1),
    # then enters (in,2) and finally exits (out,2)
    # (if you view visualize_pipe, series is 1->3->2->4)
    pos_double = [(-D_s, 0.), (0., -D_s), (D_s, 0.), (0., D_s)]

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

    k_p = 0.4     # Pipe thermal conductivity (W/m.K)
    k_s = 2.0     # Ground thermal conductivity (W/m.K)
    k_g = 1.0     # Grout thermal conductivity (W/m.K)

    # Fluid properties
    m_flow = 0.25  # Total fluid mass flow rate per borehole (kg/s)
    mixer = 'MPG'  # Propylene glycol/water mixture
    percent = 20.  # 20% of the mixture of propylene glycol
    T = 20.  # Temperature of fluid (degrees C)
    P = 101325.  # Gage pressure of the fluid (Pa)
    fluid = gt.media.Fluid(mixer=mixer, percent=percent, T=T, P=P)

    cp_f = fluid.cp  # Fluid specific isobaric heat capacity (J/kg.K)
    den_f = fluid.rho  # Fluid density (kg/m3)
    visc_f = fluid.mu  # Fluid dynamic viscosity (kg/m.s)
    k_f = fluid.k  # Fluid thermal conductivity (W/m.K)

    # Thermal resistances
    # Pipe thermal resistance
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(rp_in,
                                                               rp_out,
                                                               k_p)
    # Fluid to inner pipe wall thermal resistance (Single U-tube and double
    # U-tube in series)
    h_f_ser = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(m_flow,
                                                                          rp_in,
                                                                          visc_f,
                                                                          den_f,
                                                                          k_f,
                                                                          cp_f,
                                                                          epsilon)
    R_f_ser = 1.0 / (h_f_ser * 2 * pi * rp_in)
    # Fluid to inner pipe wall thermal resistance (Double U-tube in parallel)
    h_f_par = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow / 2, rp_in, visc_f, den_f, k_f, cp_f, epsilon)
    R_f_par = 1.0 / (h_f_par * 2 * pi * rp_in)

    # U-Tube definitions
    # Single U-tube (series)
    SingleUTube = gt.pipes.SingleUTube(
        pos_single, rp_in, rp_out, borehole, k_s, k_g, R_f_ser + R_p)

    # Double U-tube (series)
    DoubleUTube_ser = gt.pipes.MultipleUTube(pos_double, rp_in, rp_out,
                                             borehole, k_s, k_g, R_f_ser + R_p,
                                             nPipes=2, config='series')

    # Double U-tube (parallel)
    DoubleUTube_par = gt.pipes.MultipleUTube(pos_double, rp_in, rp_out,
                                             borehole, k_s, k_g, R_f_par + R_p,
                                             nPipes=2, config='parallel')

    # Effective borehole resistances
    # Single U-tube (series) effective borehole resistance
    Rb_single = gt.pipes.borehole_thermal_resistance(SingleUTube, m_flow, cp_f)

    # Double U-tube (series) effective borehole resistance
    Rb_double_ser = gt.pipes.borehole_thermal_resistance(DoubleUTube_ser, m_flow, cp_f)

    # Double U-tube (parallel) effective borehole resistance
    Rb_double_par = gt.pipes.borehole_thermal_resistance(DoubleUTube_par, m_flow, cp_f)

    print('Single U-tube (series) effective borehole resistance: {0:.4f} m.K/W'.
          format(Rb_single))
    print('Double U-tube (series) effective borehole resistance: {0:.4f} m.K/W'.
          format(Rb_double_ser))
    print('Double U-tube (parallel) effective borehole resistance: {0:.4f} m.K/W'.
          format(Rb_double_par))
    # Check the geometry to make sure it is physically possible
    #
    # This class method is automatically called at the instanciation of the
    # pipe object and raises an error if the pipe geometry is invalid. It is
    # manually called here for demosntration.
    check_single = SingleUTube._check_geometry()
    print('The geometry of the borehole is valid (realistic/possible): '
          + str(check_single))
    check_double = DoubleUTube_ser._check_geometry()
    print('The geometry of the borehole is valid (realistic/possible): '
          + str(check_double))

    # Create a borehole top view
    fig_single = SingleUTube.visualize_pipes()
    fig_double = DoubleUTube_ser.visualize_pipes()

    # Save the figure as a pdf
    fig_single.savefig('singe-u-tube-borehole-top-view.pdf')
    fig_double.savefig('double-u-tube-borehole-top-view.pdf')


if __name__ == '__main__':
    main()
