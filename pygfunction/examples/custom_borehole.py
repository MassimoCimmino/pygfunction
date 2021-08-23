# -*- coding: utf-8 -*-
""" Example definition of a borehole. A top-view plot of the borehole is
    created and the borehole resistance is computed.

"""
import pygfunction as gt
from numpy import pi
import numpy as np


def main():
    # Borehole dimensions
    H = 400.        # Borehole length (m)
    D = 5.          # Borehole buried depth (m)
    r_b = 0.0875    # Borehole radius (m)

    # Pipe dimensions
    r_out = 0.0133      # Pipe outer radius (m)
    r_in = 0.0108       # Pipe inner radius (m)
    D_s = 0.029445      # Shank spacing (m)
    epsilon = 1.0e-6    # Pipe roughness (m)

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
    m_flow_borehole = 0.25  # Total fluid mass flow rate per borehole (kg/s)
    cp_f = 3977.            # Fluid specific isobaric heat capacity (J/kg.K)
    rho_f = 1015.           # Fluid density (kg/m3)
    mu_f = 0.00203          # Fluid dynamic viscosity (kg/m.s)
    k_f = 0.492             # Fluid thermal conductivity (W/m.K)

    # Pipe thermal resistance
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
        r_in, r_out, k_p)
    # Fluid to inner pipe wall thermal resistance (Single U-tube)
    m_flow_pipe = m_flow_borehole
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow_pipe, r_in, mu_f, rho_f, k_f, cp_f, epsilon)
    R_f = 1.0 / (h_f * 2 * pi * r_in)

    SingleUTube = gt.pipes.SingleUTube(
        pos_single, r_in, r_out, borehole, k_s, k_g, R_f + R_p)

    R_b = gt.pipes.borehole_thermal_resistance(
        SingleUTube, m_flow_borehole, cp_f)

    print('Borehole thermal resistance: {0:.4f} m.K/W'.format(R_b))

    # Coaxial pipe
    # Note: Is a coaxial pipe typically set as far as the spacing between the
    #       inner and outer tubes go?
    pos = (0., 0.)  # Coordinates of the coaxial pipe axis
    # Pipe dimensions
    r_inner = np.array([44.2 / 1000. / 2., 50. / 1000. / 2.])  # Inner pipe radii (m)
    r_outer = np.array([97.4 / 1000. / 2., 110. / 1000. / 2.])  # Outer pip radii (m)
    epsilon = [1.0e-6, 1.0e-6]  # Pipe roughness (m)

    # Thermal properties
    k_p = [0.4, 0.4]  # Inner and outer pipe thermal conductivity (W/m.K)

    # Check the geometry to make sure it is physically possible
    #
    # This class method is automatically called at the instanciation of the
    # pipe object and raises an error if the pipe geometry is invalid. It is
    # manually called here for demosntration.
    # check_single = SingleUTube._check_geometry()
    # print('The geometry of the borehole is valid (realistic/possible): '
    #       + str(check_single))
    # check_double = DoubleUTube_ser._check_geometry()
    # print('The geometry of the borehole is valid (realistic/possible): '
    #       + str(check_double))
    #
    # # Create a borehole top view
    # fig_single = SingleUTube.visualize_pipes()
    # fig_double = DoubleUTube_ser.visualize_pipes()
    #
    # # Save the figure as a pdf
    # fig_single.savefig('singe-u-tube-borehole-top-view.pdf')
    # fig_double.savefig('double-u-tube-borehole-top-view.pdf')


if __name__ == '__main__':
    main()
