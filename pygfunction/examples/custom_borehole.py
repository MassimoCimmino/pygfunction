# -*- coding: utf-8 -*-
""" Example of definition and visualization of a borehole.

"""
from __future__ import absolute_import, division, print_function

import pygfunction as gt


def main():
    # Borehole dimensions
    H = 400.        # Borehole length (m)
    D = 5.          # Borehole buried depth (m)
    r_b = 0.0875    # Borehole radius (m)

    # Pipe dimensions
    rp_out = 0.0133     # Pipe outer radius (m)
    rp_in = 0.0108      # Pipe inner radius (m)
    D_s = 0.029445      # Shank spacing (m)

    # Pipe positions
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos = [(-D_s, 0.), (D_s, 0.)]

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

    # Heat transfer properties are necessary for defining a U-tube, but not
    # needed to visualize the borehole. All properties are set to 1.
    k_s = 1.0     # Ground thermal conductivity (W/m.K)
    k_g = 1.0     # Grout thermal conductivity (W/m.K)
    R_f = 1.0     # Fluid convective thermal resistance (m.K/W)
    R_p = 1.0     # Pipe conduction thermal resistance (m.K/W)

    SingleUTube = gt.pipes.SingleUTube(
        pos, rp_in, rp_out, borehole, k_s, k_g, R_f + R_p)

    # Check the geometry to make sure it is physically possible
    #
    # This class method is automatically called at the instanciation of the
    # pipe object and raises an error if the pipe geometry is invalid. It is
    # manually called here for demosntration.
    check = SingleUTube._check_geometry()
    print(check)

    # Create a borehole top view
    fig = SingleUTube.visualize_pipes()

    # Save the figure as a pdf
    fig.savefig('borehole-top-view.pdf')


if __name__ == '__main__':
    main()
