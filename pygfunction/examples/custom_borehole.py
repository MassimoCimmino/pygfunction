# -*- coding: utf-8 -*-
""" Example of definition of a bore field using custom borehole positions.

"""
from __future__ import absolute_import, division, print_function

import pygfunction as gt


def main():
    # Borehole dimensions
    H = 400  # Borehole length (m)
    D = 5  # burial depth (m)
    r_b = 0.0875  # Borehole radius (m)

    # Pipe dimensions
    rp_out = 0.0133  # Pipe outer radius (m)
    rp_in = 0.0108  # Pipe inner radius (m)
    D_s = 0.029445  # Shank spacing (m)

    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos_single = [(-D_s, 0.), (D_s, 0.)]

    # define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

    # variables necessary for defining u-tube, but not needed to visualize borehole
    k_s = 1
    k_g = 1
    R_f_ser = 1
    R_p = 1

    SingleUTube = gt.pipes.SingleUTube(pos_single, rp_in, rp_out,
                                       borehole, k_s, k_g, R_f_ser + R_p)

    # check the geometry to make sure it is physically possible
    SingleUTube._check_geometry()

    # create a borehole top view
    fig = SingleUTube.visualize_pipes()

    # save the figure as a pdf
    fig.savefig('borehole-top-view.pdf')


if __name__ == '__main__':
    main()
