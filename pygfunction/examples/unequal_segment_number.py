# -*- coding: utf-8 -*-
""" Example of calculation of g-functions with unequal segment lengths using
    uniform and equal borehole wall temperatures.

    The g-functions of a field of 6x4 boreholes is calculated with unequal
    number of segments.

"""
import matplotlib.pyplot as plt
import numpy as np

import pygfunction as gt


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    D = 4.0             # Borehole buried depth (m)
    H = 150.0           # Borehole length (m)
    r_b = 0.075         # Borehole radius (m)
    B = 7.5             # Borehole spacing (m)

    # Thermal properties
    alpha = 1.0e-6      # Ground thermal diffusivity (m2/s)

    # Geometrically expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt = 50                         # Number of time steps
    ts = H**2/(9.*alpha)            # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)

    # -------------------------------------------------------------------------
    # Borehole field
    # -------------------------------------------------------------------------

    # Field of 3x2 (n=6) boreholes
    N_1 = 6
    N_2 = 4
    boreField = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)

    # -------------------------------------------------------------------------
    # Evaluate g-functions with different segment options
    # -------------------------------------------------------------------------

    # g-Function calculation option for uniform borehole heights
    options = {'nSegments': 12, 'disp': True, 'profiles': True}
    gfunc = gt.gfunction.gFunction(
        boreField, alpha, time=time, options=options)

    print(gfunc.gFunc)

    # define number of segments as a list
    options = {'nSegments': [12] * len(boreField), 'disp': True}
    gfunc = gt.gfunction.gFunction(
        boreField, alpha, time=time, options=options)

    print(gfunc.gFunc)

    return


# Main function
if __name__ == '__main__':
    main()
