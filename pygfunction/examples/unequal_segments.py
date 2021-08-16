# -*- coding: utf-8 -*-
""" Example of calculation of g-functions with varied declaration of segments
    using uniform borehole wall temperature.

    The g-functions of a field of 6x4 boreholes is calculated with unequal
    number of segments.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
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

    # Field of 6x4 (n=24) boreholes
    N_1 = 6
    N_2 = 4
    boreField = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)
    gt.boreholes.visualize_field(boreField)

    # -------------------------------------------------------------------------
    # Evaluate g-functions with different segment options
    # -------------------------------------------------------------------------

    # Calculate g-function with equal number of segments
    nSegments = 12
    options = {'nSegments': nSegments, 'disp': True}

    gfunc_equal = gt.gfunction.gFunction(
        boreField, alpha, time=time, options=options)

    # Calculate g-function with unequal number of segments

    # Boreholes 12, 14 and 18 have more segments than the others and their
    # heat extraction rate profiles are plotted.
    nSegments = [12] * len(boreField)
    nSegments[12] = 24
    nSegments[14] = 24
    nSegments[18] = 24
    options = {'nSegments': nSegments, 'disp': True, 'profiles': True}

    gfunc_unequal = gt.gfunction.gFunction(
        boreField, alpha, time=time, options=options)

    # -------------------------------------------------------------------------
    # Plot g-functions
    # -------------------------------------------------------------------------

    ax = gfunc_equal.visualize_g_function().axes[0]
    ax.plot(np.log(time/ts), gfunc_unequal.gFunc, 'r-.')
    ax.legend(['Equal number of segments', 'Unequal number of segments'])
    plt.tight_layout()

    # Heat extraction rate profiles
    gfunc_unequal.visualize_heat_extraction_rates(
        iBoreholes=[18, 12, 14])
    gfunc_unequal.visualize_heat_extraction_rate_profiles(
        iBoreholes=[18, 12, 14])

    return


# Main function
if __name__ == '__main__':
    main()

