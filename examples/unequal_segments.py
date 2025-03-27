# -*- coding: utf-8 -*-
""" Example of calculation of g-functions with varied declaration of segments
    using uniform borehole wall temperature.

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
    Nt = 25                         # Number of time steps
    ts = H**2/(9.*alpha)            # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)

    # -------------------------------------------------------------------------
    # Borehole field
    # -------------------------------------------------------------------------

    # Field of 6x4 (n=24) boreholes
    N_1 = 6
    N_2 = 4
    borefield = gt.borefield.Borefield.rectangle_field(
        N_1, N_2, B, B, H, D, r_b)
    gt.boreholes.visualize_field(borefield)

    # -------------------------------------------------------------------------
    # Evaluate g-functions with different segment options
    # -------------------------------------------------------------------------

    # The 'similarities' method is used to consider unequal numbers of segments
    # per borehole and to plot heat extraction rate profiles along
    # individual boreholes
    method = 'similarities'

    # Calculate g-function with equal number of segments for all boreholes and
    # uniform segment lengths
    nSegments = 24
    options = {'nSegments': nSegments,
               'segment_ratios': None,
               'disp': True}

    gfunc_equal = gt.gfunction.gFunction(
        borefield, alpha, time=time, options=options, method=method)

    # Calculate g-function with predefined number of segments for each
    # borehole, the segment lengths will be uniform along each borehole, but
    # the number of segments for the boreholes can be unequal.

    # Boreholes 12, 14 and 18 have more segments than the others and their
    # heat extraction rate profiles are plotted.
    nSegments = [12] * len(borefield)
    nSegments[12] = 24
    nSegments[14] = 24
    nSegments[18] = 24
    options = {'nSegments': nSegments,
               'segment_ratios': None,
               'disp': True,
               'profiles': True}

    gfunc_unequal = gt.gfunction.gFunction(
        borefield, alpha, time=time, options=options, method=method)

    # Calculate g-function with equal number of segments for each borehole,
    # unequal segment lengths along the length of the borehole defined by
    # segment ratios. The segment ratios for each borehole are the same.

    nSegments = 8
    # Define the segment ratios for each borehole in each segment
    # the segment lengths are defined top to bottom left to right
    segment_ratios = np.array([0.05, 0.10, 0.10, 0.25, 0.25, 0.10, 0.10, 0.05])
    options = {'nSegments': nSegments,
               'segment_ratios': segment_ratios,
               'disp': True,
               'profiles': True}

    g_func_predefined = gt.gfunction.gFunction(
        borefield, alpha, time=time, options=options, method=method)

    # -------------------------------------------------------------------------
    # Plot g-functions
    # -------------------------------------------------------------------------

    ax = gfunc_equal.visualize_g_function().axes[0]
    ax.plot(np.log(time/ts), gfunc_unequal.gFunc, 'r-.')
    ax.plot(np.log(time/ts), g_func_predefined.gFunc, 'k-.')
    ax.legend(['Equal number of segments',
               'Unequal number of segments',
               'Unequal segment lengths'])
    plt.tight_layout()

    # Heat extraction rate profiles
    fig = gfunc_unequal.visualize_heat_extraction_rates(
        iBoreholes=[18, 12, 14])
    fig.suptitle('Heat extraction rates (unequal number of segments)')
    fig.tight_layout()
    fig = g_func_predefined.visualize_heat_extraction_rates(
        iBoreholes=[18, 12, 14])
    fig.suptitle('Heat extraction rates (unequal segment lengths)')
    fig.tight_layout()
    fig = gfunc_unequal.visualize_heat_extraction_rate_profiles(
        iBoreholes=[18, 12, 14])
    fig.suptitle('Heat extraction rate profiles (unequal number of segments)')
    fig.tight_layout()
    fig = g_func_predefined.visualize_heat_extraction_rate_profiles(
        iBoreholes=[18, 12, 14])
    fig.suptitle('Heat extraction rate profiles (unequal segment lengths)')
    fig.tight_layout()

    return


# Main function
if __name__ == '__main__':
    main()
