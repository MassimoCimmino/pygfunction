# -*- coding: utf-8 -*-
""" Comparison of solvers for the evaluation of g-functions using uniform and
    equal borehole wall temperatures.

    The g-function of a field of 6x4 boreholes is calculated for a boundary
    condition of uniform borehole wall temperature along the boreholes, equal
    for all boreholes. Three different solvers are compared : 'detailed',
    'similarities' and 'equivalent'. Their accuracy and calculation time are
    compared using the 'detailed' solver as a reference. This shows that the
    'similarities' solver can evaluate g-functions with high accuracy.

    The g-function of a field of 12x10 boreholes is calculated for a boundary
    condition of uniform borehole wall temperature along the boreholes, equal
    for all boreholes. Two different solvers are compared : 'similarities' and
    'equivalent'. The accuracy and calculation time of the 'equivalent' is
    compared using the 'similarities' solver as a reference. This shows that
    the 'equivalent' solver evaluates g-functions at a very high calculation
    speed while maintaining reasonable accuracy.

"""
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter

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

    # g-Function calculation options
    options = {'nSegments': 8,
               'disp': True}

    # Geometrically expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt = 15                         # Number of time steps
    ts = H**2/(9.*alpha)            # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)
    lntts = np.log(time/ts)

    # -------------------------------------------------------------------------
    # Borehole field (First bore field)
    # -------------------------------------------------------------------------

    # Field of 6x4 (n=24) boreholes
    N_1 = 6
    N_2 = 4
    field = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)

    # -------------------------------------------------------------------------
    # Evaluate g-functions
    # -------------------------------------------------------------------------
    t0 = perf_counter()
    gfunc_detailed = gt.gfunction.gFunction(
        field, alpha, time=time, options=options, method='detailed')
    t1 = perf_counter()
    t_detailed = t1 - t0
    gfunc_similarities = gt.gfunction.gFunction(
        field, alpha, time=time, options=options, method='similarities')
    t2 = perf_counter()
    t_similarities = t2 - t1
    gfunc_equivalent = gt.gfunction.gFunction(
        field, alpha, time=time, options=options, method='equivalent')
    t3 = perf_counter()
    t_equivalent = t3 - t2

    # -------------------------------------------------------------------------
    # Plot results
    # -------------------------------------------------------------------------
    # Draw g-functions
    ax = gfunc_detailed.visualize_g_function().axes[0]
    ax.plot(lntts, gfunc_similarities.gFunc, 'bx')
    ax.plot(lntts, gfunc_equivalent.gFunc, 'ro')
    ax.legend([f'detailed (t = {t_detailed:.3f} sec)',
               f'similarities (t = {t_similarities:.3f} sec)',
               f'equivalent (t = {t_equivalent:.3f} sec)'])
    ax.set_title(f'Field of {N_1} by {N_2} boreholes')
    plt.tight_layout()

    # Draw absolute error
    # Configure figure and axes
    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)
    # Axis labels
    ax.set_xlabel(r'ln$(t/t_s)$')
    ax.set_ylabel(r'Absolute error')
    gt.utilities._format_axes(ax)
    # Absolute error
    ax.plot(lntts, np.abs(gfunc_similarities.gFunc - gfunc_detailed.gFunc),
            '-', label='similarities')
    ax.plot(lntts, np.abs(gfunc_equivalent.gFunc - gfunc_detailed.gFunc),
            '--', label='equivalent')
    ax.legend()
    ax.set_title(f"Absolute error relative to the 'detailed' solver "
                 f"(Field of {N_1} by {N_2} boreholes)")
    # Adjust to plot window
    fig.tight_layout()

    # Draw relative error
    # Configure figure and axes
    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)
    # Axis labels
    ax.set_xlabel(r'ln$(t/t_s)$')
    ax.set_ylabel(r'Relative error')
    gt.utilities._format_axes(ax)
    # Relative error
    gFunc_ref = gfunc_detailed.gFunc  # reference g-function
    ax.plot(lntts, (gfunc_similarities.gFunc - gFunc_ref) / gFunc_ref,
            '-', label='similarities')
    ax.plot(lntts, (gfunc_equivalent.gFunc - gFunc_ref) / gFunc_ref,
            '--', label='equivalent')
    ax.legend()
    ax.set_title(f"Relative error relative to the 'detailed' solver "
                 f"(Field of {N_1} by {N_2} boreholes)")
    # Adjust to plot window
    fig.tight_layout()

    # -------------------------------------------------------------------------
    # Borehole field (Second bore field)
    # -------------------------------------------------------------------------

    # Field of 6x4 (n=24) boreholes
    N_1 = 12
    N_2 = 10
    field = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)

    # -------------------------------------------------------------------------
    # Evaluate g-functions
    # -------------------------------------------------------------------------
    gfunc_similarities = gt.gfunction.gFunction(
        field, alpha, time=time, options=options, method='similarities')
    t2 = perf_counter()
    t_similarities = t2 - t1
    gfunc_equivalent = gt.gfunction.gFunction(
        field, alpha, time=time, options=options, method='equivalent')
    t3 = perf_counter()
    t_equivalent = t3 - t2

    # -------------------------------------------------------------------------
    # Plot results
    # -------------------------------------------------------------------------
    # Draw g-functions
    ax = gfunc_similarities.visualize_g_function().axes[0]
    ax.plot(lntts, gfunc_equivalent.gFunc, 'ro')
    ax.legend([f'similarities (t = {t_similarities:.3f} sec)',
               f'equivalent (t = {t_equivalent:.3f} sec)'])
    ax.set_title(f'Field of {N_1} by {N_2} boreholes')
    plt.tight_layout()

    # Draw absolute error
    # Configure figure and axes
    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)
    # Axis labels
    ax.set_xlabel(r'ln$(t/t_s)$')
    ax.set_ylabel(r'Absolute error')
    gt.utilities._format_axes(ax)
    # Absolute error
    ax.plot(lntts, np.abs(gfunc_equivalent.gFunc - gfunc_similarities.gFunc),
            label='equivalent')
    ax.legend()
    ax.set_title(f"Absolute error relative to the 'similarities' solver "
                 f"(Field of {N_1} by {N_2} boreholes)")
    # Adjust to plot window
    fig.tight_layout()

    # Draw relative error
    # Configure figure and axes
    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)
    # Axis labels
    ax.set_xlabel(r'ln$(t/t_s)$')
    ax.set_ylabel(r'Relative error')
    gt.utilities._format_axes(ax)
    # Relative error
    ax.plot(lntts, (gfunc_equivalent.gFunc - gfunc_similarities.gFunc) / gfunc_similarities.gFunc,
            label='equivalent')
    ax.legend()
    ax.set_title(f"Relative error relative to the 'similarities' solver "
                 f"(Field of {N_1} by {N_2} boreholes)")
    # Adjust to plot window
    fig.tight_layout()

    return


# Main function
if __name__ == '__main__':
    main()
