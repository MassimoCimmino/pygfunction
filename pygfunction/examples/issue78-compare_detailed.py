# -*- coding: utf-8 -*-
"""
Develop new functions to find similarties by adding an intermediate step with
comparisons of segments (instead of going straight to segments).
"""

import time as tim

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import erf
from functools import partial
from multiprocessing import Pool

import pygfunction as gt

def main():
    out = []
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    D = 4.0             # Borehole buried depth (m)
    H = 150.0           # Borehole length (m)
    r_b = 0.075         # Borehole radius (m)
    B = 7.5             # Borehole spacing (m)

    # Number of segments per borehole
    nSegments = 12

    # Thermal properties
    alpha = 1.0e-6      # Ground thermal diffusivity (m2/s)

    # Geometrically expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt = 25                         # Number of time steps
    ts = H**2/(9.*alpha)           # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)

    # -------------------------------------------------------------------------
    # Compare all rectangular fields from 1 by 1 to 5 by 5 with the
    # existing detailed method using 12 segments
    # -------------------------------------------------------------------------
    options = {'nSegments':12, 'disp':True}
    N_min = 1
    N_max = 5
    time_rectangular = [[], []]
    gFunc_new_rectangular = []
    gFunc_old_rectangular = []
    N_rectangular = []
    for N_1 in range(N_min, N_max+1):
        for N_2 in range(N_1, N_max+1):
            print(' Field of {} by {} boreholes '.format(N_1, N_2).center(60, '='))
            boreholes = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)
            N_rectangular.append(len(boreholes))
            tic = tim.time()
            gfunc_new = gt.gfunction.gFunction(
                boreholes, alpha, time=time, method='new-detailed', options=options)
            toc0 = tim.time()
            gfunc_old = gt.gfunction.gFunction(
                boreholes, alpha, time=time, method='detailed', options=options)
            toc1 = tim.time()
            time_rectangular[0].append(toc0 - tic)
            time_rectangular[1].append(toc1 - toc0)
            gFunc_new_rectangular.append(gfunc_new.gFunc)
            gFunc_old_rectangular.append(gfunc_old.gFunc)

    fig = _initialize_figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'Number of boreholes')
    ax1.set_ylabel(r'Calculation time [sec]')
    ax1.set_title('Rectangular fields up to {} by {} (Detailed solver)'.format(N_max, N_max))
    _format_axes(ax1)
    ax1.loglog(N_rectangular, time_rectangular[0], 'o', label='New')
    ax1.loglog(N_rectangular, time_rectangular[1], 'x', label='Old')
    ax1.legend()
    plt.tight_layout()

    fig = _initialize_figure()
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel(r'$g_{old}$')
    ax2.set_ylabel(r'$g_{new} - g_{old}$')
    ax2.set_title('Rectangular fields up to {} by {} (Detailed solver)'.format(N_max, N_max))
    _format_axes(ax2)
    for (g_new, g_old) in zip(gFunc_new_rectangular, gFunc_old_rectangular):
        ax2.plot(g_old, g_new - g_old, 'o')
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Compare fields of 3 by 3 boreholes with increasing number of segments
    # -------------------------------------------------------------------------
    boreholes = gt.boreholes.rectangle_field(3, 3, B, B, H, D, r_b)
    time_segments = [[], []]
    gFunc_new_segments = []
    gFunc_old_segments = []
    N_segments = []
    for n in [1, 3, 6, 12]:
        nSegments = n
        options = {'nSegments':nSegments, 'disp':True}
        print(' Field of 3 by 3 boreholes ({} segments) '.format(nSegments).center(60, '='))
        N_segments.append(nSegments)
        tic = tim.time()
        gfunc_new = gt.gfunction.gFunction(
            boreholes, alpha, time=time, method='new-detailed', options=options)
        toc0 = tim.time()
        gfunc_old = gt.gfunction.gFunction(
            boreholes, alpha, time=time, method='detailed', options=options)
        toc1 = tim.time()
        time_segments[0].append(toc0 - tic)
        time_segments[1].append(toc1 - toc0)
        gFunc_new_segments.append(gfunc_new.gFunc)
        gFunc_old_segments.append(gfunc_old.gFunc)

    fig = _initialize_figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'Number of segments')
    ax1.set_ylabel(r'Calculation time [sec]')
    ax1.set_title('Field of 3 by 3 boreholes (nSegments) (Detailed solver)')
    _format_axes(ax1)
    ax1.loglog(N_segments, time_segments[0], 'o', label='New')
    ax1.loglog(N_segments, time_segments[1], 'x', label='Old')
    ax1.legend()
    plt.tight_layout()

    fig = _initialize_figure()
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel(r'$g_{old}$')
    ax2.set_ylabel(r'$g_{new} - g_{old}$')
    ax2.set_title('Field of 3 by 3 boreholes (nSegments) (Detailed solver)')
    _format_axes(ax2)
    for (g_new, g_old) in zip(gFunc_new_segments, gFunc_old_segments):
        ax2.plot(g_old, g_new - g_old, 'o')
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Compare field of 5 uneven boreholes
    # -------------------------------------------------------------------------
    options = {'nSegments':12, 'disp':True, 'profiles':True}
    H_all = np.array([150., 85., 150., 175., 125.])
    x_all = np.array([0., 5., 12.5, 17.5, 25.])
    boreholes = [gt.boreholes.Borehole(Hi, D, r_b, xi, 0.) for (Hi, xi) in zip(H_all, x_all)]
    H_mean = np.mean([b.H for b in boreholes])
    ts = H_mean**2/(9*alpha)
    
    time_uneven = [[], []]
    gFunc_new_uneven = []
    gFunc_old_uneven = []
    print(' Field of 5 uneven boreholes '.format().center(60, '='))
    tic = tim.time()
    gfunc_new = gt.gfunction.gFunction(
        boreholes, alpha, time=time, method='new-detailed', options=options)
    toc0 = tim.time()
    gfunc_old = gt.gfunction.gFunction(
        boreholes, alpha, time=time, method='detailed', options=options)
    toc1 = tim.time()
    time_uneven[0].append(toc0 - tic)
    time_uneven[1].append(toc1 - toc0)
    gFunc_new_uneven.append(gfunc_new.gFunc)
    gFunc_old_uneven.append(gfunc_old.gFunc)

    ax = gfunc_new.visualize_g_function().axes[0]
    ax.plot(np.log(gfunc_old.time/ts), gfunc_old.gFunc, 'kx')
    ax.legend(['New, t = {:.2f} sec'.format(toc0 - tic), 'Old, t = {:.2f} sec'.format(toc1 - toc0)])
    ax.set_title('Field of 5 uneven boreholes (Detailed solver)')
    plt.tight_layout()

    gfunc_new.visualize_heat_extraction_rate_profiles()
    gfunc_old.visualize_heat_extraction_rate_profiles()

    fig = _initialize_figure()
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel(r'$g_{old}$')
    ax2.set_ylabel(r'$g_{new} - g_{old}$')
    ax2.set_title('Field of 5 uneven boreholes (Detailed solver)')
    _format_axes(ax2)
    for (g_new, g_old) in zip(gFunc_new_uneven, gFunc_old_uneven):
        ax2.plot(g_old, g_new - g_old, 'o')
    plt.tight_layout()

    fig = gt.boreholes.visualize_field(boreholes)

    return

def _initialize_figure():
    """
    Initialize a matplotlib figure object with overwritten default
    parameters.

    Returns
    -------
    fig : figure
        Figure object (matplotlib).

    """
    plt.rc('font', size=9)
    plt.rc('xtick', labelsize=9)
    plt.rc('ytick', labelsize=9)
    plt.rc('lines', lw=1.5, markersize=5.0)
    plt.rc('savefig', dpi=500)
    fig = plt.figure()
    return fig

def _format_axes(ax):
    """
    Adjust axis parameters.

    Parameters
    ----------
    ax : axis
        Axis object (amtplotlib).

    """
    from matplotlib.ticker import AutoMinorLocator
    # Draw major and minor tick marks inwards
    ax.tick_params(
        axis='both', which='both', direction='in',
        bottom=True, top=True, left=True, right=True)
    # Auto-adjust minor tick marks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    return


# Main function
if __name__ == '__main__':
    out = main()
