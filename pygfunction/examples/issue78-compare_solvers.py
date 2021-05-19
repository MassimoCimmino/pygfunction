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
    # Compare all rectangular fields from 1 by 1 to 12 by 12
    # -------------------------------------------------------------------------
    options = {'nSegments':12, 'disp':True}
    N_min = 1
    N_max = 12
    time_rectangular = [[], []]
    gFunc_sim_rectangular = []
    gFunc_det_rectangular = []
    N_rectangular = []
    for N_1 in range(N_min, N_max+1):
        for N_2 in range(N_1, N_max+1):
            print(' Field of {} by {} boreholes '.format(N_1, N_2).center(60, '='))
            boreholes = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)
            N_rectangular.append(len(boreholes))
            tic = tim.time()
            gfunc_sim = gt.gfunction.gFunction(
                boreholes, alpha, time=time, method='new-similarities', options=options)
            toc0 = tim.time()
            gfunc_det = gt.gfunction.gFunction(
                boreholes, alpha, time=time, method='new-detailed', options=options)
            toc1 = tim.time()
            time_rectangular[0].append(toc0 - tic)
            time_rectangular[1].append(toc1 - toc0)
            gFunc_sim_rectangular.append(gfunc_sim.gFunc)
            gFunc_det_rectangular.append(gfunc_det.gFunc)

    fig = _initialize_figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'Number of boreholes')
    ax1.set_ylabel(r'Calculation time [sec]')
    ax1.set_title('Rectangular fields up to {} by {}'.format(N_max, N_max))
    _format_axes(ax1)
    ax1.loglog(N_rectangular, time_rectangular[0], 'o', label='Similarities')
    ax1.loglog(N_rectangular, time_rectangular[1], 'x', label='Detailed')
    ax1.legend()
    plt.tight_layout()

    fig = _initialize_figure()
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel(r'$g_{detailed}$')
    ax2.set_ylabel(r'$g_{similarities} - g_{detailed}$')
    ax2.set_title('Rectangular fields up to {} by {}'.format(N_max, N_max))
    _format_axes(ax2)
    for (g_sim, g_det) in zip(gFunc_sim_rectangular, gFunc_det_rectangular):
        ax2.plot(g_det, g_sim - g_det, 'o')
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Compare fields of 5 by 5 boreholes with increasing number of segments
    # -------------------------------------------------------------------------
    boreholes = gt.boreholes.rectangle_field(5, 5, B, B, H, D, r_b)
    time_segments = [[], []]
    gFunc_sim_segments = []
    gFunc_det_segments = []
    N_segments = []
    for n in range(0, 6):
        nSegments = 2**n
        options = {'nSegments':nSegments, 'disp':True}
        print(' Field of 5 by 5 boreholes ({} segments) '.format(nSegments).center(60, '='))
        N_segments.append(nSegments)
        tic = tim.time()
        gfunc_sim = gt.gfunction.gFunction(
            boreholes, alpha, time=time, method='new-similarities', options=options)
        toc0 = tim.time()
        gfunc_det = gt.gfunction.gFunction(
            boreholes, alpha, time=time, method='new-detailed', options=options)
        toc1 = tim.time()
        time_segments[0].append(toc0 - tic)
        time_segments[1].append(toc1 - toc0)
        gFunc_sim_segments.append(gfunc_sim.gFunc)
        gFunc_det_segments.append(gfunc_det.gFunc)

    fig = _initialize_figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'Number of segments')
    ax1.set_ylabel(r'Calculation time [sec]')
    ax1.set_title('Field of 5 by 5 boreholes (nSegments)')
    _format_axes(ax1)
    ax1.loglog(N_segments, time_segments[0], 'o', label='Similarities')
    ax1.loglog(N_segments, time_segments[1], 'x', label='Detailed')
    ax1.legend()
    plt.tight_layout()

    fig = _initialize_figure()
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel(r'$g_{detailed}$')
    ax2.set_ylabel(r'$g_{similarities} - g_{detailed}$')
    ax2.set_title('Field of 5 by 5 boreholes (nSegments)')
    _format_axes(ax2)
    for (g_sim, g_det) in zip(gFunc_sim_segments, gFunc_det_segments):
        ax2.plot(g_det, g_sim - g_det, 'o')
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Compare field of 6 by 6 boreholes with outer ring of boreholes of
    # reduced length (totalling 10 by 10 boreholes)
    # -------------------------------------------------------------------------
    options = {'nSegments':12, 'disp':True, 'profiles':True}
    boreholes = gt.boreholes.rectangle_field(6, 6, B, B, H, D, r_b)
    boreholes1 = gt.boreholes.box_shaped_field(8, 8, B, B, 85., D, r_b)
    boreholes2 = gt.boreholes.box_shaped_field(10, 10, B, B, 85., D, r_b)
    for b in boreholes1:
        b.x = b.x - B
        b.y = b.y - B
    for b in boreholes2:
        b.x = b.x - 2*B
        b.y = b.y - 2*B
    boreholes = boreholes + boreholes1 + boreholes2
    H_mean = np.mean([b.H for b in boreholes])
    ts = H_mean**2/(9*alpha)
    
    time_uneven = [[], []]
    gFunc_sim_uneven = []
    gFunc_det_uneven = []
    print(' Field of 6 by 6 boreholes with outer ring '.format(nSegments).center(60, '='))
    tic = tim.time()
    gfunc_sim = gt.gfunction.gFunction(
        boreholes, alpha, time=time, method='new-similarities', options=options)
    toc0 = tim.time()
    gfunc_det = gt.gfunction.gFunction(
        boreholes, alpha, time=time, method='new-detailed', options=options)
    toc1 = tim.time()
    time_uneven[0].append(toc0 - tic)
    time_uneven[1].append(toc1 - toc0)
    gFunc_sim_uneven.append(gfunc_sim.gFunc)
    gFunc_det_uneven.append(gfunc_det.gFunc)

    ax = gfunc_sim.visualize_g_function().axes[0]
    ax.plot(np.log(gfunc_det.time/ts), gfunc_det.gFunc, 'kx')
    ax.legend(['Similarities, t = {:.2f} sec'.format(toc0 - tic), 'Detailed, t = {:.2f} sec'.format(toc1 - toc0)])
    ax.set_title('Field of 6 by 6 boreholes with outer ring')
    plt.tight_layout()

    gfunc_sim.visualize_heat_extraction_rate_profiles(iBoreholes=[0, 36, 64])
    gfunc_det.visualize_heat_extraction_rate_profiles(iBoreholes=[0, 36, 64])

    fig = _initialize_figure()
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel(r'$g_{detailed}$')
    ax2.set_ylabel(r'$g_{similarities} - g_{detailed}$')
    ax2.set_title('Field of 6 by 6 boreholes with outer ring')
    _format_axes(ax2)
    for (g_sim, g_det) in zip(gFunc_sim_uneven, gFunc_det_uneven):
        ax2.plot(g_det, g_sim - g_det, 'o')
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
    
    time_uneven2 = [[], []]
    gFunc_sim_uneven2 = []
    gFunc_det_uneven2 = []
    print(' Field of 5 uneven boreholes '.format().center(60, '='))
    tic = tim.time()
    gfunc_sim = gt.gfunction.gFunction(
        boreholes, alpha, time=time, method='new-similarities', options=options)
    toc0 = tim.time()
    gfunc_det = gt.gfunction.gFunction(
        boreholes, alpha, time=time, method='new-detailed', options=options)
    toc1 = tim.time()
    time_uneven2[0].append(toc0 - tic)
    time_uneven2[1].append(toc1 - toc0)
    gFunc_sim_uneven2.append(gfunc_sim.gFunc)
    gFunc_det_uneven2.append(gfunc_det.gFunc)

    ax = gfunc_sim.visualize_g_function().axes[0]
    ax.plot(np.log(gfunc_det.time/ts), gfunc_det.gFunc, 'kx')
    ax.legend(['Similarities, t = {:.2f} sec'.format(toc0 - tic), 'Detailed, t = {:.2f} sec'.format(toc1 - toc0)])
    ax.set_title('Field of 5 uneven boreholes')
    plt.tight_layout()

    gfunc_sim.visualize_heat_extraction_rate_profiles()
    gfunc_det.visualize_heat_extraction_rate_profiles()

    fig = _initialize_figure()
    ax2 = fig.add_subplot(111)
    ax2.set_xlabel(r'$g_{detailed}$')
    ax2.set_ylabel(r'$g_{similarities} - g_{detailed}$')
    ax2.set_title('Field of 5 uneven boreholes')
    _format_axes(ax2)
    for (g_sim, g_det) in zip(gFunc_sim_uneven2, gFunc_det_uneven2):
        ax2.plot(g_det, g_sim - g_det, 'o')
    plt.tight_layout()

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
