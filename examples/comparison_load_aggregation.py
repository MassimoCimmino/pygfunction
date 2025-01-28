# -*- coding: utf-8 -*-
""" Comparison of the accuracy and computational speed of different load
    aggregation algorithms.

    The g-function of a single borehole is calculated for boundary condition of
    uniform borehole wall temperature along the borehole. Then, the borehole
    wall temperature variations resulting from a time-varying load profile
    are simulated using the aggregation methods of Bernier et al. (2004),
    Liu (2005), and Claesson and Javed (2012). Results are compared to the
    exact solution obtained by convolution in the Fourier domain.

    Default parameters are used for each of the aggregation schemes.

"""
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve

import pygfunction as gt


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    D = 4.0             # Borehole buried depth (m)
    H = 150.0           # Borehole length (m)
    r_b = 0.075         # Borehole radius (m)

    # Ground properties
    alpha = 1.0e-6      # Ground thermal diffusivity (m2/s)
    k_s = 2.0           # Ground thermal conductivity (W/m.K)
    T_g = 10.0          # Undisturbed ground temperature (degC)

    # g-Function calculation options
    options = {'nSegments': 8,
               'disp': True}

    # Simulation parameters
    dt = 3600.                  # Time step (s)
    tmax = 20.*8760. * 3600.    # Maximum time (s)
    Nt = int(np.ceil(tmax/dt))  # Number of time steps
    time = dt * np.arange(1, Nt+1)

    # Evaluate heat extraction rate
    Q_b = synthetic_load(time/3600.)

    # Load aggregation schemes
    ClaessonJaved = gt.load_aggregation.ClaessonJaved(dt, tmax)
    MLAA = gt.load_aggregation.MLAA(dt, tmax)
    Liu = gt.load_aggregation.Liu(dt, tmax)
    LoadAggSchemes = [ClaessonJaved, MLAA, Liu]
    loadAgg_labels = ['Claesson and Javed', 'MLAA', 'Liu']
    loadAgg_lines = ['b-', 'k--', 'r-.']

    # -------------------------------------------------------------------------
    # Calculate g-function
    # -------------------------------------------------------------------------

    # The field contains only one borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)
    # Evaluate the g-function on a geometrically expanding time grid
    time_gFunc = gt.utilities.time_geometric(dt, tmax, 50)
    # Calculate g-function
    gFunc = gt.gfunction.gFunction(
        borehole, alpha, time=time_gFunc, options=options)

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------
    nLoadAgg = len(LoadAggSchemes)
    T_b = np.zeros((nLoadAgg, Nt))

    t_calc = np.zeros(nLoadAgg)
    for n, (LoadAgg, label) in enumerate(zip(LoadAggSchemes, loadAgg_labels)):
        print(f'Simulation using {label} ...')
        # Interpolate g-function at required times
        time_req = LoadAgg.get_times_for_simulation()
        gFunc_int = interp1d(np.hstack([0., time_gFunc]),
                             np.hstack([0., gFunc.gFunc]),
                             kind='cubic',
                             bounds_error=False,
                             fill_value=(0., gFunc.gFunc[-1]))(time_req)
        # Initialize load aggregation scheme
        LoadAgg.initialize(gFunc_int / (2 * np.pi * k_s))

        tic = perf_counter()
        for i in range(Nt):
            # Increment time step by (1)
            LoadAgg.next_time_step(time[i])

            # Apply current load
            LoadAgg.set_current_load(Q_b[i]/H)

            # Evaluate borehole wall temperature
            deltaT_b = LoadAgg.temporal_superposition()
            T_b[n,i] = T_g - deltaT_b
        toc = perf_counter()
        t_calc[n] = toc - tic

    # -------------------------------------------------------------------------
    # Calculate exact solution from convolution in the Fourier domain
    # -------------------------------------------------------------------------

    # Heat extraction rate increment
    dQ = np.zeros(Nt)
    dQ[0] = Q_b[0]
    dQ[1:] = Q_b[1:] - Q_b[:-1]
    # Interpolated g-function
    g = interp1d(time_gFunc, gFunc.gFunc)(time)

    # Convolution in Fourier domain
    T_b_exact = T_g - fftconvolve(
        dQ, g / (2.0 * np.pi * k_s * H), mode='full')[0:Nt]

    # -------------------------------------------------------------------------
    # plot results
    # -------------------------------------------------------------------------

    # Configure figure and axes
    fig = gt.utilities._initialize_figure()

    ax1 = fig.add_subplot(311)
    # Axis labels
    ax1.set_xlabel(r'$t$ [hours]')
    ax1.set_ylabel(r'$Q_b$ [W]')
    gt.utilities._format_axes(ax1)
    hours = np.array([(j+1)*dt/3600. for j in range(Nt)])
    ax1.plot(hours, Q_b)

    ax2 = fig.add_subplot(312)
    # Axis labels
    ax2.set_xlabel(r'$t$ [hours]')
    ax2.set_ylabel(r'$T_b$ [degC]')
    gt.utilities._format_axes(ax2)
    for T_b_n, line, label in zip(T_b, loadAgg_lines, loadAgg_labels):
        ax2.plot(hours, T_b_n, line, label=label)
    ax2.plot(hours, T_b_exact, 'k.', label='exact')
    ax2.legend()

    ax3 = fig.add_subplot(313)
    # Axis labels
    ax3.set_xlabel(r'$t$ [hours]')
    ax3.set_ylabel(r'Error [degC]')
    gt.utilities._format_axes(ax3)
    for T_b_n, line, label in zip(T_b, loadAgg_lines, loadAgg_labels):
        ax3.plot(hours, T_b_n - T_b_exact, line, label=label)
    # Adjust to plot window
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Print performance metrics
    # -------------------------------------------------------------------------

    # Maximum errors in evaluation of borehole wall temperatures
    maxError = np.array([np.max(np.abs(T_b_n-T_b_exact)) for T_b_n in T_b])
    # Print results
    print('Simulation results')
    for label, maxError_n, t_calc_n in zip(loadAgg_labels, maxError, t_calc):
        print()
        print((f' {label} ').center(60, '-'))
        print(f'Maximum absolute error : {maxError_n:.3f} degC')
        print(f'Calculation time : {t_calc_n:.3f} sec')

    return


def synthetic_load(x):
    """
    Synthetic load profile of Bernier et al. (2004).

    Returns load y (in watts) at time x (in hours).
    """
    A = 2000.0
    B = 2190.0
    C = 80.0
    D = 2.0
    E = 0.01
    F = 0.0
    G = 0.95

    func = (168.0-C)/168.0
    for i in [1, 2, 3]:
        func += 1.0/(i*np.pi)*(np.cos(C*np.pi*i/84.0) - 1.0) \
                          *(np.sin(np.pi*i/84.0*(x - B)))
    func = func*A*np.sin(np.pi/12.0*(x - B)) \
        *np.sin(np.pi/4380.0*(x - B))

    y = func + (-1.0)**np.floor(D/8760.0*(x - B))*abs(func) \
        + E*(-1.0)**np.floor(D/8760.0*(x - B)) \
        /np.sign(np.cos(D*np.pi/4380.0*(x - F)) + G)
    return -y


# Main function
if __name__ == '__main__':
    main()
