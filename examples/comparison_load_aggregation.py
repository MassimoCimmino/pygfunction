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
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import os
from scipy.constants import pi
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
import sys
import time as tim

# Add path to pygfunction to Python path
packagePath = os.path.normpath(
        os.path.join(os.path.normpath(os.path.dirname(__file__)),
                     '..'))
sys.path.append(packagePath)

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

    # Number of segments per borehole
    nSegments = 12

    # Simulation parameters
    dt = 3600.                  # Time step (s)
    tmax = 20.*8760. * 3600.    # Maximum time (s)
    Nt = int(np.ceil(tmax/dt))  # Number of time steps

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
    boreField = [gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)]
    # Evaluate the g-function on a geometrically expanding time grid
    time_gFunc = gt.utilities.time_geometric(dt, tmax, 50)
    # Calculate g-function
    print('Calculation of the g-function ...')
    gFunc = gt.gfunction.uniform_temperature(boreField, time_gFunc, alpha,
                                             nSegments=nSegments)

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------
    nLoadAgg = len(LoadAggSchemes)
    T_b = np.zeros((nLoadAgg, Nt))
    Q = np.zeros(Nt)
    t_calc = np.zeros(nLoadAgg)
    for n in range(nLoadAgg):
        print('Simulation using {} ...'.format(loadAgg_labels[n]))
        # Select aggregation scheme
        LoadAgg = LoadAggSchemes[n]
        # Interpolate g-function at required times
        time_req = LoadAgg.get_times_for_simulation()
        gFunc_int = interp1d(np.hstack([0., time_gFunc]),
                             np.hstack([0., gFunc]),
                             kind='cubic',
                             bounds_error=False,
                             fill_value=(0., gFunc[-1]))(time_req)
        # Initialize load aggregation scheme
        LoadAgg.initialize(gFunc_int/(2*pi*k_s))

        tic = tim.time()
        time = 0.
        i = -1
        while time < tmax:
            # Increment time step by (1)
            time += dt
            i += 1
            LoadAgg.next_time_step(time)

            # Evaluate heat extraction rate
            Q[i] = synthetic_load(time/3600.)

            # Apply current load
            LoadAgg.set_current_load(Q[i]/H)

            # Evaluate borehole wall temeprature
            deltaT_b = LoadAgg.temporal_superposition()
            T_b[n,i] = T_g - deltaT_b
        toc = tim.time()
        t_calc[n] = toc - tic

    # -------------------------------------------------------------------------
    # Calculate exact solution from convolution in the Fourier domain
    # -------------------------------------------------------------------------

    # Heat extraction rate increment
    dQ = np.zeros(Nt)
    dQ[0] = Q[0]
    # Interpolated g-function
    time = np.array([(j+1)*dt for j in range(Nt)])
    g = interp1d(time_gFunc, gFunc)(time)
    for i in range(1, Nt):
        dQ[i] = Q[i] - Q[i-1]

    # Convolution in Fourier domain
    T_b_exact = T_g - fftconvolve(dQ, g/(2.0*pi*k_s*H), mode='full')[0:Nt]

    # -------------------------------------------------------------------------
    # plot results
    # -------------------------------------------------------------------------

    plt.rc('figure')
    fig = plt.figure()

    ax1 = fig.add_subplot(311)
    # Axis labels
    ax1.set_xlabel(r'$t$ (hours)')
    ax1.set_ylabel(r'$Q$ (W)')
    hours = np.array([(j+1)*dt/3600. for j in range(Nt)])
    ax1.plot(hours, Q, 'b-', lw=1.5)

    ax2 = fig.add_subplot(312)
    # Axis labels
    ax2.set_xlabel(r'$t$ (hours)')
    ax2.set_ylabel(r'$T_b$ (degC)')
    for n in range(nLoadAgg):
        ax2.plot(hours, T_b[n,:],
                 loadAgg_lines[n], lw=1.5, label=loadAgg_labels[n])
    ax2.plot(hours, T_b_exact, 'k.', lw=1.5, label='exact')
    ax2.legend()

    ax3 = fig.add_subplot(313)
    # Axis labels
    ax3.set_xlabel(r'$t$ (hours)')
    ax3.set_ylabel(r'Error (degC)')
    for n in range(nLoadAgg):
        ax3.plot(hours, T_b[n,:] - T_b_exact,
                 loadAgg_lines[n], lw=1.5, label=loadAgg_labels[n])
    # Show minor ticks
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    # Adjust to plot window
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Print performance metrics
    # -------------------------------------------------------------------------

    # Maximum errors in evaluation of borehole wall temperatures
    maxError = np.array([np.max(np.abs(T_b[n,:]-T_b_exact))
                         for n in range(nLoadAgg)])
    # Print results
    print('Simulation results')
    horizontalLine = '-'*66
    for n in range(nLoadAgg):
        print(horizontalLine)
        print(loadAgg_labels[n])
        print()
        print('Maximum absolute error : {} degC'.format(maxError[n]))
        print('Calculation time : {} seconds'.format(t_calc[n]))
        print()
    print(horizontalLine)

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
        func += 1.0/(i*pi)*(np.cos(C*pi*i/84.0) - 1.0) \
                          *(np.sin(pi*i/84.0*(x - B)))
    func = func*A*np.sin(pi/12.0*(x - B)) \
        *np.sin(pi/4380.0*(x - B))

    y = func + (-1.0)**np.floor(D/8760.0*(x - B))*abs(func) \
        + E*(-1.0)**np.floor(D/8760.0*(x - B)) \
        /np.sign(np.cos(D*pi/4380.0*(x - F)) + G)
    return -np.array([y])


# Main function
if __name__ == '__main__':
    main()
