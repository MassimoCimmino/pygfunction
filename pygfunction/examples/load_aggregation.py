# -*- coding: utf-8 -*-
""" Example of simulation of a geothermal system.

    The g-function of a single borehole is calculated for boundary condition of
    uniform borehole wall temperature along the borehole. Then, the borehole
    wall temperature variations resulting from a time-varying load profile
    are simulated using the aggregation method of Claesson and Javed (2012).

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import pi
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

    # Number of segments per borehole
    nSegments = 12

    # Simulation parameters
    dt = 3600.                  # Time step (s)
    tmax = 20.*8760. * 3600.    # Maximum time (s)
    Nt = int(np.ceil(tmax/dt))  # Number of time steps
    time = dt * np.arange(1, Nt+1)

    # Evaluate heat extraction rate
    Q_b = synthetic_load(time/3600.)

    # Load aggregation scheme
    LoadAgg = gt.load_aggregation.ClaessonJaved(dt, tmax)

    # -------------------------------------------------------------------------
    # Calculate g-function
    # -------------------------------------------------------------------------

    # The field contains only one borehole
    boreField = [gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)]
    # Get time values needed for g-function evaluation
    time_req = LoadAgg.get_times_for_simulation()
    # Calculate g-function
    gFunc = gt.gfunction.uniform_temperature(boreField, time_req, alpha,
                                             nSegments=nSegments)
    # Initialize load aggregation scheme
    LoadAgg.initialize(gFunc/(2*pi*k_s))

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    T_b = np.zeros(Nt)
    for i in range(Nt):
        # Increment time step by (1)
        LoadAgg.next_time_step(time[i])

        # Apply current load
        LoadAgg.set_current_load(Q_b[i]/H)

        # Evaluate borehole wall temeprature
        deltaT_b = LoadAgg.temporal_superposition()
        T_b[i] = T_g - deltaT_b

    # -------------------------------------------------------------------------
    # Calculate exact solution from convolution in the Fourier domain
    # -------------------------------------------------------------------------

    # Heat extraction rate increment
    dQ = np.zeros(Nt)
    dQ[0] = Q_b[0]
    # Interpolated g-function
    g = interp1d(time_req, gFunc)(time)
    for i in range(1, Nt):
        dQ[i] = Q_b[i] - Q_b[i-1]

    # Convolution in Fourier domain
    T_b_exact = T_g - fftconvolve(dQ, g/(2.0*pi*k_s*H), mode='full')[0:Nt]

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

    ax2.plot(hours, T_b)
    ax2.plot(hours, T_b_exact, 'k.')

    ax3 = fig.add_subplot(313)
    # Axis labels
    ax3.set_xlabel(r'$t$ [hours]')
    ax3.set_ylabel(r'Error [degC]')
    gt.utilities._format_axes(ax3)

    ax3.plot(hours, T_b - T_b_exact)

    # Adjust to plot window
    plt.tight_layout()

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
    for i in [1,2,3]:
        func += 1.0/(i*pi)*(np.cos(C*pi*i/84.0)-1.0) \
                          *(np.sin(pi*i/84.0*(x-B)))
    func = func*A*np.sin(pi/12.0*(x-B)) \
           *np.sin(pi/4380.0*(x-B))

    y = func + (-1.0)**np.floor(D/8760.0*(x-B))*abs(func) \
      + E*(-1.0)**np.floor(D/8760.0*(x-B))/np.sign(np.cos(D*pi/4380.0*(x-F))+G)
    return -y


# Main function
if __name__ == '__main__':
    main()
