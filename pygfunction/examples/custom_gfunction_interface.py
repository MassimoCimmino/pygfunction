# -*- coding: utf-8 -*-
""" An example showing how to make use of the common g-function interface via
    - dictionary access
    - listing arguments

"""

import pygfunction as gt
import matplotlib.pyplot as plt
import numpy as np


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    D = 4.0  # Borehole buried depth (m)
    H = 150.0  # Borehole length (m)
    r_b = 0.075  # Borehole radius (m)
    B = 7.5  # Borehole spacing (m)

    # Thermal properties
    alpha = 1.0e-6  # Ground thermal diffusivity (m2/s)

    # Geometrically expanding time vector.
    dt = 100 * 3600.  # Time step
    years = 3000.
    tmax = years * 8760. * 3600.  # Maximum time
    Nt = 50  # Number of time steps
    ts = H ** 2 / (9. * alpha)  # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)
    ln_tts = np.log(time / ts)  # ln(t / ts) for plotting versus g-function later

    # Thermal properties
    alpha = 1.0e-6  # Ground thermal diffusivity (m2/s)

    # -------------------------------------------------------------------------
    # Borehole fields
    # -------------------------------------------------------------------------

    # Field of 3x2 (n=6) boreholes
    N_1 = 3
    N_2 = 2
    boreField = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)

    # compute a g-function for uniform heat flux by listing the inputs
    g_interface = gt.gfunction.gFunction(boreholes=boreField, time=time, alpha=alpha, disp=True)

    g_UHF = g_interface.compute_g_function('UHF')

    # compute a g-function for uniform borehole wall temperature by use of dictionary
    nSegments = 12  # provide a number of segments
    method = 'linear'  # linear interpolation for h_dt thermal response factors

    inputs_UBHWT = {'boreholes': boreField, 'time': time, 'alpha': alpha, 'nSegments': nSegments, 'method': method,
                    'disp': True}

    g_interface = gt.gfunction.gFunction(**inputs_UBHWT)

    g_UBHWT = g_interface.compute_g_function('UBHWT')

    # plot different responses on the same figure
    fig, ax = plt.subplots()

    ax.plot(ln_tts, g_UHF, label='UHF', ls='--')
    ax.plot(ln_tts, g_UBHWT, label='UBHWT', ls='-.')

    ax.set_ylabel('g')
    ax.set_xlabel('ln(t/t$_s$)')

    fig.savefig('g_functions.jpg')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
