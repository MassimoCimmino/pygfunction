# -*- coding: utf-8 -*-
""" Example of calculation of g-functions using uniform heat extraction rates.

    The g-functions of fields of 3x2, 6x4 and 10x10 boreholes are calculated
    for boundary condition of uniform heat extraction rate along the boreholes,
    equal for all boreholes.

"""
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import os
import sys

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
    B = 7.5             # Borehole spacing (m)

    # Thermal properties
    alpha = 1.0e-6      # Ground thermal diffusivity (m2/s)

    # Path to validation data
    filePath = './data/CiBe14_uniform_heat_extraction_rate.txt'

    # Geometrically expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt = 50                         # Number of time steps
    ts = H**2/(9.*alpha)            # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)

    # -------------------------------------------------------------------------
    # Borehole fields
    # -------------------------------------------------------------------------

    # Field of 3x2 (n=6) boreholes
    N_1 = 3
    N_2 = 2
    boreField1 = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)

    # Field of 6x4 (n=24) boreholes
    N_1 = 6
    N_2 = 4
    boreField2 = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)

    # Field of 10x10 (n=100) boreholes
    N_1 = 10
    N_2 = 10
    boreField3 = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)

    # -------------------------------------------------------------------------
    # Initialize figure
    # -------------------------------------------------------------------------

    plt.rc('figure')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # Axis labels
    ax1.set_xlabel(r'$ln(t/t_s)$')
    ax1.set_ylabel(r'$g(t/t_s)$')
    # Axis limits
    ax1.set_xlim([-10.0, 5.0])
    ax1.set_ylim([0., 100.])
    # Show minor ticks
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    # Adjust to plot window
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Evaluate g-functions for all fields
    # -------------------------------------------------------------------------
    for field in [boreField1, boreField2, boreField3]:
        # Calculate g-function
        gfunc = gt.gfunction.uniform_heat_extraction(field, time, alpha,
                                                     disp=True)
        # Draw g-function
        ax1.plot(np.log(time/ts), gfunc, 'k-', lw=1.5)
    calculated = mlines.Line2D([], [],
                               color='black',
                               lw=1.5,
                               label='pygfunction')

    # -------------------------------------------------------------------------
    # Load data from Cimmino and Bernier (2014)
    # -------------------------------------------------------------------------
    data = np.loadtxt(filePath, skiprows=55)
    for i in range(3):
        ax1.plot(data[:,0], data[:,i+1], 'bx', lw=1.5)
    reference = mlines.Line2D([], [],
                              color='blue',
                              ls='None',
                              lw=1.5,
                              marker='x',
                              label='Cimmino and Bernier (2014)')
    ax1.legend(handles=[calculated, reference], loc='upper left')

    return


# Main function
if __name__ == '__main__':
    main()
