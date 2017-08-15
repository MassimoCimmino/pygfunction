# -*- coding: utf-8 -*-
""" Example of calculation of g-functions using uniform heat extraction rates.

    The g-functions of fields of 3x2, 6x4 and 10x10 boreholes are calculated
    for boundary condition of uniform heat extraction rate along the boreholes,
    equal for all boreholes.

"""
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
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

    # Time vector
    # The inital time step is 100 hours (360000 seconds) and doubles every 6
    # time steps. The maximum time is 3000 years (9.4608 x 10^10 seconds).
    dt = 100*3600.                      # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    ts = H**2/(9.*alpha)            # Bore field characteristic time
    cells_per_level = 6
    time = []
    _width = []
    i = 0
    t_end = 0.
    while t_end < tmax:
        i += 1
        v = np.ceil(i / cells_per_level)
        width = 2.0**(v-1)
        t_end += width*float(dt)
        time.append(t_end)
        _width.append(width)
    time = np.array(time)

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

    plt.rc('figure', figsize=(80.0/25.4, 80.0*3.0/4.0/25.4))
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
        gfunc = gt.gfunction.uniform_heat_extraction(field, time, alpha)
        # Draw g-function
        ax1.plot(np.log(time/ts), gfunc, 'k-', lw=1.5)

    return


# Main function
if __name__ == '__main__':
    main()
