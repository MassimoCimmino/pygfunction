# -*- coding: utf-8 -*-
""" Example of definition of a bore field using custom borehole positions.

"""
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
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
    # Parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    D = 4.0             # Borehole buried depth (m)
    H = 150.0           # Borehole length (m)
    r_b = 0.075         # Borehole radius (m)
    N_1 = 4             # Number of boreholes in the x-direction (columns)
    N_2 = 3             # Number of boreholes in the y-direction (rows)

    # Borehole positions
    pos = [(0.0, 0.0),
           (5.0, 0.),
           (3.5, 4.0),
           (1.0, 7.0),
           (5.5, 5.5)]

    # -------------------------------------------------------------------------
    # Borehole field
    # -------------------------------------------------------------------------

    # Build list of boreholes
    field = [gt.boreholes.Borehole(H, D, r_b, x, y) for (x, y) in pos]

    # -------------------------------------------------------------------------
    # Draw bore field
    # -------------------------------------------------------------------------
    LW = 1.5    # Line width

    i = 0   # Initialize borehole index
    # Initialize figure
    plt.rc('figure', figsize=(90.0/25.4, 90.0*4.0/4.0/25.4))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bbox_props = dict(boxstyle="circle,pad=0.3", fc="white", ec="b", lw=LW)

    for borehole in field:
        i += 1  # Increment borehole index
        (x, y) = borehole.position()    # Extract borehole position
        # Add current borehole to the figure
        ax.plot(x, y, 'k.')
        ax.text(x, y, i, ha="center", va="center", size=9, bbox=bbox_props)

    # Configure figure axes
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.axis('equal')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.tight_layout()

    return


# Main function
if __name__ == '__main__':
    main()
