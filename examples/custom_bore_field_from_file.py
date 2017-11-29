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

    # Filepath to bore field text file
    filename = './data/custom_field_32_boreholes.txt'

    # -------------------------------------------------------------------------
    # Borehole field
    # -------------------------------------------------------------------------

    # Build list of boreholes
    field = gt.boreholes.field_from_file(filename)

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
