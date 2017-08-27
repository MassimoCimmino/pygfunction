# -*- coding: utf-8 -*-
""" Example of definition of a bore field using pre-defined configurations.

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
    B = 7.5             # Borehole spacing (m)
    N_1 = 4             # Number of boreholes in the x-direction (columns)
    N_2 = 3             # Number of boreholes in the y-direction (rows)

    # Circular field
    N_b = 8     # Number of boreholes
    R = 5.0     # Distance of the boreholes from the center of the field (m)

    # -------------------------------------------------------------------------
    # Borehole fields
    # -------------------------------------------------------------------------

    # Rectangular field of 4 x 3 boreholes
    rectangularField = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)

    # Box-shaped field of 4 x 3 boreholes
    boxField = gt.boreholes.box_shaped_field(N_1, N_2, B, B, H, D, r_b)

    # U-shaped field of 4 x 3 boreholes
    UField = gt.boreholes.U_shaped_field(N_1, N_2, B, B, H, D, r_b)

    # L-shaped field of 4 x 3 boreholes
    LField = gt.boreholes.L_shaped_field(N_1, N_2, B, B, H, D, r_b)

    # Circular field of 8 boreholes
    circleField = gt.boreholes.circle_field(N_b, R, H, D, r_b)

    # -------------------------------------------------------------------------
    # Draw bore fields
    # -------------------------------------------------------------------------
    LW = 1.5    # Line width

    for field in [rectangularField, boxField, UField, LField, circleField]:
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
