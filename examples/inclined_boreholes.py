# -*- coding: utf-8 -*-
""" Example of calculation of g-functions with inclined boreholes

    <<<Enter description of what fields are computed.>>>

"""

import pygfunction as gt
import numpy as np
import matplotlib.pyplot as plt


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    D = 4.0  # Borehole buried depth (m)
    # Borehole length (m)
    H = 150.
    r_b = 0.075  # Borehole radius (m)
    B = 7.5  # Borehole spacing (m)

    # Inclination parameters
    tilt = 15.  # Angle (in radians) from vertical of the axis of the borehole
    # Orientation is computed later in this example

    # Thermal properties
    alpha = 1.0e-6      # Ground thermal diffusivity (m2/s)

    # g-Function calculation options
    options = {'nSegments': 8, 'disp': True}

    # Geometrically expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt = 15                         # Number of time steps
    ts = H**2/(9.*alpha)            # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)
    lntts = np.log(time/ts)  # Unused in this script

    # ------------------------------------------------------------------------------------------------------------------
    # Custom tilted field creation
    # ------------------------------------------------------------------------------------------------------------------
    # Create a "diamond" or rhombus shaped field with each borehole tilting 15 degrees with an orientation based on the
    # cardinal direction. For example, the top borehole will orient north, the right borehole will orient east, etc.

    north = np.pi / 2.  # Northern orientation in radians
    east = 0.  # Eastern orientation in radians
    south = -np.pi / 2.  # Southern orientation in radians
    west = np.pi  # Western orientation in radians

    # A borehole field is a list of boreholes
    diamond_field = [gt.boreholes.Borehole(H, D, r_b, 0., B, tilt=tilt, orientation=north),
                     gt.boreholes.Borehole(H, D, r_b, B, 0., tilt=tilt, orientation=east),
                     gt.boreholes.Borehole(H, D, r_b, 0., -B, tilt=tilt, orientation=south),
                     gt.boreholes.Borehole(H, D, r_b, -B, 0., tilt=tilt, orientation=west)]

    # Visualize the borehole field
    fig = gt.boreholes.visualize_field(diamond_field)
    # fig.show() will show the plot in a temporary window
    # fig.savefig('diamond_field.png') will save a high resolution png to the current directory
    plt.close(fig)  # Closing the figure given that several more figures will be opened in the example

    # Compute a uniform borehole wall temperature g-function with the custom borehole field
    # NOTE: Inclined boreholes currently are only supported with the detailed solver method
    gfunc = gt.gfunction.gFunction(
        diamond_field, alpha, time=time, boundary_condition='UBWT', options=options, method='detailed')
    # Visualize the g-function
    fig = gfunc.visualize_g_function()
    plt.close(fig)

    # ------------------------------------------------------------------------------------------------------------------
    # Utilize pygfunction tilted field creation
    # ------------------------------------------------------------------------------------------------------------------
    # NOTE: The built in borehole field creation functions automatically compute orientation.

    # Rectangular field (2 x 2)
    rectangular_field = gt.boreholes.rectangle_field(2, 2, B, B, H, D, r_b, tilt=tilt)

    fig = gt.boreholes.visualize_field(rectangular_field)
    plt.close(fig)

    # Compute a uniform borehole wall temperature g-function for the rectangular borehole field
    gfunc = gt.gfunction.gFunction(
        rectangular_field, alpha, time=time, boundary_condition='UBWT', options=options, method='detailed')
    fig = gfunc.visualize_g_function()
    plt.close(fig)


# Main function
if __name__ == '__main__':
    main()
