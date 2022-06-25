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
    tilt = 20. * np.pi / 180.  # Angle (in radians) from vertical of the axis of the borehole
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
    """
    This field is an optimal configuration discussed in Claesson and Eskilson (1987).
    
    Claesson, J. and Eskilson, P. (1987). Conductive Heat Extraction by Thermally Interacting Deep Boreholes. Lund 
    Institute of Technology, Lund, Sweden. PhD Thesis Chapter 3.   
    """
    # Make a list of angles utilized in Figure 5J on page 22 going from left to right.
    west = gt.utilities.cardinal_point('W')  # Western orientation in radians
    northwest = gt.utilities.cardinal_point('NW')
    southwest = gt.utilities.cardinal_point('SW')
    north = gt.utilities.cardinal_point('N')  # Northern orientation in radians
    south = gt.utilities.cardinal_point('S')  # Southern orientation in radians
    northeast = gt.utilities.cardinal_point('NE')
    southeast = gt.utilities.cardinal_point('SE')
    east = gt.utilities.cardinal_point('E')  # Eastern orientation in radians
    bore_orientations = [west, northwest, southwest, north, south, northeast, southeast, east]

    optimal_field = []
    for i, orientation in enumerate(bore_orientations):
        borehole = gt.boreholes.Borehole(H, D, r_b, float(i) * B, 0., tilt=tilt, orientation=orientation)
        optimal_field.append(borehole)

    # Visualize the borehole field
    fig = gt.boreholes.visualize_field(optimal_field)
    # fig.show() will show the plot in a temporary window
    # fig.savefig('diamond_field.png') will save a high resolution png to the current directory
    plt.close(fig)  # Closing the figure given that several more figures will be opened in the example

    # Compute a uniform borehole wall temperature g-function with the custom borehole field
    # NOTE: Inclined boreholes currently are only supported with the detailed and similarities solver methods.
    gfunc = gt.gfunction.gFunction(
        optimal_field, alpha, time=time, boundary_condition='UBWT', options=options, method='similarities')
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
