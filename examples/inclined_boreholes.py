# -*- coding: utf-8 -*-
""" Example of calculation of g-functions with inclined boreholesusing uniform
    and equal borehole wall temperatures.

    The g-functions of two fields of 8 boreholes are calculated for boundary
    condition of uniform borehole wall temperature along the boreholes, equal
    for all boreholes. The first field corresponds to the "optimum"
    configuration presented by Claesson and Eskilson (1987) and the second
    field corresponds to the configuration comprised of 8 boreholes in a
    circle.

    Claesson J, and Eskilson, P. (1987). Conductive heat extraction by
    thermally interacting deep boreholes, in "Thermal analysis of heat
    extraction boreholes". Ph.D. Thesis, University of Lund, Lund, Sweden.
"""

import numpy as np

import pygfunction as gt


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    D = 4.0             # Borehole buried depth (m)
    H = 150.0           # Borehole length (m)
    r_b = 0.075         # Borehole radius (m)
    tilt = np.radians(20.)  # Borehole inclination (rad)

    # Thermal properties
    alpha = 1.0e-6      # Ground thermal diffusivity (m2/s)

    # g-Function calculation options
    options = {'disp': True}

    # Geometrically expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt = 25                         # Number of time steps
    ts = H**2/(9.*alpha)            # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)
    lntts = np.log(time/ts)

    # -------------------------------------------------------------------------
    # Borehole fields
    # -------------------------------------------------------------------------
    """
    Bore field #1

    This field corresponds to the optimal configuration presented by
    Claesson and Eskilson (1987). The field is built using the `cardinal_point`
    function to define the orientation of each borehole, individually.
    """
    B = 7.5 # Borehole spacing (m)
    # Orientation of the boreholes
    borehole_orientations = [
        gt.utilities.cardinal_point('W'),
        gt.utilities.cardinal_point('NW'),
        gt.utilities.cardinal_point('SW'),
        gt.utilities.cardinal_point('N'),
        gt.utilities.cardinal_point('S'),
        gt.utilities.cardinal_point('NE'),
        gt.utilities.cardinal_point('SE'),
        gt.utilities.cardinal_point('E')]

    # "Optimal" field of 8 boreholes
    boreholes = []
    for i, orientation in enumerate(borehole_orientations):
        borehole = gt.boreholes.Borehole(
            H, D, r_b, i * B, 0., tilt=tilt, orientation=orientation)
        boreholes.append(borehole)
    borefield1 = gt.borefield.Borefield.from_boreholes(boreholes)

    # Visualize the borehole field
    fig1 = gt.boreholes.visualize_field(borefield1)

    """
    Bore field #2

    This field corresponds to the configuration comprised of 8 boreholes in
    a circle presented by Claesson and Eskilson (1987). The field is built
    using the `circle_field` function.
    """
    N = 8   # Number of boreholes
    R = 3.  # Borehole spacing from the center of the field (m)

    # Field of 6 boreholes in a circle
    borefield2 = gt.borefield.Borefield.circle_field(
        N, R, H, D, r_b, tilt=tilt)

    # Visualize the borehole field
    fig2 = gt.boreholes.visualize_field(borefield2)

    # -------------------------------------------------------------------------
    # Evaluate g-functions for all fields
    # -------------------------------------------------------------------------
    # Bore field #1
    gfunc1 = gt.gfunction.gFunction(
        borefield1, alpha, time=time, options=options, method='similarities')
    fig3 = gfunc1.visualize_g_function()
    fig3.suptitle('"Optimal" field of 8 boreholes')
    fig3.tight_layout()
    # Bore field #2
    gfunc2 = gt.gfunction.gFunction(
        borefield2, alpha, time=time, options=options, method='similarities')
    fig4 = gfunc2.visualize_g_function()
    fig4.suptitle(f'Field of {N} boreholes in a circle')
    fig4.tight_layout()


# Main function
if __name__ == '__main__':
    main()
