# -*- coding: utf-8 -*-
""" Example of definition of a bore field using custom borehole positions.

"""
import numpy as np

import pygfunction as gt


def main():
    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    D = 4.0             # Borehole buried depth (m)
    H = 150.0           # Borehole length (m)
    r_b = 0.075         # Borehole radius (m)

    # Borehole positions
    # Note: Two duplicate boreholes have been added to this list of positions.
    #       Position 1 has a borehole that is directly on top of another bore
    #       Position 2 has a borehole with radius inside of another bore
    #       The duplicates will be removed with the remove_duplicates function
    x = np.array([0., 0., 0.03, 5., 3.5, 1., 5.5])
    y = np.array([0., 0., 0., 0., 4., 7., 5.5])

    # -------------------------------------------------------------------------
    # Borehole field
    # -------------------------------------------------------------------------

    # Build list of boreholes
    borefield = gt.borefield.Borefield(H, D, r_b, x, y)

    # -------------------------------------------------------------------------
    # Find and remove duplicates from borehole field
    # -------------------------------------------------------------------------
    borefield = gt.borefield.Borefield.from_boreholes(
        gt.boreholes.remove_duplicates(borefield, disp=True))

    # -------------------------------------------------------------------------
    # Draw bore field
    # -------------------------------------------------------------------------

    borefield.visualize_field(borefield)

    return


# Main function
if __name__ == '__main__':
    main()
