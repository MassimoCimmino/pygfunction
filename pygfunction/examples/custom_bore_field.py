# -*- coding: utf-8 -*-
""" Example of definition of a bore field using custom borehole positions.

"""
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
    pos = [(0.0, 0.0),
           (0.0, 0.0),  # Duplicate (for example purposes)
           (0.03, 0.0),   # Duplicate (for example purposes)
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
    # Find and remove duplicates from borehole field
    # -------------------------------------------------------------------------

    field = gt.boreholes.remove_duplicates(field, disp=True)

    # -------------------------------------------------------------------------
    # Draw bore field
    # -------------------------------------------------------------------------

    gt.boreholes.visualize_field(field)

    return


# Main function
if __name__ == '__main__':
    main()
