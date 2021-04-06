# -*- coding: utf-8 -*-
""" Example of definition of a bore field using custom borehole positions.

"""
from __future__ import absolute_import, division, print_function

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
    pos = [(0.0, 0.0),
           (0.0, 0.0),  # Note: a duplicate borehole has been intentionally added for example purposes
           (0.03, 0.0),  # Note: a borehole within a borehole radius of another borehole has been added for example
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

    print('The number of boreholes defined: {}'.format(len(field)))
    duplicate_pairs = gt.boreholes.check_duplicates(field)
    print('Duplicate pairs found in the field: {}'.format(len(duplicate_pairs)))
    print(duplicate_pairs)
    field = gt.boreholes.remove_duplicates(field, duplicate_pairs)
    print('Duplicate pairs found in the new field:')
    print(gt.boreholes.check_duplicates(field))
    print('The number of unique boreholes left: {}'.format(len(field)))

    # -------------------------------------------------------------------------
    # Draw bore field
    # -------------------------------------------------------------------------

    gt.boreholes.visualize_field(field)

    return


# Main function
if __name__ == '__main__':
    main()
