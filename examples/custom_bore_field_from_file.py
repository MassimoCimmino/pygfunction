# -*- coding: utf-8 -*-
""" Example of definition of a bore field using custom borehole positions.

"""
import os

import pygfunction as gt


def main():
    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------

    # File path to bore field text file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'data', 'custom_field_32_boreholes.txt')

    # -------------------------------------------------------------------------
    # Borehole field
    # -------------------------------------------------------------------------

    # Build list of boreholes
    borefield = gt.borefield.Borefield.from_file(file_path)

    # -------------------------------------------------------------------------
    # Draw bore field
    # -------------------------------------------------------------------------

    borefield.visualize_field()

    return


# Main function
if __name__ == '__main__':
    main()
