# -*- coding: utf-8 -*-
""" Example of definition of a bore field using custom borehole positions.

"""
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
    borefield = gt.borefield.Borefield.from_file(filename)

    # -------------------------------------------------------------------------
    # Draw bore field
    # -------------------------------------------------------------------------

    gt.boreholes.visualize_field(borefield)

    return


# Main function
if __name__ == '__main__':
    main()
