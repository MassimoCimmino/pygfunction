# -*- coding: utf-8 -*-
""" Example of definition of a bore field using pre-defined configurations.

"""
import matplotlib.pyplot as plt

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
    rectangle_field = gt.borefield.Borefield.rectangle_field(
        N_1, N_2, B, B, H, D, r_b)

    # Rectangular field triangular field of 4 x 3 borehole rows
    staggered_rectangle_field = gt.borefield.Borefield.staggered_rectangle_field(
        N_1, N_2, B, B, H, D, r_b, False)

    # Dense field triangular field of 4 x 3 borehole rows
    dense_rectangle_field = gt.borefield.Borefield.dense_rectangle_field(
        N_1, N_2, B, H, D, r_b, False)

    # Box-shaped field of 4 x 3 boreholes
    box_shaped_field = gt.borefield.Borefield.box_shaped_field(
        N_1, N_2, B, B, H, D, r_b)

    # U-shaped field of 4 x 3 boreholes
    U_shaped_field = gt.borefield.Borefield.U_shaped_field(
        N_1, N_2, B, B, H, D, r_b)

    # L-shaped field of 4 x 3 boreholes
    L_shaped_field = gt.borefield.Borefield.L_shaped_field(
        N_1, N_2, B, B, H, D, r_b)

    # Circular field of 8 boreholes
    circle_field = gt.borefield.Borefield.circle_field(
        N_b, R, H, D, r_b)

    # -------------------------------------------------------------------------
    # Draw bore fields
    # -------------------------------------------------------------------------
    for field in [
            rectangle_field, staggered_rectangle_field, dense_rectangle_field,
            box_shaped_field, U_shaped_field, L_shaped_field, circle_field]:
        gt.boreholes.visualize_field(field)
        plt.show()

    return


# Main function
if __name__ == '__main__':
    main()
