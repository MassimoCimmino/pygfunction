# -*- coding: utf-8 -*-
""" Example of definition of a bore field using pre-defined configurations.

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
    rectangular_field = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)

    # Rectangular field triangular field of 4 x 3 borehole rows
    rectangular_field_triangular = gt.boreholes.rectangle_field_triangular(N_1, N_2, B, B, H, D, r_b, False)

    # Dense field triangular field of 4 x 3 borehole rows
    dense_field = gt.boreholes.dense_rectangle_field(N_1, N_2, B, H, D, r_b, False)

    # Box-shaped field of 4 x 3 boreholes
    box_field = gt.boreholes.box_shaped_field(N_1, N_2, B, B, H, D, r_b)

    # U-shaped field of 4 x 3 boreholes
    U_field = gt.boreholes.U_shaped_field(N_1, N_2, B, B, H, D, r_b)

    # L-shaped field of 4 x 3 boreholes
    L_field = gt.boreholes.L_shaped_field(N_1, N_2, B, B, H, D, r_b)

    # Circular field of 8 boreholes
    circle_field = gt.boreholes.circle_field(N_b, R, H, D, r_b)

    # -------------------------------------------------------------------------
    # Draw bore fields
    # -------------------------------------------------------------------------
    for field in [rectangular_field, rectangular_field_triangular, dense_field, box_field, U_field, L_field, circle_field]:
        gt.boreholes.visualize_field(field)
        import matplotlib.pyplot as plt
        plt.show()

    return


# Main function
if __name__ == '__main__':
    main()
