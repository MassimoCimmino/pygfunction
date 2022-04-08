# -*- coding: utf-8 -*-
""" Example of calculation of g-functions with inclined boreholes

    <<<Enter description of what fields are computed.>>>

"""

import pygfunction as gt
import numpy as np


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Number of boreholes
    nBoreholes = 5
    # Borehole dimensions
    D = 4.0  # Borehole buried depth (m)
    # Borehole length (m)
    H = 150.
    r_b = 0.075  # Borehole radius (m)
    B = 7.5  # Borehole spacing (m)

    # Pipe dimensions
    r_out = 0.02  # Pipe outer radius (m)
    r_in = 0.015  # Pipe inner radius (m)
    D_s = 0.05  # Shank spacing (m)
    epsilon = 1.0e-6  # Pipe roughness (m)

    # Pipe positions
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos_pipes = [(-D_s, 0.), (D_s, 0.)]

    # Ground properties
    k_s = 2.0  # Ground thermal conductivity (W/m.K)

    # Grout properties
    k_g = 1.0  # Grout thermal conductivity (W/m.K)

    # Pipe properties
    k_p = 0.4  # Pipe thermal conductivity (W/m.K)

    # Fluid properties
    # Total fluid mass flow rate per borehole (kg/s)
    m_flow_borehole = 1.0
    # The fluid is propylene-glycol (20 %) at 20 degC
    fluid = gt.media.Fluid('MPG', 20.)
    cp_f = fluid.cp  # Fluid specific isobaric heat capacity (J/kg.K)
    rho_f = fluid.rho  # Fluid density (kg/m3)
    mu_f = fluid.mu  # Fluid dynamic viscosity (kg/m.s)
    k_f = fluid.k  # Fluid thermal conductivity (W/m.K)


# Main function
if __name__ == '__main__':
    main()
