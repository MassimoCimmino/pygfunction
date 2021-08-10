# -*- coding: utf-8 -*-
""" Example of calculation of g-functions with varied declaration of segments
    using uniform borehole wall temperature.

    The g-functions of a field of 6x4 boreholes is calculated with unequal
    number of segments.
"""

import pygfunction as gt


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    D = 4.0             # Borehole buried depth (m)
    H = 150.0           # Borehole length (m)
    r_b = 0.075         # Borehole radius (m)
    B = 7.5             # Borehole spacing (m)

    # Thermal properties
    alpha = 1.0e-6      # Ground thermal diffusivity (m2/s)

    # Geometrically expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt = 50                         # Number of time steps
    ts = H**2/(9.*alpha)            # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)

    # -------------------------------------------------------------------------
    # Borehole field
    # -------------------------------------------------------------------------

    # Field of 3x2 (n=6) boreholes
    N_1 = 6
    N_2 = 4
    boreField = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)

    # -------------------------------------------------------------------------
    # Evaluate g-functions with different segment options
    # -------------------------------------------------------------------------

    nSegments = 12  # number of segments toggle

    # g-Function calculation option for uniform borehole segment lengths
    # in the field by defining nSegments as an integer >= 1
    options = {'nSegments': nSegments, 'disp': True}
    gfunc = gt.gfunction.gFunction(
        boreField, alpha, time=time, options=options)

    print(gfunc.gFunc)

    # g-Function calculation with nSegments passed as list, where the list is
    # of the same length as boreField and each borehole is defined to have
    # >= 1 segment
    # Note: nSegments[i] pertains to boreField[i]
    options = {'nSegments': [nSegments] * len(boreField), 'disp': True,
               'profiles': True}
    gfunc = gt.gfunction.gFunction(
        boreField, alpha, time=time, options=options)

    print(gfunc.gFunc)

    gfunc.visualize_heat_extraction_rates(iBoreholes=[18, 12, 14])
    gfunc.visualize_heat_extraction_rate_profiles(iBoreholes=[14])

    # TODO: Possibly add UIFT calculation since UBHWT has equal bh wall temps
    gfunc.visualize_temperatures(iBoreholes=[18, 12, 14])
    gfunc.visualize_temperature_profiles(iBoreholes=[14])

    return


# Main function
if __name__ == '__main__':
    main()