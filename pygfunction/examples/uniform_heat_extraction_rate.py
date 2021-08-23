# -*- coding: utf-8 -*-
""" Example of calculation of g-functions using uniform heat extraction rates.

    The g-functions of fields of 3x2, 6x4 and 10x10 boreholes are calculated
    for boundary condition of uniform heat extraction rate along the boreholes,
    equal for all boreholes.

"""
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

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

    # Path to validation data
    filePath = './data/CiBe14_uniform_heat_extraction_rate.txt'

    # g-Function calculation options
    # The second field is evaluated with more segments to draw the
    # temperature profiles
    options = [{'nSegments':1, 'disp':True, 'profiles':True},
               {'nSegments':12, 'disp':True, 'profiles':True},
               {'nSegments':1, 'disp':True, 'profiles':True}]

    # Geometrically expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt = 50                         # Number of time steps
    ts = H**2/(9.*alpha)            # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)

    # -------------------------------------------------------------------------
    # Borehole fields
    # -------------------------------------------------------------------------

    # Field of 3x2 (n=6) boreholes
    N_1 = 3
    N_2 = 2
    boreField1 = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)

    # Field of 6x4 (n=24) boreholes
    N_1 = 6
    N_2 = 4
    boreField2 = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)

    # Field of 10x10 (n=100) boreholes
    N_1 = 10
    N_2 = 10
    boreField3 = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)

    # -------------------------------------------------------------------------
    # Load data from Cimmino and Bernier (2014)
    # -------------------------------------------------------------------------
    data = np.loadtxt(filePath, skiprows=55)

    # -------------------------------------------------------------------------
    # Evaluate g-functions for all fields
    # -------------------------------------------------------------------------
    i = 0
    for field in [boreField1, boreField2, boreField3]:
        gfunc = gt.gfunction.gFunction(
            field, alpha, time=time, boundary_condition='UHTR',
            options=options[i])
        # Draw g-function
        ax = gfunc.visualize_g_function().axes[0]
        # Draw reference g-function
        ax.plot(data[:,0], data[:,i+1], 'bx')
        ax.legend(['pygfunction', 'Cimmino and Bernier (2014)'])
        ax.set_title('Field of {} boreholes'.format(len(field)))
        plt.tight_layout()
        i += 1

        # For the second borefield, draw the evolution of heat extraction rates
        if i == 2:
            gfunc.visualize_temperatures(iBoreholes=[18, 12, 14])
            gfunc.visualize_temperature_profiles(iBoreholes=[14])

    return


# Main function
if __name__ == '__main__':
    main()
