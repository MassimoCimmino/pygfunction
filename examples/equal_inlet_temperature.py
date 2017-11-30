# -*- coding: utf-8 -*-
""" Example of calculation of g-functions using equal inlet temperatures.

    The g-functions of a field of 6x4 boreholes are calculated for boundary
    conditions of (a) uniform heat extraction rate, equal for all boreholes,
    (b) uniform borehole wall temperature along the boreholes, equal for all
    boreholes, and (c) equal inlet fluid temperature into all boreholes.

"""
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import os, sys
from scipy import pi

# Add path to pygfunction to Python path
packagePath = os.path.normpath(
        os.path.join(os.path.normpath(os.path.dirname(__file__)),
                     '..'))
sys.path.append(packagePath)

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

    # Pipe dimensions
    rp_out = 0.0211     # Pipe outer radius (m)
    rp_in = 0.0147      # Pipe inner radius (m)
    D_s = 0.052         # Shank spacing (m)
    epsilon = 1.0e-6    # Pipe roughness (m)

    # Pipe positions
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos_pipes = [(-D_s, 0.), (D_s, 0.)]

    # Ground properties
    alpha = 1.0e-6      # Ground thermal diffusivity (m2/s)
    k_s = 2.0           # Ground thermal conductivity (W/m.K)

    # Grout properties
    k_g = 1.0           # Grout thermal conductivity (W/m.K)

    # Pipe properties
    k_p = 0.4           # Pipe thermal conductivity (W/m.K)

    # Fluid properties
    m_flow = 0.25       # Total fluid mass flow rate per borehole (kg/s)
    cp_f = 3977.        # Fluid specific isobaric heat capacity (J/kg.K)
    den_f = 1015.       # Fluid density (kg/m3)
    visc_f = 0.00203    # Fluid dynamic viscosity (kg/m.s)
    k_f = 0.492         # Fluid thermal conductivity (W/m.K)

    # Number of segments per borehole
    nSegments = 12

    # Geometrically expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt = 50                         # Number of time steps
    ts = H**2/(9.*alpha)            # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)

    # -------------------------------------------------------------------------
    # Borehole field
    # -------------------------------------------------------------------------

    # Field of 6x4 (n=24) boreholes
    N_1 = 6
    N_2 = 4
    boreField = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)

    # -------------------------------------------------------------------------
    # Initialize pipe model
    # -------------------------------------------------------------------------

    # Pipe thermal resistance
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(rp_in,
                                                               rp_out,
                                                               k_p)
    # Fluid to inner pipe wall thermal resistance (Single U-tube)
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(m_flow,
                                                                      rp_in,
                                                                      visc_f,
                                                                      den_f,
                                                                      k_f,
                                                                      cp_f,
                                                                      epsilon)
    R_f = 1.0/(h_f*2*pi*rp_in)

    # Single U-tube, same for all boreholes in the bore field
    UTubes = []
    for borehole in boreField:
        SingleUTube = gt.pipes.SingleUTube(pos_pipes, rp_in, rp_out,
                                           borehole, k_s, k_g, R_f + R_p)
        UTubes.append(SingleUTube)

    # -------------------------------------------------------------------------
    # Evaluate the g-functions for the borefield
    # -------------------------------------------------------------------------

    # Calculate the g-function for uniform heat extraction rate
    gfunc_uniform_Q = gt.gfunction.uniform_heat_extraction(
            boreField, time, alpha, disp=True)

    # Calculate the g-function for uniform borehole wall temperature
    gfunc_uniform_T = gt.gfunction.uniform_temperature(
            boreField, time, alpha, nSegments=nSegments, disp=True)

    # Calculate the g-function for equal inlet fluid temperature
    gfunc_equal_Tf_in = gt.gfunction.equal_inlet_temperature(
            boreField, UTubes, m_flow, cp_f, time, alpha,
            nSegments=nSegments, disp=True)

    # -------------------------------------------------------------------------
    # Plot g-functions
    # -------------------------------------------------------------------------

    plt.rc('figure')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # g-functions
    ax1.plot(np.log(time/ts), gfunc_uniform_Q,
             'k-', lw=1.5, label='Uniform heat extraction rate')
    ax1.plot(np.log(time/ts), gfunc_uniform_T,
             'b--', lw=1.5, label='Uniform borehole wall temperature')
    ax1.plot(np.log(time/ts), gfunc_equal_Tf_in,
             'r-.', lw=1.5, label='Equal inlet temperature')
    ax1.legend()
    # Axis labels
    ax1.set_xlabel(r'$ln(t/t_s)$')
    ax1.set_ylabel(r'$g(t/t_s)$')
    # Axis limits
    ax1.set_xlim([-10.0, 5.0])
    ax1.set_ylim([0., 50.])
    # Show minor ticks
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    # Adjust to plot window
    plt.tight_layout()

    return


# Main function
if __name__ == '__main__':
    main()
