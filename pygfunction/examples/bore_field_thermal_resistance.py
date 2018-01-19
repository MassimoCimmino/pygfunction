# -*- coding: utf-8 -*-
""" Example of calculation of g-functions using mixed inlet temperatures.

    The g-functions of a field of 5 boreholes of different lengths connected
    in series are calculated for 2 boundary conditions: (a) uniform borehole
    wall temperature, and (b) series connections between boreholes. The
    g-function for case (b) is based on the effective borehole wall
    temperature, rather than the average borehole wall temperature.

"""
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy import pi

import pygfunction as gt


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    D = 4.0             # Borehole buried depth (m)
    # Borehole length (m)
    H_boreholes = np.array([75.0, 100.0, 125.0, 150.0, 75.0])
    r_b = 0.075         # Borehole radius (m)
    B = 7.5             # Borehole spacing (m)

    # Pipe dimensions
    rp_out = 0.02       # Pipe outer radius (m)
    rp_in = 0.015       # Pipe inner radius (m)
    D_s = 0.05          # Shank spacing (m)
    epsilon = 1.0e-6    # Pipe roughness (m)

    # Pipe positions
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos_pipes = [(-D_s, 0.), (D_s, 0.)]

    # Ground properties
    k_s = 2.0           # Ground thermal conductivity (W/m.K)

    # Grout properties
    k_g = 1.0           # Grout thermal conductivity (W/m.K)

    # Pipe properties
    k_p = 0.4           # Pipe thermal conductivity (W/m.K)

    # Fluid properties
    # Total fluid mass flow rate per borehole (kg/s)
    m_flow_boreholes = 10**np.arange(-2, 0.001, 0.1)
    cp_f = 4000.        # Fluid specific isobaric heat capacity (J/kg.K)
    den_f = 1015.       # Fluid density (kg/m3)
    visc_f = 0.002      # Fluid dynamic viscosity (kg/m.s)
    k_f = 0.5           # Fluid thermal conductivity (W/m.K)

    # -------------------------------------------------------------------------
    # Borehole field
    # -------------------------------------------------------------------------

    boreField = []
    bore_connectivity = []
    for i in range(len(H_boreholes)):
        H = H_boreholes[i]
        x = i*B
        borehole = gt.boreholes.Borehole(H, D, r_b, x, 0.)
        boreField.append(borehole)
        # Boreholes are connected in series: The index of the upstream
        # borehole is that of the previous borehole
        bore_connectivity.append(i - 1)

    # -------------------------------------------------------------------------
    # Initialize pipe model
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Evaluate the effective bore field thermal resistance
    # -------------------------------------------------------------------------

    R = np.zeros((len(H_boreholes), len(m_flow_boreholes)))
    for i in range(len(H_boreholes)):
        for j in range(len(m_flow_boreholes)):
            nBoreholes = i + 1
            m_flow = m_flow_boreholes[j]

            # Pipe thermal resistance
            R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(rp_in,
                                                                       rp_out,
                                                                       k_p)
            # Fluid to inner pipe wall thermal resistance (Single U-tube)
            h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
                    m_flow, rp_in, visc_f, den_f, k_f, cp_f, epsilon)
            R_f = 1.0/(h_f*2*pi*rp_in)

            # Single U-tube, same for all boreholes in the bore field
            UTubes = []
            for borehole in boreField:
                SingleUTube = gt.pipes.SingleUTube(pos_pipes, rp_in, rp_out,
                                                   borehole, k_s, k_g, R_f + R_p)
                UTubes.append(SingleUTube)

            R_field = gt.pipes.field_thermal_resistance(
                    UTubes[:nBoreholes], bore_connectivity[:nBoreholes],
                    m_flow, cp_f)
            R[i,j] = R_field

    # -------------------------------------------------------------------------
    # Plot bore field thermal resistances
    # -------------------------------------------------------------------------

    plt.rc('figure')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # Bore field thermal resistances
    ax1.plot(m_flow_boreholes, R[0,:], 'k-', lw=1.5, label='1 borehole')
    ax1.plot(m_flow_boreholes, R[2,:], 'r--', lw=1.5, label='3 boreholes')
    ax1.plot(m_flow_boreholes, R[4,:], 'b-.', lw=1.5, label='5 boreholes')
    ax1.legend()
    # Axis labels
    ax1.set_xlabel(r'$\dot{m}$ [kg/s]')
    ax1.set_ylabel(r'$R^*_{field}$ [m.K/W]')
    # Axis limits
    ax1.set_xlim([0., 1.])
    ax1.set_ylim([0., 1.])
    # Show minor ticks
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    # Adjust to plot window
    plt.tight_layout()

    return


# Main function
if __name__ == '__main__':
    main()
