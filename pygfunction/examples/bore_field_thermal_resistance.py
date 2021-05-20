# -*- coding: utf-8 -*-
""" Example of calculation of effective bore field thermal resistance.

    The effective bore field thermal resistance of fields of up to 5 boreholes
    of equal lengths connected in series is calculated for various fluid flow
    rates.

"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from scipy import pi

import pygfunction as gt


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Number of boreholes
    nBoreholes = 5
    # Borehole dimensions
    D = 4.0             # Borehole buried depth (m)
    # Borehole length (m)
    H = 150.
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
    # Total fluid mass flow rate per borehole (kg/s), from 0.01 kg/s to 1 kg/s
    m_flow_boreholes = 10**np.arange(-2, 0.001, 0.05)
    # The fluid is propylene-glycol (20 %) at 20 degC
    fluid = gt.media.Fluid('MPG', 20.)
    cp_f = fluid.cp     # Fluid specific isobaric heat capacity (J/kg.K)
    den_f = fluid.rho   # Fluid density (kg/m3)
    visc_f = fluid.mu   # Fluid dynamic viscosity (kg/m.s)
    k_f = fluid.k       # Fluid thermal conductivity (W/m.K)

    # -------------------------------------------------------------------------
    # Borehole field
    # -------------------------------------------------------------------------

    boreField = []
    bore_connectivity = []
    for i in range(nBoreholes):
        x = i*B
        borehole = gt.boreholes.Borehole(H, D, r_b, x, 0.)
        boreField.append(borehole)
        # Boreholes are connected in series: The index of the upstream
        # borehole is that of the previous borehole
        bore_connectivity.append(i - 1)

    # -------------------------------------------------------------------------
    # Evaluate the effective bore field thermal resistance
    # -------------------------------------------------------------------------

    # Initialize result array
    R = np.zeros((nBoreholes, len(m_flow_boreholes)))
    for i in range(nBoreholes):
        for j in range(len(m_flow_boreholes)):
            nBoreholes = i + 1
            m_flow = m_flow_boreholes[j]

            # Pipe thermal resistance
            R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
                    rp_in, rp_out, k_p)
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
            network = gt.networks.Network(
                boreField[:nBoreholes],
                UTubes[:nBoreholes],
                bore_connectivity=bore_connectivity[:nBoreholes])

            # Effective bore field thermal resistance
            R_field = gt.networks.network_thermal_resistance(
                network, m_flow, cp_f)
            # Add to result array
            R[i,j] = R_field

    # -------------------------------------------------------------------------
    # Plot bore field thermal resistances
    # -------------------------------------------------------------------------

    # Configure figure and axes
    fig = gt.utilities._initialize_figure()

    ax1 = fig.add_subplot(111)
    # Axis labels
    ax1.set_xlabel(r'$\dot{m}$ [kg/s]')
    ax1.set_ylabel(r'$R^*_{field}$ [m.K/W]')
    # Axis limits
    ax1.set_xlim([0., 1.])
    ax1.set_ylim([0., 1.])

    gt.utilities._format_axes(ax1)

    # Bore field thermal resistances
    ax1.plot(m_flow_boreholes, R[0,:], '-', label='1 borehole')
    ax1.plot(m_flow_boreholes, R[2,:], '--', label='3 boreholes')
    ax1.plot(m_flow_boreholes, R[4,:], '-.', label='5 boreholes')
    ax1.legend()
    # Adjust to plot window
    plt.tight_layout()

    return


# Main function
if __name__ == '__main__':
    main()
