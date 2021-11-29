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
    r_out = 0.02        # Pipe outer radius (m)
    r_in = 0.015        # Pipe inner radius (m)
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
    m_flow_network = 10**np.arange(-2, 0.001, 0.05)
    # The fluid is propylene-glycol (20 %) at 20 degC
    fluid = gt.media.Fluid('MPG', 20.)
    cp_f = fluid.cp     # Fluid specific isobaric heat capacity (J/kg.K)
    rho_f = fluid.rho   # Fluid density (kg/m3)
    mu_f = fluid.mu     # Fluid dynamic viscosity (kg/m.s)
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
    R = np.zeros((nBoreholes, len(m_flow_network)))
    for i in range(nBoreholes):
        for j, m_flow_network_j in enumerate(m_flow_network):
            nBoreholes = i + 1
            # Boreholes are connected in series
            m_flow_borehole = m_flow_network_j
            # Boreholes are single U-tube
            m_flow_pipe = m_flow_borehole

            # Pipe thermal resistance
            R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
                    r_in, r_out, k_p)
            # Fluid to inner pipe wall thermal resistance (Single U-tube)
            h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
                    m_flow_pipe, r_in, mu_f, rho_f, k_f, cp_f, epsilon)
            R_f = 1.0/(h_f*2*pi*r_in)

            # Single U-tube, same for all boreholes in the bore field
            UTubes = []
            for borehole in boreField:
                SingleUTube = gt.pipes.SingleUTube(
                    pos_pipes, r_in, r_out, borehole, k_s, k_g, R_f + R_p)
                UTubes.append(SingleUTube)
            network = gt.networks.Network(
                boreField[:nBoreholes],
                UTubes[:nBoreholes],
                bore_connectivity=bore_connectivity[:nBoreholes])

            # Effective bore field thermal resistance
            R_field = gt.networks.network_thermal_resistance(
                network, m_flow_network_j, cp_f)
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
    ax1.plot(m_flow_network, R[0,:], '-', label='1 borehole')
    ax1.plot(m_flow_network, R[2,:], '--', label='3 boreholes')
    ax1.plot(m_flow_network, R[4,:], '-.', label='5 boreholes')
    ax1.legend()
    # Adjust to plot window
    plt.tight_layout()

    return


# Main function
if __name__ == '__main__':
    main()
