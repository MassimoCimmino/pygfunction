# -*- coding: utf-8 -*-
""" Example of calculation of g-functions using mixed inlet temperatures.

    The g-functions of a field of 5 boreholes of different lengths connected
    in series are calculated for 2 boundary conditions: (a) uniform borehole
    wall temperature, and (b) series connections between boreholes. The
    g-function for case (b) is based on the effective borehole wall
    temperature, rather than the average borehole wall temperature.

"""
import matplotlib.pyplot as plt
import numpy as np

import pygfunction as gt


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    D = 4.0             # Borehole buried depth (m)
    # Borehole length (m)
    H_boreholes = np.array([75.0, 100.0, 125.0, 150.0, 75.0])
    H_mean = np.mean(H_boreholes)
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
    alpha = 1.0e-6      # Ground thermal diffusivity (m2/s)
    k_s = 2.0           # Ground thermal conductivity (W/m.K)

    # Grout properties
    k_g = 1.0           # Grout thermal conductivity (W/m.K)

    # Pipe properties
    k_p = 0.4           # Pipe thermal conductivity (W/m.K)

    # Fluid properties
    # Total fluid mass flow rate in network (kg/s)
    m_flow_network = np.array([-0.25, 0.5])   
    # All boreholes are in series
    m_flow_borehole = m_flow_network
    # The fluid is propylene-glycol (20 %) at 20 degC
    fluid = gt.media.Fluid('MPG', 20.)
    cp_f = fluid.cp     # Fluid specific isobaric heat capacity (J/kg.K)
    rho_f = fluid.rho   # Fluid density (kg/m3)
    mu_f = fluid.mu     # Fluid dynamic viscosity (kg/m.s)
    k_f = fluid.k       # Fluid thermal conductivity (W/m.K)

    # g-Function calculation options
    nSegments = 8
    options = {'nSegments': nSegments,
               'disp': True,
               'profiles': True}
    # The similarities method is used since the 'equivalent' method does not
    # apply if boreholes are connected in series
    method = 'similarities'

    # Geometrically expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt = 25                         # Number of time steps
    ts = H_mean**2/(9.*alpha)   # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)

    # -------------------------------------------------------------------------
    # Borehole field
    # -------------------------------------------------------------------------

    nBoreholes = len(H_boreholes)
    x = np.arange(nBoreholes) * B
    borefield = gt.borefield.Borefield(H_boreholes, D, r_b, x, 0.)
    # Boreholes are connected in series: The index of the upstream
    # borehole is that of the previous borehole
    bore_connectivity = [i - 1 for i in range(nBoreholes)]

    # -------------------------------------------------------------------------
    # Initialize pipe model
    # -------------------------------------------------------------------------

    # Pipe thermal resistance
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
        r_in, r_out, k_p)
    # Fluid to inner pipe wall thermal resistance (Single U-tube)
    m_flow_pipe = np.max(np.abs(m_flow_borehole))
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow_pipe,  r_in, mu_f, rho_f, k_f, cp_f, epsilon)
    R_f = 1.0 / (h_f * 2 * np.pi * r_in)

    # Single U-tube, same for all boreholes in the bore field
    UTubes = []
    for borehole in borefield:
        SingleUTube = gt.pipes.SingleUTube(
            pos_pipes, r_in, r_out, borehole, k_s, k_g, R_f + R_p)
        UTubes.append(SingleUTube)
    network = gt.networks.Network(
        borefield, UTubes, bore_connectivity=bore_connectivity)

    # -------------------------------------------------------------------------
    # Evaluate the g-functions for the borefield
    # -------------------------------------------------------------------------

    # Calculate the g-function for uniform temperature
    gfunc_Tb = gt.gfunction.gFunction(
        borefield, alpha, time=time, boundary_condition='UBWT',
        options=options, method=method)

    # Calculate the g-function for mixed inlet fluid conditions
    gfunc_equal_Tf_mixed = gt.gfunction.gFunction(
        network, alpha, time=time, m_flow_network=m_flow_network, cp_f=cp_f,
        boundary_condition='MIFT', options=options, method=method)

    # -------------------------------------------------------------------------
    # Plot g-functions
    # -------------------------------------------------------------------------

    ax = gfunc_Tb.visualize_g_function().axes[0]
    ax.plot(np.log(time/ts), gfunc_equal_Tf_mixed.gFunc[0, 0, :], 'C1')
    ax.plot(np.log(time/ts), gfunc_equal_Tf_mixed.gFunc[1, 1, :], 'C2')
    ax.legend([
        'Uniform temperature',
        f'Mixed inlet temperature (m_flow={m_flow_network[0]} kg/s)',
        f'Mixed inlet temperature (m_flow={m_flow_network[1]} kg/s)'])
    plt.tight_layout()

    # For the mixed inlet fluid temperature condition, draw the temperatures
    # and heat extraction rates
    gfunc_equal_Tf_mixed.visualize_temperatures()
    gfunc_equal_Tf_mixed.visualize_temperature_profiles()
    gfunc_equal_Tf_mixed.visualize_heat_extraction_rates()
    gfunc_equal_Tf_mixed.visualize_heat_extraction_rate_profiles()

    return


# Main function
if __name__ == '__main__':
    main()
