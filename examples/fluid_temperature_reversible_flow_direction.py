# -*- coding: utf-8 -*-
""" Example of simulation of a geothermal system with multiple series-
    connected boreholes and reversible flow direction.

    The g-function of a bore field is calculated for boundary condition of
    mixed inlet fluid temperature into the boreholes and a combination of two
    fluid mass flow rates in two opposing flow direction. Then, the borehole
    wall temperature variations resulting from a time-varying load profile
    are simulated using the aggregation method of Claesson and Javed (2012).
    Predicted outlet fluid temperatures of double U-tube borehole are
    evaluated.

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
    H = 150.0           # Borehole length (m)
    r_b = 0.075         # Borehole radius (m)

    # Bore field geometry (rectangular array)
    N_1 = 1             # Number of boreholes in the x-direction (columns)
    N_2 = 6             # Number of boreholes in the y-direction (rows)
    B = 7.5             # Borehole spacing, in both directions (m)

    # Pipe dimensions
    r_out = 33.6e-3/2   # Pipe outer radius (m)
    r_in = 27.4e-3/2    # Pipe inner radius (m)
    D_s = 0.11/2        # Shank spacing (m)
    epsilon = 1.0e-6    # Pipe roughness (m)

    # Pipe positions
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos_pipes = [(-D_s, 0.), (D_s, 0.)]

    # Ground properties
    alpha = 2.5/2.2e6   # Ground thermal diffusivity (m2/s)
    k_s = 2.5           # Ground thermal conductivity (W/m.K)
    T_g = 10.           # Undisturbed ground temperatue (degC)

    # Grout properties
    k_g = 1.5           # Grout thermal conductivity (W/m.K)

    # Pipe properties
    k_p = 0.42          # Pipe thermal conductivity (W/m.K)

    # Fluid properties
    m_flow_borehole_min = 0.5   # Minimum fluid mass flow rate per borehole (kg/s)
    m_flow_borehole_max = 1.2   # Maximum fluid mass flow rate per borehole (kg/s)
    m_flow_borehole = np.array([
        -m_flow_borehole_max,
        -m_flow_borehole_min,
        m_flow_borehole_min,
        m_flow_borehole_max
        ])
    nModes = len(m_flow_borehole)
    # Total fluid mass flow rate (kg/s)
    m_flow_network = m_flow_borehole
    # The fluid is propylene-glycol (20 %) at 20 degC
    fluid = gt.media.Fluid('MPG', 20.)
    cp_f = fluid.cp     # Fluid specific isobaric heat capacity (J/kg.K)
    rho_f = fluid.rho   # Fluid density (kg/m3)
    mu_f = fluid.mu     # Fluid dynamic viscosity (kg/m.s)
    k_f = fluid.k       # Fluid thermal conductivity (W/m.K)

    # g-Function calculation options
    options = {'approximate_FLS': True,
               'nSegments': 8,
               'disp': True}

    # Simulation parameters
    dt = 600.                   # Time step (s)
    tmax = 48 * 3600.           # Maximum time (s)
    Nt = int(np.ceil(tmax/dt))  # Number of time steps
    time = dt * np.arange(1, Nt+1)
    time_h = time / 3600.

    # Load aggregation scheme
    LoadAgg = gt.load_aggregation.ClaessonJaved(dt, tmax, nSources=nModes)

    # -------------------------------------------------------------------------
    # Initialize bore field and pipe models
    # -------------------------------------------------------------------------

    # The field is a retangular array
    borefield = gt.borefield.Borefield.rectangle_field(
        N_1, N_2, B, B, H, D, r_b)
    nBoreholes = len(borefield)
    H_tot = np.sum([b.H for b in borefield])

    # Boreholes are connected in series
    bore_connectivity = [i-1 for i in range(nBoreholes)]

    # Pipe thermal resistance
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
            r_in, r_out, k_p)

    # Fluid to inner pipe wall thermal resistance (Double U-tube in parallel)
    m_flow_pipe = np.max(np.abs(m_flow_borehole))
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
            m_flow_pipe, r_in, mu_f, rho_f, k_f, cp_f, epsilon)
    R_f = 1.0 / (h_f * 2 * np.pi * r_in)

    # Double U-tube (parallel), same for all boreholes in the bore field
    UTubes = []
    for borehole in borefield:
        UTube = gt.pipes.SingleUTube(
            pos_pipes, r_in, r_out, borehole, k_s, k_g, R_f + R_p)
        UTubes.append(UTube)
    # Build a network object from the list of UTubes
    network = gt.networks.Network(
        borefield, UTubes, bore_connectivity=bore_connectivity)

    # -------------------------------------------------------------------------
    # Calculate g-function
    # -------------------------------------------------------------------------

    # Get time values needed for g-function evaluation
    time_req = LoadAgg.get_times_for_simulation()
    # Calculate g-function
    gFunc = gt.gfunction.gFunction(
        network, alpha, time=time_req, boundary_condition='MIFT',
        m_flow_network=m_flow_network, cp_f=cp_f, options=options)
    # Initialize load aggregation scheme
    LoadAgg.initialize(gFunc.gFunc / (2 * np.pi * k_s))

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    # Sinusoidal variation between -80 and 80 W/m with minimum of 10 W/m
    Q_min = -80. * H * nBoreholes
    Q_max = 80. * H * nBoreholes
    Q_small = 10. * H * nBoreholes
    Q_avg = 0.5 * (Q_min + Q_max)
    Q_amp = 0.5 * (Q_max - Q_min)
    Q_tot = -Q_avg - Q_amp * np.sin(2 * np.pi * time_h / 24)
    # Mass flow rate is proportional to the absolute load
    m_flow_min = np.min(np.abs(m_flow_network))
    m_flow_max = np.max(np.abs(m_flow_network))
    m_flow = np.sign(Q_tot) * np.maximum(
        m_flow_min,
        np.minimum(m_flow_max, m_flow_max * np.abs(Q_tot) / Q_max))
    # The minimum heat load is 10 W/m and the mass flow rate is 0 when Q_tot=0
    Q_tot[np.abs(Q_tot) < Q_small] = 0.
    m_flow[np.abs(Q_tot) < Q_small] = 0.

    T_b = np.zeros(Nt)
    T_f_in = np.zeros(Nt)
    T_f_out = np.zeros(Nt)
    for i, (t, Q_i) in enumerate(zip(time, Q_tot)):
        # Increment time step by (1)
        LoadAgg.next_time_step(t)

        # Solve for temperatures and heat extraction rates (variable mass flow rate)
        if np.abs(Q_tot[i]) >= Q_small:
            xi = np.zeros(nModes)
            iMode = np.argmax(m_flow[i] <= m_flow_network)
            if iMode > 0:
                xi[iMode-1] = (m_flow_network[iMode] - m_flow[i]) / (m_flow_network[iMode] - m_flow_network[iMode-1])
                xi[iMode] = (m_flow[i] - m_flow_network[iMode-1]) / (m_flow_network[iMode] - m_flow_network[iMode-1])
            else:
                xi[0] = 1.
            LoadAgg.set_current_load(Q_tot[i] / H_tot * xi)
            deltaT_b = LoadAgg.temporal_superposition()
            T_b[i] = T_g - deltaT_b @ xi
            T_f_in[i] = network.get_network_inlet_temperature(
                Q_tot[i], T_b[i], m_flow[i], cp_f, nSegments=1, segment_ratios=None)
            T_f_out[i] = network.get_network_outlet_temperature(
                T_f_in[i], T_b[i], m_flow[i], cp_f, nSegments=1, segment_ratios=None)
        else:
            T_b[i] = np.nan
            T_f_in[i] = np.nan
            T_f_out[i] = np.nan

    # -------------------------------------------------------------------------
    # Plot hourly heat extraction rates and temperatures
    # -------------------------------------------------------------------------

    # Configure figure and axes
    fig = gt.utilities._initialize_figure()

    ax1 = fig.add_subplot(221)
    # Axis labels
    ax1.set_xlabel(r'Time [hours]')
    ax1.set_ylabel(r'Total heat extraction rate [W/m]')
    gt.utilities._format_axes(ax1)

    # Plot heat extraction rates
    hours = np.arange(1, Nt+1) * dt / 3600.
    ax1.plot(hours, Q_tot / H_tot)

    ax2 = fig.add_subplot(222)
    # Axis labels
    ax2.set_xlabel(r'Time [hours]')
    ax2.set_ylabel(r'Fluid mass flow rate [kg/s]')
    gt.utilities._format_axes(ax2)

    # Plot temperatures
    ax2.plot(hours, m_flow)

    ax3 = fig.add_subplot(223)
    # Axis labels
    ax3.set_xlabel(r'Time [hours]')
    ax3.set_ylabel(r'Temperature [degC]')
    gt.utilities._format_axes(ax3)

    # Plot temperatures
    ax3.plot(hours, T_b, label='Borehole wall')
    ax3.plot(hours, T_f_out, '-.',
             label='Outlet')
    ax3.legend()

    # Adjust to plot window
    plt.tight_layout()

    return


# Main function
if __name__ == '__main__':
    main()
