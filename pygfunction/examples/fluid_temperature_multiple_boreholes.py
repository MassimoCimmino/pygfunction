# -*- coding: utf-8 -*-
""" Example of simulation of a geothermal system with multiple boreholes.

    The g-function of a bore field is calculated for boundary condition of
    mixed inlet fluid temperature into the boreholes. Then, the borehole
    wall temperature variations resulting from a time-varying load profile
    are simulated using the aggregation method of Claesson and Javed (2012).
    Predicted outlet fluid temperatures of double U-tube borehole are
    evaluated.

"""
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy.constants import pi

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
    N_1 = 6             # Number of boreholes in the x-direction (columns)
    N_2 = 4             # Number of boreholes in the y-direction (rows)
    B = 7.5             # Borehole spacing, in both directions (m)

    # Pipe dimensions
    rp_out = 0.0211     # Pipe outer radius (m)
    rp_in = 0.0147      # Pipe inner radius (m)
    D_s = 0.052         # Shank spacing (m)
    epsilon = 1.0e-6    # Pipe roughness (m)

    # Pipe positions
    # Double U-tube [(x_in1, y_in1), (x_in2, y_in2),
    #                (x_out1, y_out1), (x_out2, y_out2)]
    pos = [(-D_s, 0.), (0., -D_s), (D_s, 0.), (0., D_s)]

    # Ground properties
    alpha = 1.0e-6      # Ground thermal diffusivity (m2/s)
    k_s = 2.0           # Ground thermal conductivity (W/m.K)
    T_g = 10.0          # Undisturbed ground temperature (degC)

    # Grout properties
    k_g = 1.0           # Grout thermal conductivity (W/m.K)

    # Pipe properties
    k_p = 0.4           # Pipe thermal conductivity (W/m.K)

    # Fluid properties
    m_flow_borehole = 0.25      # Total fluid mass flow rate per borehole (kg/s)
    m_flow = m_flow_borehole*N_1*N_2    # Total fluid mass flow rate (kg/s)
    cp_f = 3977.                # Fluid specific heat capacity (J/kg.K)
    den_f = 1015.               # Fluid density (kg/m3)
    visc_f = 0.00203            # Fluid dynamic viscosity (kg/m.s)
    k_f = 0.492                 # Fluid thermal conductivity (W/m.K)

    # Number of segments per borehole
    nSegments = 12

    # Simulation parameters
    dt = 3600.                  # Time step (s)
    tmax = 1.*8760. * 3600.     # Maximum time (s)
    Nt = int(np.ceil(tmax/dt))  # Number of time steps

    # Load aggregation scheme
    LoadAgg = gt.load_aggregation.ClaessonJaved(dt, tmax)

    # -------------------------------------------------------------------------
    # Initialize bore field and pipe models
    # -------------------------------------------------------------------------

    # The field is a retangular array
    boreField = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)
    nBoreholes = len(boreField)

    # Pipe thermal resistance
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
            rp_in, rp_out, k_p)

    # Fluid to inner pipe wall thermal resistance (Double U-tube in parallel)
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
            m_flow_borehole/2, rp_in, visc_f, den_f, k_f, cp_f, epsilon)
    R_f = 1.0/(h_f*2*pi*rp_in)

    # Double U-tube (parallel), same for all boreholes in the bore field
    UTubes = []
    for borehole in boreField:
        UTube = gt.pipes.MultipleUTube(
            pos, rp_in, rp_out, borehole, k_s, k_g, R_f + R_p,
            nPipes=2, config='parallel')
        UTubes.append(UTube)
    # Build a network object from the list of UTubes
    network = gt.networks.Network(boreField, UTubes)

    # -------------------------------------------------------------------------
    # Calculate g-function
    # -------------------------------------------------------------------------

    # Get time values needed for g-function evaluation
    time_req = LoadAgg.get_times_for_simulation()
    # Calculate g-function
    gFunc = gt.gfunction.mixed_inlet_temperature(
            network, m_flow, cp_f, time_req, alpha,
            nSegments=nSegments, disp=True)
    # Initialize load aggregation scheme
    LoadAgg.initialize(gFunc/(2*pi*k_s))

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    time = 0.
    i = -1
    T_b = np.zeros(Nt)
    Q_tot = np.zeros(Nt)
    T_f_in = np.zeros(Nt)
    T_f_out = np.zeros(Nt)
    while time < tmax:
        # Increment time step by (1)
        time += dt
        i += 1
        LoadAgg.next_time_step(time)

        # Evaluate heat extraction rate
        Q_tot[i] = nBoreholes*synthetic_load(time/3600.)

        # Apply current load (in watts per meter of borehole)
        Q_b = Q_tot[i]/nBoreholes
        LoadAgg.set_current_load(Q_b/H)

        # Evaluate borehole wall temperature
        deltaT_b = LoadAgg.temporal_superposition()
        T_b[i] = T_g - deltaT_b

        # Evaluate inlet fluid temperature (all boreholes are the same)
        T_f_in[i] = network.get_network_inlet_temperature(
                Q_tot[i], T_b[i], m_flow, cp_f, nSegments=1)

        # Evaluate outlet fluid temperature
        T_f_out[i] = network.get_network_outlet_temperature(
                T_f_in[i],  T_b[i], m_flow, cp_f, nSegments=1)

    # -------------------------------------------------------------------------
    # Plot hourly heat extraction rates and temperatures
    # -------------------------------------------------------------------------

    plt.rc('figure')
    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    # Axis labels
    ax1.set_xlabel(r'Time (hours)')
    ax1.set_ylabel(r'Total heat extraction rate (W)')
    hours = np.array([(j+1)*dt/3600. for j in range(Nt)])
    # Plot heat extraction rates
    ax1.plot(hours, Q_tot, 'b-', lw=1.5)

    ax2 = fig.add_subplot(212)
    # Axis labels
    ax2.set_xlabel(r'Time (hours)')
    ax2.set_ylabel(r'Temperature (degC)')
    # Plot temperatures
    ax2.plot(hours, T_b, 'k-', lw=1.5, label='Borehole wall')
    ax2.plot(hours, T_f_out, 'r-.', lw=1.5,
             label='Outlet, double U-tube (parallel)')
    ax2.legend()

    # Show minor ticks
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    # Adjust to plot window
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Plot fluid temperature profiles
    # -------------------------------------------------------------------------

    # Evaluate temperatures at nz evenly spaced depths along the borehole
    # at the (it+1)-th time step
    nz = 20
    it = 8724
    z = np.linspace(0., H, num=nz)
    T_f = UTubes[0].get_temperature(z, T_f_in[it], T_b[it], m_flow, cp_f)

    plt.rc('figure')
    fig = plt.figure()

    ax3 = fig.add_subplot(111)
    # Axis labels
    ax3.set_xlabel(r'Temperature (degC)')
    ax3.set_ylabel(r'Depth from borehole head (m)')
    # Plot temperatures
    pltFlu = ax3.plot(T_f, z, 'b-', lw=1.5, label='Fluid')
    pltWal = ax3.plot(np.array([T_b[it], T_b[it]]), np.array([0., H]),
                      'k--', lw=1.5, label='Borehole wall')
    ax3.legend(handles=[pltFlu[0]]+pltWal)

    # Show minor ticks
    ax3.xaxis.set_minor_locator(AutoMinorLocator())
    ax3.yaxis.set_minor_locator(AutoMinorLocator())
    # Reverse y-axes
    ax3.set_ylim(ax3.get_ylim()[::-1])
    # Adjust to plot window
    plt.tight_layout()

    return


def synthetic_load(x):
    """
    Synthetic load profile of Bernier et al. (2004).

    Returns load y (in watts) at time x (in hours).
    """
    A = 2000.0
    B = 2190.0
    C = 80.0
    D = 2.0
    E = 0.01
    F = 0.0
    G = 0.95

    func = (168.0-C)/168.0
    for i in [1, 2, 3]:
        func += 1.0/(i*pi)*(np.cos(C*pi*i/84.0)-1.0) \
                          *(np.sin(pi*i/84.0*(x-B)))
    func = func*A*np.sin(pi/12.0*(x-B)) \
           *np.sin(pi/4380.0*(x-B))

    y = func + (-1.0)**np.floor(D/8760.0*(x-B))*abs(func) \
      + E*(-1.0)**np.floor(D/8760.0*(x-B))/np.sign(np.cos(D*pi/4380.0*(x-F))+G)
    return -y


# Main function
if __name__ == '__main__':
    main()
