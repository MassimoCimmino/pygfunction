# -*- coding: utf-8 -*-
""" Example of simulation of a geothermal system and comparison between
    single and double U-tube pipe configurations.

    The g-function of a single borehole is calculated for boundary condition of
    uniform borehole wall temperature along the borehole. Then, the borehole
    wall temperature variations resulting from a time-varying load profile
    are simulated using the aggregation method of Claesson and Javed (2012).
    Predicted outlet fluid temperatures of three pipe configurations
    (single U-tube, double U-tube in series, double U-tube in parallel) are
    compared.

"""
import matplotlib.pyplot as plt
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

    # Pipe dimensions
    rp_out = 0.0211     # Pipe outer radius (m)
    rp_in = 0.0147      # Pipe inner radius (m)
    D_s = 0.052         # Shank spacing (m)
    epsilon = 1.0e-6    # Pipe roughness (m)

    # Pipe positions
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos_single = [(-D_s, 0.), (D_s, 0.)]
    # Double U-tube [(x_in1, y_in1), (x_in2, y_in2),
    #                (x_out1, y_out1), (x_in2, y_in2)]
    # Note: in series configuration, fluid enters pipe (in,1), exits (out,1),
    # then enters (in,2) and finally exits (out,2)
    pos_double = [(-D_s, 0.), (0., -D_s), (D_s, 0.), (0., D_s)]

    # Ground properties
    alpha = 1.0e-6      # Ground thermal diffusivity (m2/s)
    k_s = 2.0           # Ground thermal conductivity (W/m.K)
    T_g = 10.0          # Undisturbed ground temperature (degC)

    # Grout properties
    k_g = 1.0           # Grout thermal conductivity (W/m.K)

    # Pipe properties
    k_p = 0.4           # Pipe thermal conductivity (W/m.K)

    # Fluid properties
    m_flow = 0.25       # Total fluid mass flow rate (kg/s)
    # The fluid is propylene-glycol (20 %) at 20 degC
    fluid = gt.media.Fluid('MPG', 20.)
    cp_f = fluid.cp     # Fluid specific isobaric heat capacity (J/kg.K)
    den_f = fluid.rho   # Fluid density (kg/m3)
    visc_f = fluid.mu   # Fluid dynamic viscosity (kg/m.s)
    k_f = fluid.k       # Fluid thermal conductivity (W/m.K)

    # g-Function calculation options
    options = {'nSegments': 8,
               'disp': True}

    # Simulation parameters
    dt = 3600.                  # Time step (s)
    tmax = 1.*8760. * 3600.     # Maximum time (s)
    Nt = int(np.ceil(tmax/dt))  # Number of time steps
    time = dt * np.arange(1, Nt+1)

    # Evaluate heat extraction rate
    Q = synthetic_load(time/3600.)

    # Load aggregation scheme
    LoadAgg = gt.load_aggregation.ClaessonJaved(dt, tmax)

    # -------------------------------------------------------------------------
    # Calculate g-function
    # -------------------------------------------------------------------------

    # The field contains only one borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)
    boreField = [borehole]
    # Get time values needed for g-function evaluation
    time_req = LoadAgg.get_times_for_simulation()
    # Calculate g-function
    gFunc = gt.gfunction.gFunction(
        boreField, alpha, time=time_req, options=options)
    # gt.gfunction.uniform_temperature(boreField, time_req, alpha,
    #                                          nSegments=nSegments)
    # Initialize load aggregation scheme
    LoadAgg.initialize(gFunc.gFunc/(2*pi*k_s))

    # -------------------------------------------------------------------------
    # Initialize pipe models
    # -------------------------------------------------------------------------

    # Pipe thermal resistance
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
        rp_in, rp_out, k_p)
    # Fluid to inner pipe wall thermal resistance (Single U-tube and double
    # U-tube in series)
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow, rp_in, visc_f, den_f, k_f, cp_f, epsilon)
    R_f_ser = 1.0/(h_f*2*pi*rp_in)
    # Fluid to inner pipe wall thermal resistance (Double U-tube in parallel)
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow/2, rp_in, visc_f, den_f, k_f, cp_f, epsilon)
    R_f_par = 1.0/(h_f*2*pi*rp_in)

    # Single U-tube
    SingleUTube = gt.pipes.SingleUTube(pos_single, rp_in, rp_out,
                                       borehole, k_s, k_g, R_f_ser + R_p)
    # Double U-tube (parallel)
    DoubleUTube_par = gt.pipes.MultipleUTube(pos_double, rp_in, rp_out,
                                             borehole, k_s, k_g, R_f_par + R_p,
                                             nPipes=2, config='parallel')
    # Double U-tube (series)
    DoubleUTube_ser = gt.pipes.MultipleUTube(pos_double, rp_in, rp_out,
                                             borehole, k_s, k_g, R_f_ser + R_p,
                                             nPipes=2, config='series')

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    T_b = np.zeros(Nt)
    T_f_in_single = np.zeros(Nt)
    T_f_in_double_par = np.zeros(Nt)
    T_f_in_double_ser = np.zeros(Nt)
    T_f_out_single = np.zeros(Nt)
    T_f_out_double_par = np.zeros(Nt)
    T_f_out_double_ser = np.zeros(Nt)
    for i, (t, Q_b_i) in enumerate(zip(time, Q)):
        # Increment time step by (1)
        LoadAgg.next_time_step(t)

        # Apply current load
        LoadAgg.set_current_load(Q_b_i/H)

        # Evaluate borehole wall temperature
        deltaT_b = LoadAgg.temporal_superposition()
        T_b[i] = T_g - deltaT_b

        # Evaluate inlet fluid temperature
        T_f_in_single[i] = SingleUTube.get_inlet_temperature(
                Q[i], T_b[i], m_flow, cp_f)
        T_f_in_double_par[i] = DoubleUTube_par.get_inlet_temperature(
                Q[i], T_b[i], m_flow, cp_f)
        T_f_in_double_ser[i] = DoubleUTube_ser.get_inlet_temperature(
                Q[i], T_b[i], m_flow, cp_f)

        # Evaluate outlet fluid temperature
        T_f_out_single[i] = SingleUTube.get_outlet_temperature(
                T_f_in_single[i],  T_b[i], m_flow, cp_f)
        T_f_out_double_par[i] = DoubleUTube_par.get_outlet_temperature(
                T_f_in_double_par[i],  T_b[i], m_flow, cp_f)
        T_f_out_double_ser[i] = DoubleUTube_ser.get_outlet_temperature(
                T_f_in_double_ser[i],  T_b[i], m_flow, cp_f)

    # -------------------------------------------------------------------------
    # Plot hourly heat extraction rates and temperatures
    # -------------------------------------------------------------------------

    # Configure figure and axes
    fig = gt.utilities._initialize_figure()

    ax1 = fig.add_subplot(211)
    # Axis labels
    ax1.set_xlabel(r'Time [hours]')
    ax1.set_ylabel(r'Total heat extraction rate [W]')
    gt.utilities._format_axes(ax1)

    # Plot heat extraction rates
    hours = np.arange(1, Nt+1) * dt / 3600.
    ax1.plot(hours, Q)

    ax2 = fig.add_subplot(212)
    # Axis labels
    ax2.set_xlabel(r'Time [hours]')
    ax2.set_ylabel(r'Temperature [degC]')
    gt.utilities._format_axes(ax2)

    # Plot temperatures
    ax2.plot(hours, T_b, 'k-', lw=1.5, label='Borehole wall')
    ax2.plot(hours, T_f_out_single, '--',
             label='Outlet, single U-tube')
    ax2.plot(hours, T_f_out_double_par, '-.',
             label='Outlet, double U-tube (parallel)')
    ax2.plot(hours, T_f_out_double_ser, ':',
             label='Outlet, double U-tube (series)')
    ax2.legend()

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
    T_f_single = SingleUTube.get_temperature(z,
                                             T_f_in_single[it],
                                             T_b[it],
                                             m_flow,
                                             cp_f)
    T_f_double_par = DoubleUTube_par.get_temperature(z,
                                                     T_f_in_double_par[it],
                                                     T_b[it],
                                                     m_flow,
                                                     cp_f)
    T_f_double_ser = DoubleUTube_ser.get_temperature(z,
                                                     T_f_in_double_ser[it],
                                                     T_b[it],
                                                     m_flow,
                                                     cp_f)

    # Configure figure and axes
    fig = gt.utilities._initialize_figure()

    ax3 = fig.add_subplot(131)
    # Axis labels
    ax3.set_xlabel(r'Temperature [degC]')
    ax3.set_ylabel(r'Depth from borehole head [m]')
    gt.utilities._format_axes(ax3)

    # Plot temperatures
    ax3.plot(np.array([T_b[it], T_b[it]]), np.array([0., H]), 'k--')
    ax3.plot(T_f_single, z, 'b-')
    ax3.legend(['Borehole wall', 'Fluid'])

    ax4 = fig.add_subplot(132)
    # Axis labels
    ax4.set_xlabel(r'Temperature [degC]')
    ax4.set_ylabel(r'Depth from borehole head [m]')
    gt.utilities._format_axes(ax4)

    # Plot temperatures
    ax4.plot(T_f_double_par, z, 'b-')
    ax4.plot(np.array([T_b[it], T_b[it]]), np.array([0., H]), 'k--')

    ax5 = fig.add_subplot(133)
    # Axis labels
    ax5.set_xlabel(r'Temperature [degC]')
    ax5.set_ylabel(r'Depth from borehole head [m]')
    gt.utilities._format_axes(ax5)

    # Plot temperatures
    ax5.plot(T_f_double_ser, z, 'b-')
    ax5.plot(np.array([T_b[it], T_b[it]]), np.array([0., H]), 'k--')

    # Reverse y-axes
    ax3.set_ylim(ax3.get_ylim()[::-1])
    ax4.set_ylim(ax4.get_ylim()[::-1])
    ax5.set_ylim(ax5.get_ylim()[::-1])
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
