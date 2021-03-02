# -*- coding: utf-8 -*-
""" An example showing how to make use of the common g-function interface via
    - dictionary access
    - listing arguments

"""

import pygfunction as gt
import matplotlib.pyplot as plt
import numpy as np


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    D = 4.0  # Borehole buried depth (m)
    H = 150.0  # Borehole length (m)
    r_b = 0.075  # Borehole radius (m)
    B = 7.5  # Borehole spacing (m)

    # Pipe dimensions
    rp_out = 0.0211  # Pipe outer radius (m)
    rp_in = 0.0147  # Pipe inner radius (m)
    D_s = 0.052  # Shank spacing (m)
    epsilon = 1.0e-6  # Pipe roughness (m)

    # Pipe positions
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos_pipes = [(-D_s, 0.), (D_s, 0.)]

    # Ground properties
    alpha = 1.0e-6  # Ground thermal diffusivity (m2/s)
    k_s = 2.0  # Ground thermal conductivity (W/m.K)

    # Grout properties
    k_g = 1.0  # Grout thermal conductivity (W/m.K)

    # Pipe properties
    k_p = 0.4  # Pipe thermal conductivity (W/m.K)

    # Fluid properties
    m_flow = 0.25  # Total fluid mass flow rate per borehole (kg/s)
    cp_f = 3977.  # Fluid specific isobaric heat capacity (J/kg.K)
    den_f = 1015.  # Fluid density (kg/m3)
    visc_f = 0.00203  # Fluid dynamic viscosity (kg/m.s)
    k_f = 0.492  # Fluid thermal conductivity (W/m.K)

    # Geometrically expanding time vector.
    dt = 100 * 3600.  # Time step
    years = 3000.
    tmax = years * 8760. * 3600.  # Maximum time
    Nt = 50  # Number of time steps
    ts = H ** 2 / (9. * alpha)  # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)
    ln_tts = np.log(time / ts)  # ln(t / ts) for plotting versus g-function later

    # Thermal properties
    alpha = 1.0e-6  # Ground thermal diffusivity (m2/s)

    # -------------------------------------------------------------------------
    # Borehole fields
    # -------------------------------------------------------------------------

    # Field of 3x2 (n=6) boreholes
    N_1 = 3
    N_2 = 2
    boreField = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)

    # -------------------------------------------------------------------------
    # Uniform heat flux
    # -------------------------------------------------------------------------
    # compute a g-function for uniform heat flux by using the common g-function interface object
    g_UHF_interface = gt.gfunction.gFunction(boreField, alpha, time=time, boundary_condition='UHTR', options={'disp': True})
    g_UHF = g_UHF_interface.gFunc

    # -------------------------------------------------------------------------
    # Uniform borehole wall temperature
    # -------------------------------------------------------------------------
    # compute a uniform borehole wall temperature g-function using the common g-function interface
    nSegments = 12  # provide a number of segments
    method = 'linear'  # linear interpolation for h_dt thermal response factors
    disp = True

    # define the options as a dictionary
    options = {'nSegments': nSegments, 'method': method, 'disp': disp}

    g_UBHWT_interface = gt.gfunction.gFunction(boreField, alpha, time=time, boundary_condition='UBWT', options=options)
    g_UBHWT = g_UBHWT_interface.gFunc

    # -------------------------------------------------------------------------
    # Equal (uniform) inlet fluid temperature
    # -------------------------------------------------------------------------
    # Initialize pipe model
    # Pipe thermal resistance
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(rp_in, rp_out, k_p)
    # Fluid to inner pipe wall thermal resistance (Single U-tube)
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(m_flow,
                                                                      rp_in,
                                                                      visc_f,
                                                                      den_f,
                                                                      k_f,
                                                                      cp_f,
                                                                      epsilon)
    R_f = 1.0 / (h_f * 2 * np.pi * rp_in)
    # Single U-tube, same for all boreholes in the bore field
    UTubes = []
    for borehole in boreField:
        SingleUTube = gt.pipes.SingleUTube(pos_pipes, rp_in, rp_out,
                                           borehole, k_s, k_g, R_f + R_p)
        UTubes.append(SingleUTube)
    g_EIFT_interface = gt.gfunction.gFunction(boreField, alpha, UTubes=UTubes, m_flow=m_flow, cp=cp_f,
                                              time=time, boundary_condition='EIFT', options=options)
    g_EIFT = g_EIFT_interface.gFunc

    # plot different responses on the same figure
    fig, ax = plt.subplots()

    ax.plot(ln_tts, g_UHF, label='UHF', ls='--')
    ax.plot(ln_tts, g_UBHWT, label='UBHWT', ls='-.')
    ax.plot(ln_tts, g_EIFT, label='EIFT', ls='-')

    ax.set_ylabel('g')
    ax.set_xlabel('ln(t/t$_s$)')

    fig.savefig('g_functions.jpg')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
