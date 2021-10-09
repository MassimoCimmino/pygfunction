# -*- coding: utf-8 -*-
""" Example of g-function calculation using discretized segment lengths along
    the boreholes.
"""

import pygfunction as gt
from numpy import pi
import matplotlib.pyplot as plt
import numpy as np


def compute_rmse(reference, predicted):
    rmse = np.linalg.norm(predicted - reference) / len(reference)
    return rmse


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
    r_out = 0.0211      # Pipe outer radius (m)
    r_in = 0.0147       # Pipe inner radius (m)
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
    m_flow_borehole = 0.25  # Total fluid mass flow rate per borehole (kg/s)
    # The fluid is propylene-glycol (20 %) at 20 degC
    fluid = gt.media.Fluid('MPG', 20.)
    cp_f = fluid.cp     # Fluid specific isobaric heat capacity (J/kg.K)
    rho_f = fluid.rho   # Fluid density (kg/m3)
    mu_f = fluid.mu     # Fluid dynamic viscosity (kg/m.s)
    k_f = fluid.k       # Fluid thermal conductivity (W/m.K)

    # g-Function calculation options
    nSegments = 24
    options = {'nSegments':nSegments, 'disp':True}

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
    gt.boreholes.visualize_field(boreField)
    nBoreholes = len(boreField)

    # -------------------------------------------------------------------------
    # Initialize pipe model
    # -------------------------------------------------------------------------

    # Pipe thermal resistance
    R_p = gt.pipes.conduction_thermal_resistance_circular_pipe(
        r_in, r_out, k_p)
    # Fluid to inner pipe wall thermal resistance (Single U-tube)
    m_flow_pipe = m_flow_borehole
    h_f = gt.pipes.convective_heat_transfer_coefficient_circular_pipe(
        m_flow_pipe, r_in, mu_f, rho_f, k_f, cp_f, epsilon)
    R_f = 1.0/(h_f*2*pi*r_in)

    # Single U-tube, same for all boreholes in the bore field
    UTubes = []
    for borehole in boreField:
        SingleUTube = gt.pipes.SingleUTube(pos_pipes, r_in, r_out,
                                           borehole, k_s, k_g, R_f + R_p)
        UTubes.append(SingleUTube)
    m_flow_network = m_flow_borehole*nBoreholes

    # -------------------------------------------------------------------------
    # Evaluate the g-functions for the borefield
    # -------------------------------------------------------------------------

    # Compute g-function for the converged MIFT case with equal number of
    # segments per borehole, and equal segment lengths along the boreholes
    network = gt.networks.Network(
        boreField, UTubes, m_flow_network=m_flow_network, cp_f=cp_f,
        nSegments=nSegments)
    gfunc_EFT_ref = gt.gfunction.gFunction(
        network, alpha, time=time, boundary_condition='MIFT', options=options)

    # Calculate the g-function for uniform borehole wall temperature
    gfunc_UT_ref = gt.gfunction.gFunction(
        boreField, alpha, time=time, boundary_condition='UBWT', options=options)

    # Compute g-function for discretized UIFT with equal number of segments per
    # borehole, and equal segment lengths across the boreholes
    # Compute g-function with predefined segment lengths

    segment_ratios = gt.utilities.discretize(H, end_length_ratio=0.05)
    nSegments = len(segment_ratios)
    segment_ratios = np.array(segment_ratios)
    options = {'nSegments': nSegments,
               'segment_ratios': segment_ratios, 'disp': False}

    # Update network for MIFT g-function and compute g-function with discretized
    # segment lengths
    network = gt.gfunction.Network(
        boreField, UTubes, m_flow_network=m_flow_network, cp_f=fluid.cp,
        nSegments=nSegments, segment_ratios=segment_ratios)
    gfunc_EFT_pred = gt.gfunction.gFunction(
        network, alpha, time=time, boundary_condition='MIFT',
        options=options)

    # Calculate the g-function for uniform borehole wall temperature with
    # discretized segment lengths
    gfunc_UT_pred = gt.gfunction.gFunction(
        boreField, alpha, time=time, boundary_condition='UBWT', options=options)

    # Compute the rmse between the reference cases and the discretized
    # (predicted) cases
    rmse = compute_rmse(gfunc_EFT_ref.gFunc, gfunc_EFT_pred.gFunc)
    print('RMSE (MIFT) = {0:.5f}'.format(rmse))
    rmse = compute_rmse(gfunc_UT_ref.gFunc, gfunc_UT_pred.gFunc)
    print('RMSE (UT) = {0:.5f}'.format(rmse))

    # -------------------------------------------------------------------------
    # Plot g-functions
    # -------------------------------------------------------------------------

    ax = gfunc_EFT_ref.visualize_g_function().axes[0]
    ax.plot(np.log(time / ts), gfunc_EFT_pred.gFunc, 'k--')
    ax.plot(np.log(time / ts), gfunc_UT_ref.gFunc)
    ax.plot(np.log(time/ts), gfunc_UT_pred.gFunc)
    ax.legend(['Equal inlet temperature Converged',
               'Equal inlet temperature Discretized',
               'UT Converged', 'UT Discretized'])
    plt.tight_layout()

    return


# Main function
if __name__ == '__main__':
    main()
