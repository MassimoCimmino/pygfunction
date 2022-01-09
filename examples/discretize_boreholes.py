# -*- coding: utf-8 -*-
""" Example of g-function calculation using non-uniform segment lengths along
    the boreholes.

    The g-functions of a field of 6x4 boreholes are calculated for two
    boundary conditions : (1) a uniform borehole wall temperature along the
    boreholes equal for all boreholes, and (2) an equal inlet fluid
    temperature into the boreholes. g-Functions using 8 segments in a
    non-uniform discretization are compared to reference g-functions
    calculated using 48 segments of equal lengths. It is shown that g-functions
    can be calculated accurately using a small number of segments.
"""

import pygfunction as gt
from numpy import pi
import matplotlib.pyplot as plt
import numpy as np


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

    # Number of segments used in the reference calculation with uniform
    # discretization
    nSegments_uniform = 48
    options_uniform = {'nSegments': nSegments_uniform,
                       'segment_ratios': None,
                       'disp': True}
    # Number of segments used in the calculation with non-uniform
    # discretization
    nSegments_unequal = 8
    segment_ratios = gt.utilities.segment_ratios(
        nSegments_unequal, end_length_ratio=0.02)
    options_unequal = {'nSegments': nSegments_unequal,
                       'segment_ratios': segment_ratios,
                       'disp':True}

    # Geometrically expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt = 25                         # Number of time steps
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
        SingleUTube = gt.pipes.SingleUTube(
            pos_pipes, r_in, r_out, borehole, k_s, k_g, R_f + R_p)
        UTubes.append(SingleUTube)
    m_flow_network = m_flow_borehole*nBoreholes

    # Network of boreholes connected in parallel
    network = gt.networks.Network(
        boreField, UTubes, m_flow_network=m_flow_network, cp_f=cp_f)

    # -------------------------------------------------------------------------
    # Evaluate the g-functions for the borefield
    # -------------------------------------------------------------------------

    # Compute g-function for the converged MIFT case with equal number of
    # segments per borehole, and equal segment lengths along the boreholes
    gfunc_MIFT_uniform = gt.gfunction.gFunction(
        network, alpha, time=time, boundary_condition='MIFT',
        options=options_uniform)

    # Calculate the g-function for uniform borehole wall temperature
    gfunc_UBWT_uniform = gt.gfunction.gFunction(
        boreField, alpha, time=time, boundary_condition='UBWT',
        options=options_uniform)

    # Compute g-function for the MIFT case with equal number of segments per
    # borehole, and non-uniform segment lengths along the boreholes
    gfunc_MIFT_unequal = gt.gfunction.gFunction(
        network, alpha, time=time, boundary_condition='MIFT',
        options=options_unequal)

    # Calculate the g-function for uniform borehole wall temperature
    gfunc_UBWT_unequal = gt.gfunction.gFunction(
        boreField, alpha, time=time, boundary_condition='UBWT',
        options=options_unequal)

    # Compute the rmse between the reference cases and the discretized
    # (predicted) cases
    RMSE_MIFT = RMSE(gfunc_MIFT_uniform.gFunc, gfunc_MIFT_unequal.gFunc)
    print(f'RMSE (MIFT) = {RMSE_MIFT:.5f}')
    RMSE_UBWT = RMSE(gfunc_UBWT_uniform.gFunc, gfunc_UBWT_unequal.gFunc)
    print(f'RMSE (UBWT) = {RMSE_UBWT:.5f}')

    # -------------------------------------------------------------------------
    # Plot g-functions
    # -------------------------------------------------------------------------

    ax = gfunc_MIFT_uniform.visualize_g_function().axes[0]
    ax.plot(np.log(time / ts), gfunc_UBWT_uniform.gFunc)
    ax.plot(np.log(time / ts), gfunc_MIFT_unequal.gFunc, 'o')
    ax.plot(np.log(time / ts), gfunc_UBWT_unequal.gFunc, 'o')
    ax.legend(
        ['Equal inlet temperature (uniform segments)',
         'Uniform borehole wall temperature (uniform segments)',
         'Equal inlet temperature (non-uniform segments)',
         'Uniform borehole wall temperature (non-uniform segments)'])
    plt.tight_layout()

    return


def RMSE(reference, predicted):
    rmse = np.linalg.norm(predicted - reference) / len(reference)
    return rmse


# Main function
if __name__ == '__main__':
    main()
