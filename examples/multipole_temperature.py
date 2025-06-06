# -*- coding: utf-8 -*-
""" Example of calculation of grout and ground temperatures using the multipole
    method.

    The thermal resistances of a borehole with two pipes are evaluated using
    the multipole method of Claesson and Hellstrom (2011). Based on the
    calculated thermal resistances, the heat flows from the pipes required to
    obtain pipe temperatures of 1 degC are evaluated. The temperatures in and
    around the borehole with 2 pipes are then calculated. Results are verified
    against the results of Claesson and Hellstrom (2011).

"""
import os

import matplotlib.pyplot as plt
import numpy as np

import pygfunction as gt


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    r_b = 0.070         # Borehole radius (m)

    # Pipe dimensions
    n_p = 2             # Number of pipes
    # Pipe outer radius (m)
    rp_out = 0.02*np.ones(n_p)

    # Pipe positions
    # Single U-tube [(x_1, y_1), (x_2, y_2)]
    pos_pipes = [(0.03, 0.00), (-0.03, 0.02)]

    # Ground properties
    k_s = 2.5           # Ground thermal conductivity (W/m.K)

    # Grout properties
    k_g = 1.5           # Grout thermal conductivity (W/m.K)

    # Fluid properties
    # Fluid to outer pipe wall thermal resistance (m.K/W)
    R_fp = 1.2 / (2 * np.pi * k_g) * np.ones(n_p)

    # Borehole wall temperature (degC)
    T_b = 0.0

    # Fluid temperatures (degC)
    T_f = np.array([1., 1.])

    # Path to validation data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'data', 'ClaHel11_multipole_temperature.txt')

    # Thermal resistances for J=3
    R_Claesson = 0.01*np.array([25.592, 1.561, 25.311])

    # Number of multipoles per pipe
    J = 3

    # -------------------------------------------------------------------------
    # Evaluate the internal thermal resistances
    # -------------------------------------------------------------------------

    # Thermal resistances
    (R, Rd) = gt.pipes.thermal_resistances(pos_pipes, rp_out, r_b, k_s, k_g,
                                           R_fp, J=3)
    print(50*'-')
    print('Thermal resistance:\t\t100*R11\t100*R12\t100*R22')
    print(f'Claesson and Hellstrom:\t{100*R_Claesson[0]:.3f}'
          f'\t{100*R_Claesson[1]:.3f}\t{100*R_Claesson[2]:.3f}')
    print(f'Present:\t\t\t\t{100*R[0,0]:.3f}\t{100*R[0,1]:.3f}'
          f'\t{100*R[1,1]:.3f}')
    print(50*'-')

    # Heat flows
    Q = np.linalg.solve(R, T_f - T_b)

    # -------------------------------------------------------------------------
    # Temperatures along y=0.
    # -------------------------------------------------------------------------

    # Grid points to evaluate temperatures
    x = np.linspace(-0.1, 0.1, num=200)
    y = np.zeros_like(x)

    # Evaluate temperatures using multipole method
    (T_f, T, it, eps_max) = gt.pipes.multipole(pos_pipes, rp_out, r_b, k_s,
                                               k_g, R_fp, T_b, Q, J,
                                               x_T=x, y_T=y)

    # Load validation data
    data = np.loadtxt(file_path, skiprows=1)

    # Configure figure and axes
    fig = gt.utilities._initialize_figure()

    ax1 = fig.add_subplot(111)
    # Axis labels
    ax1.set_xlabel(r'x (m)')
    ax1.set_ylabel(r'$T(x,0)$')
    # Axis limits
    ax1.set_xlim([-0.1, 0.1])
    ax1.set_ylim([-0.2, 1.2])
    # Show grid
    ax1.grid()
    gt.utilities._format_axes(ax1)

    ax1.plot(x, T, label='pygfunction')
    ax1.plot(data[:,0], data[:,1], 'ko',
             label='Claesson and Hellstrom (2011)')
    ax1.legend(loc='upper left')

    # Adjust to plot window
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Temperatures in -0.1 < x < 0.1, -0.1 < y < 0.1
    # -------------------------------------------------------------------------

    # Grid points to evaluate temperatures
    N_xy = 200
    x = np.linspace(-0.1, 0.1, num=N_xy)
    y = np.linspace(-0.1, 0.1, num=N_xy)
    X, Y = np.meshgrid(x, y)

    # Evaluate temperatures using multipole method
    (T_f, T, it, eps_max) = gt.pipes.multipole(pos_pipes, rp_out, r_b, k_s,
                                               k_g, R_fp, T_b, Q, J,
                                               x_T=X.flatten(),
                                               y_T=Y.flatten())

    # Configure figure and axes
    fig = gt.utilities._initialize_figure()

    ax1 = fig.add_subplot(111)
    # Axis labels
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    # Axis limits
    plt.axis([-0.1, 0.1, -0.1, 0.1])
    plt.gca().set_aspect('equal', adjustable='box')
    gt.utilities._format_axes(ax1)

    # Borehole wall outline
    borewall = plt.Circle((0., 0.), radius=r_b,
                          fill=False, linestyle='--', linewidth=2.)
    ax1.add_patch(borewall)
    # Pipe outlines
    for pos, r_out_n in zip(pos_pipes, rp_out):
        pipe = plt.Circle(pos, radius=r_out_n,
                          fill=False, linestyle='-', linewidth=4.)
        ax1.add_patch(pipe)
    # Temperature contours
    CS = ax1.contour(X, Y, T.reshape((N_xy, N_xy)),
                     np.linspace(-0.2, 1.0, num=7))
    plt.clabel(CS, inline=1, fontsize=10)

    # Adjust to plot window
    plt.tight_layout()

    return


# Main function
if __name__ == '__main__':
    main()
