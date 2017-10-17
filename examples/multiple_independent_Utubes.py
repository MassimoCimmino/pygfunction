# -*- coding: utf-8 -*-
""" Example of calculation of fluid temperature profiles in a borehole with
    independent U-tubes.

    The fluid temperature profiles in a borehole with 4 independent U-tubes are
    calculated. The borehole has 4 U-tubes, each with different inlet fluid
    temperatures and different inlet fluid mass flow rates. The borehole wall
    temperature is uniform. Results are verified against the results of
    Cimmino (2016).

"""
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import os, sys
from scipy import pi

# Add path to pygfunction to Python path
packagePath = os.path.normpath(
        os.path.join(os.path.normpath(os.path.dirname(__file__)),
                     '..'))
sys.path.append(packagePath)

import pygfunction as gt


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    D = 2.5             # Borehole buried depth (m)
    H = 100.0           # Borehole length (m)
    r_b = 0.075         # Borehole radius (m)

    # Pipe dimensions
    rp_out = 0.010      # Pipe outer radius (m)
    rp_in = 0.008       # Pipe inner radius (m)
    D_s = 0.060         # Shank spacing (m)

    # Pipe positions
    nPipes = 4          # Number of U-tube pipes (-)
    pos_pipes = _pipePositions(D_s, nPipes)

    # Ground properties
    k_s = 2.0           # Ground thermal conductivity (W/m.K)

    # Grout properties
    k_g = 1.0           # Grout thermal conductivity (W/m.K)

    # Fluid properties
    R_fp = 0.0          # Fluid to outer pipe wall thermal resistance (m.K/W)
    # Fluid specific isobaric heat capacity per U-tube (J/kg.K)
    cp = 4000.*np.ones(nPipes)

    # Borehole wall temperature (degC)
    T_b = 2.0
    # Total fluid mass flow rate per U-tube (kg/s)
    m_flow_in = np.array([0.40, 0.35, 0.30, 0.25])
    # Inlet fluid temperatures per U-tube (degC)
    T_f_in = np.array([6.0, -6.0, 5.0, -5.0])

    # Path to validation data
    filePath = './data/Cimmi16_multiple_independent_Utubes.txt'

    # -------------------------------------------------------------------------
    # Initialize pipe model
    # -------------------------------------------------------------------------

    # Borehole object
    borehole = gt.boreholes.Borehole(H, D, r_b, 0., 0.)
    # Multiple independent U-tubes
    MultipleUTube = gt.pipes.IndependentMultipleUTube(
            pos_pipes, rp_in, rp_out, borehole, k_s, k_g, R_fp, nPipes, J=0)

    # -------------------------------------------------------------------------
    # Evaluate the outlet fluid temperatures and fluid temperature profiles
    # -------------------------------------------------------------------------

    # Calculate the outlet fluid temperatures
    T_f_out = MultipleUTube.get_outlet_temperature(T_f_in, T_b, m_flow_in, cp)

    # Evaluate temperatures at nz evenly spaced depths along the borehole
    nz = 20
    z = np.linspace(0., H, num=nz)
    T_f = MultipleUTube.get_temperature(z, T_f_in, T_b, m_flow_in, cp)

    # -------------------------------------------------------------------------
    # Plot fluid temperature profiles
    # -------------------------------------------------------------------------

    plt.rc('figure')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # Axis labels
    ax1.set_xlabel(r'Temperature (degC)')
    ax1.set_ylabel(r'Depth from borehole head (m)')
    # Plot temperatures
    ax1.plot(T_f, z, 'k.', lw=1.5)
    ax1.plot(np.array([T_b, T_b]), np.array([0., H]), 'k--', lw=1.5)
    # Labels
    calculated = mlines.Line2D([], [],
                               color='black',
                               ls='None',
                               lw=1.5,
                               marker='.',
                               label='Fluid')
    borehole_temp = mlines.Line2D([], [],
                                  color='black',
                                  ls='--',
                                  lw=1.5,
                                  marker='None',
                                  label='Borehole wall')

    # Show minor ticks
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    # Reverse y-axis
    ax1.set_ylim(ax1.get_ylim()[::-1])
    # Adjust to plot window
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Load data from Cimmino (2016)
    # -------------------------------------------------------------------------
    data = np.loadtxt(filePath, skiprows=1)
    ax1.plot(data[:,2:], data[:,0], 'b-', lw=1.5)
    reference = mlines.Line2D([], [],
                              color='blue',
                              ls='-',
                              lw=1.5,
                              marker='None',
                              label='Cimmino (2016)')
    ax1.legend(handles=[borehole_temp, calculated, reference],
               loc='upper left')

    return


def _pipePositions(Ds, nPipes):
    """ Positions pipes in an axisymetric configuration.
    """
    dt = pi / float(nPipes)
    pos = [(0., 0.) for i in range(2*nPipes)]
    for i in range(nPipes):
        pos[i] = (Ds*np.cos(2.0*i*dt+pi), Ds*np.sin(2.0*i*dt+pi))
        pos[i+nPipes] = (Ds*np.cos(2.0*i*dt+pi+dt), Ds*np.sin(2.0*i*dt+pi+dt))
    return pos


# Main function
if __name__ == '__main__':
    main()
