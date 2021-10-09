# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def time_ClaessonJaved(dt, tmax, cells_per_level=5):
    """
    Build a time vector of expanding cell width following the method of
    Claesson and Javed [#ClaessonJaved2012]_.

    Parameters
    ----------
    dt : float
        Simulation time step (in seconds).
    tmax : float
        Maximum simulation time (in seconds).
    cells_per_level : int, optional
        Number of time steps cells per level. Cell widths double every
        cells_per_level cells.
        Default is 5.

    Returns
    -------
    time : array
        Time vector.

    Examples
    --------
    >>> time = gt.utilities.time_ClaessonJaved(3600., 12*3600.)
    array([3600.0, 7200.0, 10800.0, 14400.0, 18000.0, 25200.0, 32400.0,
           39600.0, 46800.0])

    References
    ----------
    .. [#ClaessonJaved2012] Claesson, J., & Javed, S. (2012). A
       load-aggregation method to calculate extraction temperatures of
       borehole heat exchangers. ASHRAE Transactions, 118 (1): 530-539.

    """
    # Initialize time (t), time vector (time) and cell count(i)
    t = 0.0
    i = 0
    time = []
    while t < tmax:
        # Increment cell count
        i += 1
        # Cell size doubles every cells_per_level time steps
        v = np.ceil(i / cells_per_level)
        width = 2.0**(v-1)
        t += width*float(dt)
        # Append time vector
        time.append(t)
    time = np.array(time)

    return time


def time_MarcottePasquier(dt, tmax, non_expanding_cells=48):
    """
    Build a time vector of expanding cell width following the method of
    Marcotte and Pasquier [#MarcottePasquier2008]_.

    Parameters
    ----------
    dt : float
        Simulation time step (in seconds).
    tmax : float
        Maximum simulation time (in seconds).
    non_expanding_cells : int, optional
        Number of cells before geomteric expansion starts.
        Default is 48.

    Returns
    -------
    time : array
        Time vector.

    Examples
    --------
    >>> time = gt.utilities.time_MarcottePasquier(3600., 13*3600.,
                                                  non_expanding_cells=6)
    array([3600., 7200., 10800., 14400., 18000., 21600., 28800., 43200.,
           72000.])

    References
    ----------
    .. [#MarcottePasquier2008] Marcotte, D., & Pasquier, P. (2008). Fast fluid
       and ground temperature computation for geothermal ground-loop heat
       exchanger systems. Geothermics, 37: 651-665.

    """
    # Initialize time (t), time vector (time), cell width and cell count(i)
    t = 0.0
    i = 0
    width = 1
    time = []
    while t < tmax:
        # Increment cell count
        i += 1
        if i > non_expanding_cells:
            width = 2*width
        t += width*float(dt)
        # Append time vector
        time.append(t)
    time = np.array(time)

    return time


def time_geometric(dt, tmax, Nt):
    """
    Build a time vector of geometrically expanding cell width.

    Parameters
    ----------
    dt : float
        Simulation time step (in seconds).
    tmax : float
        Maximum simulation time (in seconds).
    Nt : int
        Total number of time steps.

    Returns
    -------
    time : array
        Time vector.

    Examples
    --------
    >>> time = gt.utilities.time_geometric(3600., 13*3600., 5)
    array([3600., 8971.99474335, 16988.19683297, 28950.14002383, 46800.])

    """
    if tmax > Nt*dt:
        # Identify expansion rate (r)
        dr = 1.0e99
        r = 2.
        while np.abs(dr) > 1.0e-10:
            dr = (1+tmax/dt*(r-1))**(1/Nt) - r
            r += dr
        time = np.array([dt*(1-r**(j+1))/(1-r) for j in range(Nt)])
        time[-1] = tmax
    else:
        time = np.array([dt*(j+1) for j in range(Nt)])

    return time


def _initialize_figure():
    """
    Initialize a matplotlib figure object with overwritten default parameters.

    Returns
    -------
    fig : figure
        Figure object (matplotlib).

    """
    plt.rc('font', size=9)
    plt.rc('xtick', labelsize=9)
    plt.rc('ytick', labelsize=9)
    plt.rc('lines', lw=1.5, markersize=5.0)
    plt.rc('savefig', dpi=500)
    fig = plt.figure()
    return fig


def _format_axes(ax):
    """
    Adjust axis parameters.

    Parameters
    ----------
    ax : axis
        Axis object (matplotlib).

    """
    from matplotlib.ticker import AutoMinorLocator
    # Draw major and minor tick marks inwards
    ax.tick_params(
        axis='both', which='both', direction='in',
        bottom=True, top=True, left=True, right=True)
    # Auto-adjust minor tick marks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    return


def _format_axes_3d(ax):
    """
    Adjust axis parameters.

    Parameters
    ----------
    ax : axis
        Axis object (matplotlib).

    """
    from matplotlib.ticker import AutoMinorLocator
    # Draw major and minor tick marks inwards
    ax.tick_params(
        axis='both', which='major', direction='in',
        bottom=True, top=True, left=True, right=True)
    # Auto-adjust minor tick marks
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # ax.zaxis.set_minor_locator(AutoMinorLocator())
    return


def discretize(H, end_length_ratio=0.05, ratio=np.sqrt(2.)):
    """
    Discretize a borehole into segments of differing length. Eskilson (1987)
    [#Eskilson_1987]_ proposed that segment lengths increase with a factor of
    sqrt(2) towards the center of the borehole, and is implemented here. The
    segments are symmetric about the center of the borehole.

    Parameters
    ----------
    H : float
         Borehole length (in meters).
    end_length_ratio: float
        The ratio of the height of the borehole that accounts for the end
        segment lengths.
    ratio: float
        The ratio of segment length increase towards the center of the borehole.

    Returns
    -------
    segment_ratios : list
        The segment ratios along the borhole from top to bottom.

    Examples
    --------
    >>> H = 150. # Borehole length (m)
    >>> segment_ratios = gt.utilities.discretize(H)

    References
    ----------
    .. [#Eskilson_1987] Eskilson, P. (1987). Thermal analysis of heat extraction
        boreholes. PhD Thesis. University of Lund, Department of Mathematical
        Physics. Lund, Sweden.
    """
    H_half = H / 2.
    end_length = H * end_length_ratio

    segment_lengths = []
    nSegments = 0
    length = end_length
    while (sum(segment_lengths) + length) <= H_half:

        segment_lengths.append(length)
        length *= ratio

        nSegments += 1

    diff = H_half - sum(segment_lengths)
    segment_lengths.append(diff * 2.)

    segment_lengths = segment_lengths + list(reversed(segment_lengths[0:len(segment_lengths)-1]))

    nSegments = (nSegments * 2) + 1
    assert nSegments == len(segment_lengths)

    assert (sum(segment_lengths) - H) <= 1.0e-08

    segment_ratios = [segment_lengths[i] / H for i in range(len(segment_lengths))]

    assert (sum(segment_ratios) - 1.0) <= 1.0e-8

    return segment_ratios
