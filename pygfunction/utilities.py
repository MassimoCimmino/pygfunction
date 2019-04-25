from __future__ import absolute_import, division, print_function

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
