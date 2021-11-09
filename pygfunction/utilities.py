# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import warnings


def segment_ratios(nSegments, end_length_ratio=0.02):
    """
    Discretize a borehole into segments of different lengths using a
    geometrically expanding mesh from the provided end-length-ratio towards the
    middle of the borehole. Eskilson (1987) [#Eskilson_1987]_ proposed that
    segment lengths increase with a factor of sqrt(2) towards the middle of the
    borehole. Here, the expansion factor is inferred from the provided number
    of segments and end-length-ratio.

    Parameters
    ----------
    nSegments : int
        Number of line segments along the borehole.
    end_length_ratio: float, optional
        The ratio of the height of the borehole that accounts for the end
        segment lengths.
        Default is 0.02.

    Returns
    -------
    segment_ratios : array
        The segment ratios along the borehole, from top to bottom.

    Examples
    --------
    >>> gt.utilities.segment_ratios(5)
    array([0.02, 0.12, 0.72, 0.12, 0.02])

    References
    ----------
    .. [#Eskilson_1987] Eskilson, P. (1987). Thermal analysis of heat
        extraction boreholes. PhD Thesis. University of Lund, Department of
        Mathematical Physics. Lund, Sweden.
    """
    def is_even(n):
        "Returns True if n is even."
        return not(n & 0x1)
    assert nSegments >= 1 and isinstance(nSegments, int), \
            "The number of segments `nSegments` should be greater or equal " \
            "to 1 and of type int."
    assert 0. < end_length_ratio < 0.5 and \
           isinstance(end_length_ratio, (float, np.floating)), \
            "The end-length-ratio `end_length_ratio` should be greater than " \
            "0, less than 0.5 (0 < end_length_ratio < 0.5) and of type float."

    # If nSegments == 1, the only segment covers the entire length
    if nSegments == 1:
        return np.array([1.0])
    # If nSegments == 2, split the borehole in two even segments
    elif nSegments == 2:
        warnings.warn('nSegments = 2 has been provided. The '
                      '`end_length_ratio` will be over-ridden. Two segment '
                      'ratios of [0.5, 0.5] will be returned.')
        return np.array([0.5, 0.5])
    # If nSegments == 3, then the middle segment is simply the remainder of the
    # length
    elif nSegments == 3:
        segment_ratios = np.array(
            [end_length_ratio,
             1 - 2 * end_length_ratio,
             end_length_ratio])
        return segment_ratios
    else:
        pass

    # If end_length_ratio == 1 / nSegments, then the discretization is
    # uniform
    if np.abs(1. - nSegments * end_length_ratio) < 1e-6:
        return np.full(nSegments, 1 / nSegments)

    # Find the required constant expansion ratio to fill the borehole length
    # from the provided end-length-ratio inwards with the provided nSegments
    if is_even(nSegments):
        # The ratio is a root of the polynomial expression :
        # 0 = (1 - 2 * end_length_ratio)
        #     - ratio * x
        #     + 2 * end_length_ratio * x**nz
        nz = int(nSegments / 2)
        coefs = np.zeros(nz + 1)
        coefs[0] = 1 - 2 * end_length_ratio
        coefs[1] = -1
        coefs[-1] = 2 * end_length_ratio
        # Roots of the polynomial
        roots = poly.Polynomial(coefs).roots()
        # Find the correct root
        for x in roots:
            if np.isreal(x):
                factor = np.real(x)
                dz = [factor**i * end_length_ratio for i in range(nz)]
                segment_ratios = np.concatenate(
                    (dz,
                     dz[::-1]))
                if (np.abs(1. - np.sum(segment_ratios)) < 1e-6
                    and np.all(segment_ratios > 0.)):
                    break
        else:
            raise RuntimeError(
                'utilities.segment_ratios failed to generate segment '
                'discretization for the given input parameters : '
                ' nSegments={}, end_length_ratio={}.'.format(
                    nSegments, end_length_ratio))
    else:
        # The ratio is a root of the polynomial expression
        # 0 = (1 - 2 * end_length_ratio) - ratio * x
        #     + end_length_ratio * x**nz
        #     + end_length_ratio * x**(nz + 1)
        nz = int((nSegments - 1) / 2)
        coefs = np.zeros(nz + 2)
        coefs[0] = 1 - 2 * end_length_ratio
        coefs[1] = -1
        coefs[-2] = end_length_ratio
        coefs[-1] = end_length_ratio
        # Roots of the polynomial
        roots = poly.Polynomial(coefs).roots()
        # Find the correct root
        for x in roots:
            if np.isreal(x):
                factor = np.real(x)
                dz = [factor**i * end_length_ratio for i in range(nz)]
                segment_ratios = np.concatenate(
                    (dz,
                     np.array([factor**nz]) * end_length_ratio,
                     dz[::-1]))
                if (np.abs(1. - np.sum(segment_ratios)) < 1e-6
                    and np.all(segment_ratios > 0.)):
                    break
        else:
            raise RuntimeError(
                'utilities.segment_ratios failed to generate segment '
                'discretization for the given input parameters : '
                ' nSegments={}, end_length_ratio={}.'.format(
                    nSegments, end_length_ratio))

    if factor < 1.:
        warnings.warn(
            'A decreasing segment ratios discretization was found by '
            'utilities.segment_ratios(). Better accuracy is expected for '
            'increasing segment ratios. Consider decreasing the '
            'end-length-ratio such that end_length_ratio < 1 / nSegments.')

    return segment_ratios


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
