# -*- coding: utf-8 -*-
import warnings

import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.special import erf


def cardinal_point(direction):
    accepted_inputs = ['E', 'ENE', 'NE', 'NNE',
                       'N', 'NNW', 'NW', 'WNW',
                       'W', 'WSW', 'SW', 'SSW',
                       'S', 'SSE', 'SE', 'ESE']
    direction = direction.upper()
    if direction not in accepted_inputs:
        raise ValueError("The indicated direction {} is not encompassed in the "
                         "cardinal point function.".format(direction))
    # Create the cardinal compass
    compass = {}
    for i, pointer in enumerate(accepted_inputs):
        compass[pointer] = i * np.pi / 8.
    return compass[direction]


def erfint(x):
    """
    Integral of the error function.

    Parameters
    ----------
    x : float or array
        Argument.

    Returns
    -------
    float or array
        Integral of the error function.

    """
    return x * erf(x) - 1.0 / np.sqrt(np.pi) * (1.0 - np.exp(-x**2))


def exp1(x):
    """
    Exponential integral E1.

    Based on the approximation presented by Barry et al. [#BaPaLi2000]_.

    Parameters
    ----------
    x : float or array
        Argument.

    Returns
    -------
    E1 : float or array
        Exponential integral.

    References
    ----------
    .. [#BaPaLi2000]  Barry, D.A., Parlange, J.-Y. & Li, L. (2000).
       Approximation for the exponential integral (Theis well function).
       Journal of Hydrology, 227 (1-4): 287-291.

    """
    q = 20 / 47 * x**np.sqrt(31/26)
    G = np.exp(-np.euler_gamma)
    b = np.sqrt(2 * (1 - G) / (G * (2 - G)))
    h_inf = (1 - G) * (G**2 - 6 * G + 12.) / (3 * G * (2 - G)**2 * b)
    h = 1 / (1 + x * np.sqrt(x)) + h_inf * q / (1 + q)
    E1 = np.exp(-x) * np.log(1 + G / x - (1 - G) / (h + b * x)**2) / (
        G + (1 - G) * np.exp(-x / (1 - G)))
    return E1


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
    assert nSegments <= 2 or 0. < end_length_ratio < 0.5 and \
           isinstance(end_length_ratio, (float, np.floating)), \
            "The end-length-ratio `end_length_ratio` should be greater than " \
            "0, less than 0.5 (0 < end_length_ratio < 0.5) and of type float."

    # If nSegments == 1, the only segment covers the entire length
    if nSegments == 1:
        return np.array([1.0])
    # If nSegments == 2, split the borehole in two even segments
    elif nSegments == 2:
        if not np.abs(end_length_ratio - 0.5) < 1e-6:
            warnings.warn('nSegments = 2 has been provided. The '
                          '`end_length_ratio` will be over-ridden. Two '
                          'segment ratios of [0.5, 0.5] will be returned.')
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
                f'utilities.segment_ratios failed to generate segment '
                f'discretization for the given input parameters : '
                f'nSegments={nSegments}, end_length_ratio={end_length_ratio}.')
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
                f'utilities.segment_ratios failed to generate segment '
                f'discretization for the given input parameters : '
                f'nSegments={nSegments}, end_length_ratio={end_length_ratio}.')

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
    from ._mpl import plt

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
    from ._mpl import AutoMinorLocator

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
    # Draw major and minor tick marks inwards
    ax.tick_params(
        axis='both', which='major', direction='in',
        bottom=True, top=True, left=True, right=True)
    # Auto-adjust minor tick marks
    # ax.xaxis.set_minor_locator(AutoMinorLocator())
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # ax.zaxis.set_minor_locator(AutoMinorLocator())
    return


# Coefficients of the approximation of the error function from Tanash and
# Riihonen (2020).
_a_Q = [np.array([0.5]),
np.array([0.38896664, 0.11103336]),
np.array([0.33527962, 0.12810593, 0.03661445]),
np.array([0.30030983, 0.13487919, 0.05013556, 0.01467542]),
np.array([0.27479399, 0.13748584, 0.05862135, 0.02248433, 0.00661449]),
np.array([0.25503407, 0.13802966, 0.0643485, 0.02837111, 0.0109824, 0.00323426]),
np.array([0.23911489, 0.13748632, 0.06838034, 0.03292679, 0.01470502, 0.00570587, 0.00168077]),
np.array([0.22591697, 0.13634258, 0.07131091, 0.03652669, 0.01786631, 0.00801018, 0.00311014, 0.00091621]),
np.array([0.21473387, 0.13486132, 0.07348597, 0.03943108, 0.02055793, 0.01011012, 0.00453815, 0.00176237, 0.00051918]),
np.array([0.20509396, 0.1331934 , 0.07511994, 0.04181489, 0.02286797, 0.01200445, 0.00591433, 0.00265578, 0.00103142, 0.00030385]),
np.array([1.96668237e-01, 1.31429201e-01, 7.63533671e-02, 4.37988006e-02, 2.48670891e-02, 1.37105484e-02, 7.21518244e-03, 3.55697135e-03, 1.59742839e-03, 6.20405304e-04, 1.82769115e-04]),
np.array([1.89218959e-01, 1.29624502e-01, 7.72824770e-02, 4.54686466e-02, 2.66106820e-02, 1.52489276e-02, 8.43419941e-03, 4.44252880e-03, 2.19058207e-03, 9.83829957e-04, 3.82100123e-04, 1.12565366e-04]),
np.array([1.82569305e-01, 1.27814364e-01, 7.79755534e-02, 4.68871585e-02, 2.81422454e-02, 1.66393673e-02, 9.57165126e-03, 5.30057319e-03, 2.79291283e-03, 1.37728068e-03, 6.18571899e-04, 2.40241728e-04, 7.07743946e-05]),
np.array([1.76584557e-01, 1.26021005e-01, 7.84824737e-02, 4.81012603e-02, 2.94961900e-02, 1.78997594e-02, 1.06309534e-02, 6.12490957e-03, 3.39347815e-03, 1.78828708e-03, 8.81893551e-04, 3.96083441e-04, 1.53831535e-04, 4.53182508e-05]),
np.array([1.71159961e-01, 1.24258455e-01, 7.88405624e-02, 4.91467816e-02, 3.06999814e-02, 1.90458340e-02, 1.16169465e-02, 6.91265586e-03, 3.98520556e-03, 2.20841598e-03, 1.16384796e-03, 5.73958822e-04, 2.57781871e-04, 1.00117785e-04, 2.94943626e-05]),
np.array([1.66212632e-01, 1.22535416e-01, 7.90783215e-02, 5.00515783e-02, 3.17757508e-02, 2.00912371e-02, 1.25349870e-02, 7.66294319e-03, 4.56353949e-03, 2.63162740e-03, 1.45844437e-03, 7.68625638e-04, 3.79054499e-04, 1.70244761e-04, 6.61199763e-05, 1.94787228e-05]),
np.array([1.61675984e-01, 1.20857065e-01, 7.92178935e-02, 5.08376544e-02, 3.27415005e-02, 2.10477375e-02, 1.33904628e-02, 8.37612314e-03, 5.12563255e-03, 3.05356264e-03, 1.76108222e-03, 9.76023394e-04, 5.14386238e-04, 2.53674676e-04, 1.13932968e-04, 4.42495009e-05, 1.30357543e-05]),
np.array([1.57495813e-01, 1.19226223e-01, 7.92767352e-02, 5.15226387e-02, 3.36120087e-02, 2.19254680e-02, 1.41885448e-02, 9.05327242e-03, 5.66980754e-03, 3.47109109e-03, 2.06820289e-03, 1.19285611e-03, 6.61112401e-04, 3.48422527e-04, 1.71828197e-04, 7.71732546e-05, 2.99726950e-05, 8.82985530e-06]),
np.array([1.53627474e-01, 1.17644127e-01, 7.92687848e-02, 5.21208339e-02, 3.43995123e-02, 2.27331580e-02, 1.49340669e-02, 9.69588035e-03, 6.19519161e-03, 3.88198666e-03, 2.37705668e-03, 1.41643711e-03, 8.16962629e-04, 4.52785417e-04, 2.38629519e-04, 1.17682685e-04, 5.28548678e-05, 2.05278741e-05, 6.04744278e-06]),
np.array([1.50033809e-01, 1.16110948e-01, 7.92052948e-02, 5.26439756e-02, 3.51142271e-02, 2.34783431e-02, 1.56314796e-02, 1.03056504e-02, 6.70146463e-03, 4.28469271e-03, 2.68552435e-03, 1.64458087e-03, 9.80002388e-04, 5.65244865e-04, 3.13276809e-04, 1.65105050e-04, 8.14233279e-05, 3.65696909e-05, 1.42030062e-05, 4.18415798e-06]),
np.array([1.46683603e-01, 1.14626141e-01, 7.90954371e-02, 5.31017916e-02, 3.57647474e-02, 2.41675463e-02, 1.62848424e-02, 1.08843745e-02, 7.18868427e-03, 4.67814996e-03, 2.99197923e-03, 1.87551307e-03, 1.14859099e-03, 6.84453518e-04, 3.94780424e-04, 2.18800279e-04, 1.15313498e-04, 5.68680943e-05, 2.55411908e-05, 9.91973698e-06, 2.92232130e-06]),
np.array([1.43550410e-01, 1.13188689e-01, 7.89467519e-02, 5.35024187e-02, 3.63583546e-02, 2.48064324e-02, 1.68978380e-02, 1.14338539e-02, 7.65716384e-03, 5.06166959e-03, 3.29518022e-03, 2.10779583e-03, 1.32134163e-03, 8.09225621e-04, 4.82226839e-04, 2.78140410e-04, 1.54154672e-04, 8.12436032e-05, 4.00662071e-05, 1.79950304e-05, 6.98908441e-06, 2.05922993e-06]),
np.array([1.40611651e-01, 1.11797270e-01, 7.87654842e-02, 5.38527210e-02, 3.69012618e-02, 2.53999394e-02, 1.74737973e-02, 1.19558501e-02, 8.10738541e-03, 5.43483905e-03, 3.59418932e-03, 2.34026692e-03, 1.49708548e-03, 9.38524175e-04, 5.74783762e-04, 3.42521474e-04, 1.97560912e-04, 1.09494904e-04, 5.77068046e-05, 2.84589040e-05, 1.27820425e-05, 4.96480326e-06, 1.46355998e-06]),
np.array([1.37847922e-01, 1.10450380e-01, 7.85568414e-02, 5.41585303e-02, 3.73987982e-02, 2.59523825e-02, 1.80157254e-02, 1.24520552e-02, 8.53994009e-03, 5.79745153e-03, 3.88830651e-03, 2.57198922e-03, 1.67483888e-03, 1.07144549e-03, 6.71699504e-04, 4.11373469e-04, 2.45143495e-04, 1.41394927e-04, 7.83658119e-05, 4.13008575e-05, 2.03679647e-05, 9.14787205e-06, 3.55286785e-06, 1.04666292e-06]),
np.array([1.35242468e-01, 1.09146424e-01, 7.83251937e-02, 5.44248334e-02, 3.78555618e-02, 2.64675469e-02, 1.85263310e-02, 1.29240698e-02, 8.95547615e-03, 6.14944718e-03, 4.17701546e-03, 2.80220699e-03, 1.85377306e-03, 1.20720279e-03, 7.72298434e-04, 4.84164314e-04, 2.96520688e-04, 1.76701163e-04, 1.01918520e-04, 5.64866191e-05, 2.97698937e-05, 1.46813183e-05, 6.59377671e-06, 2.56091315e-06, 7.54429001e-07])]

_b_Q = [np.array([1.4751134849381495]),
np.array([0.86508615, 26.74445877]),
np.array([0.72369789, 6.56251898, 241.88951974]),
np.array([6.61598113e-01, 3.58507653e+00, 4.06543950e+01, 1.50259915e+03]),
np.array([6.26675417e-01, 2.51523207e+00, 1.74704549e+01, 2.00130058e+02, 7.39412740e+03]),
np.array([6.04267679e-01, 1.98270748e+00, 1.03318189e+01, 7.30667703e+01, 8.37106164e+02, 3.09244471e+04]),
np.array([5.88652923e-01, 1.66838620e+00, 7.14997233e+00, 3.82243005e+01, 2.70591171e+02, 3.09970819e+03, 1.14505983e+05]),
np.array([5.77138909e-01, 1.46235483e+00, 5.42341290e+00, 2.40129665e+01, 1.28650391e+02, 9.10672296e+02, 1.04315510e+04, 3.85347387e+05]),
np.array([5.68292223e-01, 1.31739756e+00, 4.36474383e+00, 1.68305504e+01, 7.47796509e+01, 4.00670467e+02, 2.83606494e+03, 3.24860046e+04, 1.20004827e+06]),
np.array([5.61279078e-01, 1.21008463e+00, 3.65938357e+00, 1.26773324e+01, 4.91263804e+01, 2.18338000e+02, 1.16981070e+03, 8.28009812e+03, 9.48447839e+04, 3.50360802e+06]),
np.array([5.55581123e-01, 1.12753963e+00, 3.16038928e+00, 1.00435965e+01, 3.50205685e+01, 1.35786493e+02, 6.03483129e+02, 3.23326442e+03, 2.28853649e+04, 2.62141075e+05, 9.68360394e+06]),
np.array([5.50858966e-01, 1.06212559e+00, 2.79106923e+00, 8.25766830e+00, 2.64546133e+01, 9.23255146e+01, 3.57987810e+02, 1.59098809e+03, 8.52390794e+03, 6.03328955e+04, 6.91084483e+05, 2.55289553e+07]),
np.array([5.46880974e-01, 1.00903726e+00, 2.50793237e+00, 6.98358314e+00, 2.08614297e+01, 6.69165938e+01, 2.33557562e+02, 9.05592305e+02, 4.02463435e+03, 2.15623730e+04, 1.52620025e+05, 1.74818906e+06, 6.45788464e+07]),
np.array([5.43483589e-01, 9.65105077e-01, 2.28467213e+00, 6.03785278e+00, 1.70020329e+01, 5.08730855e+01, 1.63211843e+02, 5.69650299e+02, 2.20872597e+03, 9.81597342e+03, 5.25899577e+04, 3.72235348e+05, 4.26377669e+06, 1.57505720e+08]),
np.array([5.40547997e-01, 9.28156233e-01, 2.10453418e+00, 5.31316070e+00, 1.42210313e+01, 4.01291491e+01, 1.20105287e+02, 3.85326438e+02, 1.34487038e+03, 5.21448432e+03, 2.31740523e+04, 1.24156985e+05, 8.78791556e+05, 1.00661338e+07, 3.71847252e+08]),
np.array([5.37985789e-01, 8.96652683e-01, 1.95638816e+00, 4.74319987e+00, 1.21460117e+01, 3.25928353e+01, 9.20054866e+01, 2.75377156e+02, 8.83467203e+02, 3.08346629e+03, 1.19555341e+04, 5.31323727e+04, 2.84661195e+05, 2.01485110e+06, 2.30791482e+07, 8.52553520e+08]),
np.array([5.35729805e-01, 8.69475812e-01, 1.83257556e+00, 4.28511737e+00, 1.05528354e+01, 2.71050303e+01, 7.27704479e+01, 2.05432895e+02, 6.14867611e+02, 1.97260977e+03, 6.88475630e+03, 2.66942568e+04, 1.18633650e+05, 6.35589774e+05, 4.49874709e+06, 5.15309790e+07, 1.90357621e+09]),
np.array([5.33728106e-01, 8.45792970e-01, 1.72766783e+00, 3.91015978e+00, 9.30008641e+00, 2.29840397e+01, 5.90722445e+01, 1.58608247e+02, 4.47755113e+02, 1.34013739e+03, 4.29939475e+03, 1.50056243e+04, 5.81812571e+04, 2.58566993e+05, 1.38529437e+06, 9.80520645e+06, 1.12313912e+08, 4.14892355e+09]),
np.array([5.31939889e-01, 8.24971801e-01, 1.63771800e+00, 3.59841172e+00, 8.29495341e+00, 1.98088089e+01, 4.89935581e+01, 1.25935868e+02, 3.38138670e+02, 9.54568527e+02, 2.85702593e+03, 9.16582180e+03, 3.19902665e+04, 1.24035724e+05, 5.51234936e+05, 2.95328739e+06, 2.09035660e+07, 2.39440270e+08, 8.84502510e+09]),
np.array([5.30332650e-01, 8.06523567e-01, 1.55979241e+00, 3.33569866e+00, 7.47445459e+00, 1.73083833e+01, 4.13724934e+01, 1.02344439e+02, 2.63076444e+02, 7.06358971e+02, 1.99404968e+03, 5.96818340e+03, 1.91469242e+04, 6.68259783e+04, 2.59104051e+05, 1.15150051e+06, 6.16926048e+06, 4.36664387e+07, 5.00177999e+08, 1.84767871e+10]),
np.array([5.28880175e-01, 7.90064646e-01, 1.49166818e+00, 3.11169123e+00, 6.79460687e+00, 1.53022232e+01, 3.54744802e+01, 8.48133183e+01, 2.09811327e+02, 5.39318941e+02, 1.44806292e+03, 4.08786889e+03, 1.22349644e+04, 3.92517841e+04, 1.36995296e+05, 5.31171201e+05, 2.36061113e+06, 1.26471714e+07, 8.95175167e+07, 1.02537996e+09, 3.78779667e+10]),
np.array([5.27561086e-01, 7.75289770e-01, 1.43163225e+00, 2.91870531e+00, 6.22392126e+00, 1.36663812e+01, 3.08180875e+01, 7.14633096e+01, 1.70863574e+02, 4.22682886e+02, 1.08650017e+03, 2.91722891e+03, 8.23530196e+03, 2.46481913e+04, 7.90754456e+04, 2.75986521e+05, 1.07008124e+06, 4.75561505e+06, 2.54786070e+07, 1.80339276e+08, 2.06569949e+09, 7.63078699e+10]),
np.array([5.26357773e-01, 7.61953003e-01, 1.37834431e+00, 2.75091946e+00, 5.73935690e+00, 1.23135225e+01, 2.70778648e+01, 6.10812250e+01, 1.41648012e+02, 3.38671568e+02, 8.37804176e+02, 2.15355779e+03, 5.78224688e+03, 1.63232020e+04, 4.88551970e+04, 1.56735488e+05, 5.47033061e+05, 2.12100871e+06, 9.42610747e+06, 5.05011626e+07, 3.57450664e+08, 4.09442418e+09, 1.51249719e+11]),
np.array([5.25255604e-01, 7.49854068e-01, 1.33074174e+00, 2.60385218e+00, 5.32372895e+00, 1.11806970e+01, 2.40278084e+01, 5.28584810e+01, 1.19245289e+02, 2.76533587e+02, 6.61173063e+02, 1.63560280e+03, 4.20427626e+03, 1.12883658e+04, 3.18668911e+04, 9.53773075e+04, 3.05986021e+05, 1.06794226e+06, 4.14072713e+06, 1.84020642e+07, 9.85906029e+07, 6.97830919e+08, 7.99331715e+09, 2.95276609e+11]),
np.array([5.24242327e-01, 7.38828327e-01, 1.28797215e+00, 2.47400433e+00, 4.96400225e+00, 1.02216181e+01, 2.15072525e+01, 4.62410786e+01, 1.01735082e+02, 2.29511118e+02, 5.32243145e+02, 1.27255497e+03, 3.14803038e+03, 8.09193378e+03, 2.17266328e+04, 6.13340152e+04, 1.83572221e+05, 5.88930254e+05, 2.05546784e+06, 7.96968224e+06, 3.54186923e+07, 1.89760150e+08, 1.34314020e+09, 1.53849689e+10, 5.68317516e+11])]

_a_erf = [np.array([1.] + [-2*a for a in a_Qi]) for a_Qi in _a_Q]
_b_erf = [np.array([0.] + [b*2 for b in b_Qi]) for b_Qi in _b_Q]


def _erf_coeffs(N):
    """
    Return the coefficients of the approximation of the error function.

    This returns the coefficients (a_n, b_n) of the approximation of the error
    function adapted from the work of Tanash and Riihonen (2020)
    [#erf-TanRii2020]_. The approximation of the error function is given by:

        .. math::
            erf(x) \\approx \\sum_{n=0}^{N} a_n exp(-b_n^2 x^2)

    for `x > 0`.

    Parameters
    ----------
    N : int
        Number of terms in the approximation of the error function.

    Returns
    -------
    a : array
        Array of coefficients a_n.
    b : array
        Array of coefficients b_n.

    References
    ----------
    .. [#erf-TanRii2020] Tanash, I.M., & Riihonen, T. (2020). Global Minimax
        approximations and bounds for the Gaussian Q-function by sums of
        exponentials. IEEE Transactions on communications, 68(10), 6514-6524.
    """
    return _a_erf[N], _b_erf[N]
