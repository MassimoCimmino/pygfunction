# -*- coding: utf-8 -*-
from typing import Union, List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad_vec

from ..borefield import Borefield
from ..boreholes import Borehole
from .finite_line_source_vertical import finite_line_source_vertical
from .finite_line_source_inclined import finite_line_source_inclined
from ..utilities import erfint


def finite_line_source(
        time: npt.ArrayLike,
        alpha: float,
        borefield_j: Union[Borehole, Borefield, List[Borehole]],
        borefield_i: Union[Borehole, Borefield, List[Borehole]],
        outer: bool = True,
        reaSource: bool = True,
        imgSource: bool = True,
        approximation: bool = False,
        M: int = 11,
        N: int = 10
        ) -> np.ndarray:
    """
    Evaluate the Finite Line Source (FLS) solution.

    This function uses a numerical quadrature to evaluate the one-integral form
    of the FLS solution. For vertical boreholes, the FLS solution was proposed
    by Claesson and Javed [#FLS-ClaJav2011]_ and extended to boreholes with
    different vertical positions by Cimmino and Bernier [#FLS-CimBer2014]_.
    The FLS solution is given by:

        .. math::
            h_{ij}(t) &= \\frac{1}{2H_i}
            \\int_{\\frac{1}{\\sqrt{4\\alpha t}}}^{\\infty}
            e^{-d_{ij}^2s^2}(I_{real}(s)+I_{imag}(s))ds


            d_{ij} &= \\sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}


            I_{real}(s) &= erfint((D_i-D_j+H_i)s) - erfint((D_i-D_j)s)

            &+ erfint((D_i-D_j-H_j)s) - erfint((D_i-D_j+H_i-H_j)s)

            I_{imag}(s) &= erfint((D_i+D_j+H_i)s) - erfint((D_i+D_j)s)

            &+ erfint((D_i+D_j+H_j)s) - erfint((D_i+D_j+H_i+H_j)s)


            erfint(X) &= \\int_{0}^{X} erf(x) dx

                      &= Xerf(X) - \\frac{1}{\\sqrt{\\pi}}(1-e^{-X^2})

    For inclined boreholes, the FLS solution was proposed by Lazzarotto
    [#FLS-Lazzar2016]_ and Lazzarotto and Björk [#FLS-LazBjo2016]_.
    The FLS solution is given by:

        .. math::
            h_{1\\rightarrow2}(t) &= \\frac{H_j}{2H_i}
            \\int_{\\frac{1}{\\sqrt{4\\alpha t}}}^{\\infty}
            \\frac{1}{s}
            \\int_{0}^{1} (I_{real}(u, s)+I_{imag}(u, s)) du ds


            I_{real}(u, s) &=
            e^{-((x_i - x_j)^2 + (y_i - y_j)^2 + (D_i - D_j)^2) s^2}

            &\\cdot (erf((u H_j k_{0,real} + k_{2,real}) s)
             - erf((u H_j k_{0,real} + k_{2,real} - H_i) s))

            &\\cdot  e^{(u^2 H_j^2 (k_{0,real}^2 - 1)
                + 2 u H_j (k_{0,real} k_{2,real} - k_{1,real}) + k_{2,real}^2) s^2}
            du ds


            I_{imag}(u, s) &=
            -e^{-((x_i - x_j)^2 + (y_i - y_j)^2 + (D_i + D_j)^2) s^2}

            &\\cdot (erf((u H_j k_{0,imag} + k_{2,imag}) s)
             - erf((u H_j k_{0,imag} + k_{2,imag} - H_i) s))

            &\\cdot e^{(u^2 H_j^2 (k_{0,imag}^2 - 1)
                + 2 u H_j (k_{0,imag} k_{2,imag} - k_1) + k_{2,imag}^2) s^2}
            du ds


            k_{0,real} &=
            sin(\\beta_j) sin(\\beta_i) cos(\\theta_j - \\theta_i)
            + cos(\\beta_j) cos(\\beta_i)


            k_{0,imag} &=
            sin(\\beta_j) sin(\\beta_i) cos(\\theta_j - \\theta_i)
            - cos(\\beta_j) cos(\\beta_i)


            k_{1,real} &= sin(\\beta_j)
            (cos(\\theta_j) (x_j - x_i) + sin(\\theta_j) (y_j - y_i))
            + cos(\\beta_j) (D_j - D_i)


            k_{1,imag} &= sin(\\beta_j)
            (cos(\\theta_j) (x_j - x_i) + sin(\\theta_j) (y_j - y_i))
            + cos(\\beta_j) (D_i + D_j)


            k_{2,real} &= sin(\\beta_i)
            (cos(\\theta_i) (x_j - x_i) + sin(\\theta_i) (y_j - y_i))
            + cos(\\beta_i) (D_j - D_i)


            k_{2,imag} &= sin(\\beta_i)
            (cos(\\theta_i) (x_j - x_i) + sin(\\theta_i) (y_j - y_i))
            - cos(\\beta_i) (D_i + D_j)

    where :math:`\\beta_j` and :math:`\\beta_i` are the tilt angle of the
    boreholes (relative to vertical), and :math:`\\theta_j` and
    :math:`\\theta_i` are the orientation of the boreholes (relative to the
    x-axis).

        .. Note::
            The reciprocal thermal response factor
            :math:`h_{ji}(t)` can be conveniently calculated by:

                .. math::
                    h_{ji}(t) = \\frac{H_i}{H_j}
                    h_{ij}(t)

    Parameters
    ----------
    time : float or (nTimes,) array
        Value of time (in seconds) for which the FLS solution is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    borefield_j : Borehole or Borefield object
        Borehole or Borefield object of the boreholes extracting heat.
    borefield_i : Borehole or Borefield object
        Borehole or Borefield object object for which the FLS is evaluated.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
        Default is True.
    imgSource : bool, optional
        True if the image part of the FLS solution is to be included.
        Default is True.
    outer : bool, optional
        True if the finite line source is to be evaluated for all boreholes_j
        onto all boreholes_i to return a (nBoreholes_i, nBoreholes_j, nTime,)
        array. If false, the finite line source is evaluated pairwise between
        boreholes_j and boreholes_i. The numbers of boreholes should be the
        same (i.e. nBoreholes_j == nBoreholes_i) and a (nBoreholes, nTime,)
        array is returned.
        Default is True.
    approximation : bool, optional
        Set to true to use the approximation of the FLS solution of Cimmino
        (2021) [#FLS-Cimmin2021]_. This approximation does not require
        the numerical evaluation of any integral.
        Default is False.
    M : int, optional
        Number of Gauss-Legendre sample points for the quadrature over
        :math:`u`. This is only used for inclined boreholes.
        Default is 11.
    N : int, optional
        Number of terms in the approximation of the FLS solution. This
        parameter is unused if `approximation` is set to False.
        Default is 10. Maximum is 25.

    Returns
    -------
    h : float or array, shape (nBoreholes_i, nBoreholes_j, nTime,), (nBoreholes, nTime,) or (nTime,)
        Value of the FLS solution. The average (over the length) temperature
        drop on the wall of borehole_i due to heat extracted from borehole_j
        is:

        .. math:: \\Delta T_{b,i} = T_g - \\frac{Q_j}{2\\pi k_s H_j} h

    Notes
    -----
    The function returns a float if time is a float and both of borehole_j and
    borehole_i are Borehole objects. If time is a float and any of borehole_j
    and borehole_i are Borefield objects, the function returns an array. If
    time is an array and both of borehole_j and borehole_i are Borehole
    objects, the function returns an  1d array, shape (nTime,). If time is an
    array and any of borehole_j and borehole_i are are Borefield objects, the
    function returns an array.

    Examples
    --------
    >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
    >>> b2 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
    >>> h = gt.heat_transfer.finite_line_source(4*168*3600., 1.0e-6, b1, b2)
    h = 0.0110473635393
    >>> h = gt.heat_transfer.finite_line_source(
        4*168*3600., 1.0e-6, b1, b2, approximation=True, N=10)
    h = 0.0110474667731
    >>> b3 = gt.boreholes.Borehole(
        H=150., D=4., r_b=0.075, x=5., y=0., tilt=3.1415/15, orientation=0.)
    >>> h = gt.heat_transfer.finite_line_source(
        4*168*3600., 1.0e-6, b1, b3, M=21)
    h = 0.0002017450051

    References
    ----------
    .. [#FLS-ClaJav2011] Claesson, J., & Javed, S. (2011). An analytical
       method to calculate borehole fluid temperatures for time-scales from
       minutes to decades. ASHRAE Transactions, 117(2), 279-288.
    .. [#FLS-CimBer2014] Cimmino, M., & Bernier, M. (2014). A
       semi-analytical method to generate g-functions for geothermal bore
       fields. International Journal of Heat and Mass Transfer, 70, 641-650.
    .. [#FLS-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.
    .. [#FLS-Lazzar2016] Lazzarotto, A. (2016). A methodology for the
       calculation of response functions for geothermal fields with
       arbitrarily oriented boreholes – Part 1, Renewable Energy, 86,
       1380-1393.
    .. [#FLS-LazBjo2016] Lazzarotto, A., & Björk, F. (2016). A methodology for
       the calculation of response functions for geothermal fields with
       arbitrarily oriented boreholes – Part 2, Renewable Energy, 86,
       1353-1361.

    """
    if np.any(borefield_j.is_tilted) or np.any(borefield_i.is_tilted):
        # Inclined boreholes
        h = finite_line_source_inclined(
            time, alpha, borefield_j, borefield_i, reaSource=reaSource,
            imgSource=imgSource, outer=outer, M=M, N=N)
    else:
        # Vertical boreholes
        h = finite_line_source_vertical(
            time, alpha, borefield_j, borefield_i,
            reaSource=reaSource, imgSource=imgSource, outer=outer, N=N)
    return h


def finite_line_source_equivalent_boreholes_vectorized(
        time, alpha, dis, wDis, H1, D1, H2, D2, N2, reaSource=True, imgSource=True):
    """
    Evaluate the equivalent Finite Line Source (FLS) solution.

    This function uses a numerical quadrature to evaluate the one-integral form
    of the FLS solution, as proposed by Prieto and Cimmino
    [#eqFLSVec-PriCim2021]_. The equivalent FLS solution is given by:

        .. math::
            h_{1\\rightarrow2}(t) &= \\frac{1}{2 H_2 N_{b,2}}
            \\int_{\\frac{1}{\\sqrt{4\\alpha t}}}^{\\infty}
            \\sum_{G_1} \\sum_{G_2}
            e^{-d_{12}^2s^2}(I_{real}(s)+I_{imag}(s))ds


            I_{real}(s) &= erfint((D_2-D_1+H_2)s) - erfint((D_2-D_1)s)

            &+ erfint((D_2-D_1-H_1)s) - erfint((D_2-D_1+H_2-H_1)s)

            I_{imag}(s) &= erfint((D_2+D_1+H_2)s) - erfint((D_2+D_1)s)

            &+ erfint((D_2+D_1+H_1)s) - erfint((D_2+D_1+H_2+H_1)s)


            erfint(X) &= \\int_{0}^{X} erf(x) dx

                      &= Xerf(X) - \\frac{1}{\\sqrt{\\pi}}(1-e^{-X^2})

        .. Note::
            The reciprocal thermal response factor
            :math:`h_{2\\rightarrow1}(t)` can be conveniently calculated by:

                .. math::
                    h_{2\\rightarrow1}(t) = \\frac{H_2 N_{b,2}}{H_1 N_{b,1}}
                    h_{1\\rightarrow2}(t)

    Parameters
    ----------
    time : float or array, shape (K)
        Value of time (in seconds) for which the FLS solution is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    dis : array
        Unique radial distances to evaluate the FLS solution.
    wDis : array
        Number of instances of each unique radial distances.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    N2 : float or array,
        Number of segments represented by the receiving heat sources.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
        Default is True.
    imgSource : bool
        True if the image part of the FLS solution is to be included.
        Default is True.

    Returns
    -------
    h : float
        Value of the FLS solution. The average (over the length) temperature
        drop on the wall of borehole2 due to heat extracted from borehole1 is:

        .. math:: \\Delta T_{b,2} = T_g - \\frac{Q_1}{2\\pi k_s H_2} h

    Notes
    -----
    This is a vectorized version of the :func:`finite_line_source` function
    using scipy.integrate.quad_vec to speed up calculations. All arrays
    (dis, H1, D1, H2, D2) must follow numpy array broadcasting rules. If time
    is an array, the integrals for different time values are stacked on the
    last axis.


    References
    ----------
    .. [#eqFLSVec-PriCim2021] Prieto, C., & Cimmino, M.
       (2021). Thermal interactions in large irregular fields of geothermal
       boreholes: the method of equivalent borehole. Journal of Building
       Performance Simulation, 14 (4), 446-460.

    """
    # Integrand of the finite line source solution
    f = _finite_line_source_equivalent_boreholes_integrand(
        dis, wDis, H1, D1, H2, D2, N2, reaSource, imgSource)

    # Evaluate integral
    if isinstance(time, (np.floating, float)):
        # Lower bound of integration
        a = 1.0 / np.sqrt(4.0*alpha*time)
        h = 0.5 / (N2*H2) * quad_vec(f, a, np.inf)[0]
    else:
        # Lower bound of integration
        a = 1.0 / np.sqrt(4.0*alpha*time)
        # Upper bound of integration
        b = np.concatenate(([np.inf], a[:-1]))
        h = np.cumsum(np.stack(
            [0.5 / (N2*H2) * quad_vec(f, a_i, b_i)[0]
             for t, a_i, b_i in zip(time, a, b)],
            axis=-1), axis=-1)
    return h


def _finite_line_source_equivalent_boreholes_integrand(dis, wDis, H1, D1, H2, D2, N2, reaSource, imgSource):
    """
    Integrand of the finite line source solution.

    Parameters
    ----------
    dis : array
        Unique radial distances to evaluate the FLS solution.
    wDis : array
        Number of instances of each unique radial distances.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    N2 : float or array,
        Number of segments represented by the receiving heat sources.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
    imgSource : bool
        True if the image part of the FLS solution is to be included.

    Returns
    -------
    f : callable
        Integrand of the finite line source solution. Can be vector-valued.

    """
    if reaSource and imgSource:
        # Full (real + image) FLS solution
        p = np.array([1, -1, 1, -1, 1, -1, 1, -1])
        q = np.stack([D2 - D1 + H2,
                      D2 - D1,
                      D2 - D1 - H1,
                      D2 - D1 + H2 - H1,
                      D2 + D1 + H2,
                      D2 + D1,
                      D2 + D1 + H1,
                      D2 + D1 + H2 + H1],
                     axis=-1)
        f = lambda s: s**-2 * (np.exp(-dis**2*s**2) @ wDis).T * np.inner(p, erfint(q*s))
    elif reaSource:
        # Real FLS solution
        p = np.array([1, -1, 1, -1])
        q = np.stack([D2 - D1 + H2,
                      D2 - D1,
                      D2 - D1 - H1,
                      D2 - D1 + H2 - H1],
                     axis=-1)
        f = lambda s: s**-2 * (np.exp(-dis**2*s**2) @ wDis).T * np.inner(p, erfint(q*s))
    elif imgSource:
        # Image FLS solution
        p = np.array([1, -1, 1, -1])
        q = np.stack([D2 + D1 + H2,
                      D2 + D1,
                      D2 + D1 + H1,
                      D2 + D1 + H2 + H1],
                     axis=-1)
        f = lambda s: s**-2 * (np.exp(-dis**2*s**2) @ wDis).T * np.inner(p, erfint(q*s))
    else:
        # No heat source
        f = lambda s: np.zeros(np.broadcast_shapes(
            *[np.shape(arg) for arg in (H1, D1, H2, D2, N2)]))
    return f


def _finite_line_source_coefficients(
        borefield_j: Borefield,
        borefield_i: Borefield,
        reaSource: bool = True,
        imgSource: bool = True,
        outer: bool = True
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coefficients for the finite line source solutions of vertical boreholes.

    Parameters
    ----------
    borefield_j : Borehole or Borefield object
        Borehole or Borefield object of the boreholes extracting heat.
    borefield_i : Borehole or Borefield object
        Borehole or Borefield object object for which the FLS is evaluated.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
        Default is True.
    imgSource : bool, optional
        True if the image part of the FLS solution is to be included.
        Default is True.
    outer : bool, optional
        True if the finite line source is to be evaluated for all boreholes_j
        onto all boreholes_i to return a (nBoreholes_i, nBoreholes_j, nTime,)
        array. If false, the finite line source is evaluated pairwise between
        boreholes_j and boreholes_i. The numbers of boreholes should be the
        same (i.e. nBoreholes_j == nBoreholes_i) and a (nBoreholes, nTime,)
        array is returned.
        Default is True.

    Returns
    -------
    p : array
        Weights for the superposition of terms in the integrand of the finite
        line source solution.
    q : array
        Arguments of the integral of the error function in the integrand of the
        finite line source solution.

    """
    H_j = borefield_j.H
    D_j = borefield_j.D
    H_i = borefield_i.H
    D_i = borefield_i.D

    if reaSource and imgSource:
        # Full (real + image) FLS solution
        p = np.array([1, -1, 1, -1, 1, -1, 1, -1])
        if outer:
            q = np.stack(
                    [np.subtract.outer(D_i + H_i, D_j),
                     np.subtract.outer(D_i, D_j),
                     np.subtract.outer(D_i, D_j + H_j),
                     np.subtract.outer(D_i + H_i, D_j + H_j),
                     np.add.outer(D_i + H_i, D_j),
                     np.add.outer(D_i, D_j),
                     np.add.outer(D_i, D_j + H_j),
                     np.add.outer(D_i + H_i, D_j + H_j),
                        ],
                    axis=0)
        else:
            q = np.stack(
                    [D_i + H_i - D_j,
                     D_i - D_j,
                     D_i - (D_j + H_j),
                     D_i + H_i - (D_j + H_j),
                     D_i + H_i + D_j,
                     D_i + D_j,
                     D_i + D_j + H_j,
                     D_i + H_i + D_j + H_j,
                        ],
                    axis=0)
    elif reaSource:
        # Real FLS solution
        p = np.array([1, -1, 1, -1])
        if outer:
            q = np.stack(
                    [np.subtract.outer(D_i + H_i, D_j),
                     np.subtract.outer(D_i, D_j),
                     np.subtract.outer(D_i, D_j + H_j),
                     np.subtract.outer(D_i + H_i, D_j + H_j),
                        ],
                    axis=0)
        else:
            q = np.stack(
                    [D_i + H_i - D_j,
                     D_i - D_j,
                     D_i - (D_j + H_j),
                     D_i + H_i - (D_j + H_j),
                        ],
                    axis=0)
    elif imgSource:
        # Image FLS solution
        p = np.array([1, -1, 1, -1])
        if outer:
            q = np.stack(
                    [np.add.outer(D_i + H_i, D_j),
                     np.add.outer(D_i, D_j),
                     np.add.outer(D_i, D_j + H_j),
                     np.add.outer(D_i + H_i, D_j + H_j),
                        ],
                    axis=0)
        else:
            q = np.stack(
                    [D_i + H_i + D_j,
                     D_i + D_j,
                     D_i + D_j + H_j,
                     D_i + H_i + D_j + H_j,
                        ],
                    axis=0)
    else:
        # No heat source
        p = np.zeros(0)
        if outer:
            q = np.zeros((0, len(borefield_i), len(borefield_j)))
        else:
            q = np.zeros((0, len(borefield_i)))
    return p, q
