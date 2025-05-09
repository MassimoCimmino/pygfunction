# -*- coding: utf-8 -*-
from collections.abc import Callable
from typing import Union, List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad_vec
from scipy.special import erfc

from ..borefield import Borefield
from ..boreholes import Borehole
from ..utilities import erfint, exp1, _erf_coeffs


def finite_line_source_vertical(
        time: npt.ArrayLike,
        alpha: float,
        borefield_j: Union[Borehole, Borefield, List[Borehole]],
        borefield_i: Union[Borehole, Borefield, List[Borehole]],
        distances: Union[None, npt.ArrayLike] = None,
        weights: Union[None, npt.ArrayLike] = None,
        outer: bool = True,
        reaSource: bool = True,
        imgSource: bool = True,
        approximation: bool = False,
        N: int = 10
        ) -> np.ndarray:
    """
    Evaluate the Finite Line Source (FLS) solution for vertical boreholes.

    This function uses a numerical quadrature to evaluate the one-integral form
    of the FLS solution. For vertical boreholes, the FLS solution was proposed
    by Claesson and Javed [#FLSv-ClaJav2011]_ and extended to boreholes with
    different vertical positions by Cimmino and Bernier [#FLSv-CimBer2014]_.
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

        .. Note::
            The reciprocal thermal response factor
            :math:`h_{ji}(t)` can be conveniently calculated by:

                .. math::
                    h_{ji}(t) = \\frac{H_i}{H_j}
                    h_{ij}(t)

    Parameters
    ----------
    time : float or (nTime,) array
        Value of time (in seconds) for which the FLS solution is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    borefield_j : Borehole or Borefield object
        Borehole or Borefield object of the boreholes extracting heat.
    borefield_i : Borehole or Borefield object
        Borehole or Borefield object object for which the FLS is evaluated.
    distances : float or (nDistances,) array, optional
        If None, distances are evaluated from distances between boreholes in
        borefield_j and borefield_i. If not None, distances between boreholes
        in boreholes_j and boreholes_i are overwritten and an array of shape
        (nBoreholes_i, nBoreholes_j, nDistances, nTime,) (if outer == True)
        or (nBoreholes, nDistances, nTime,) (if outer == False) is returned.
        Default is None.
    weights : (nDistances,) or (nSums, nDistances,) array, optional
        If None, the FLS solution is unmodified. If not None, the FLS solution
        at each distance is multiplied by the corresponding weight and summed
        over the axis. An array of shape
        (nBoreholes_i, nBoreholes_j, nSums, nTime,) (if outer == True)
        or (nBoreholes, nSums, nTime,) (if outer == False) is returned. Only
        applies if distances is not None.
        Default is None.
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
        (2021) [#FLSv-Cimmin2021]_. This approximation does not require
        the numerical evaluation of any integral.
        Default is False.
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
    .. [#FLSv-ClaJav2011] Claesson, J., & Javed, S. (2011). An analytical
       method to calculate borehole fluid temperatures for time-scales from
       minutes to decades. ASHRAE Transactions, 117(2), 279-288.
    .. [#FLSv-CimBer2014] Cimmino, M., & Bernier, M. (2014). A
       semi-analytical method to generate g-functions for geothermal bore
       fields. International Journal of Heat and Mass Transfer, 70, 641-650.
    .. [#FLSv-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.

    """
    # Check if both bore fields are Borehole objects
    single_pair =  (
        isinstance(borefield_j, Borehole)
        and isinstance(borefield_i, Borehole)
        )
    # Convert bore fields to Borefield objects
    if isinstance(borefield_j, (Borehole, list)):
        borefield_j = Borefield.from_boreholes(borefield_j)
    if isinstance(borefield_i, (Borehole, list)):
        borefield_i = Borefield.from_boreholes(borefield_i)
    # Convert time to array if it is a list
    if isinstance(time, list):
        time = np.asarray(time)
    # Convert distances to array if it is a list
    if isinstance(distances, list):
        distances = np.asarray(distances)
    # Convert weights to array if it is a list
    if isinstance(weights, list):
        weights = np.asarray(weights)

    # Evaluate the finite line source solution
    if time is np.inf:
        # Steady-state finite line source solution
        h = _finite_line_source_vertical_steady_state(
            borefield_j, borefield_i, distances=distances, weights=weights,
            outer=outer, reaSource=reaSource, imgSource=imgSource)
    elif approximation:
        # Approximation of the finite line source solution
        h = _finite_line_source_vertical_approximation(
            time, alpha, borefield_j, borefield_i, distances=distances,
            weights=weights, outer=outer, reaSource=reaSource,
            imgSource=imgSource, N=N)
    else:
        # Integrand of the finite line source solution
        f = _finite_line_source_vertical_integrand(
            borefield_j, borefield_i, distances=distances, weights=weights,
            outer=outer, reaSource=reaSource, imgSource=imgSource)
        # Evaluate integral
        if isinstance(time, (np.floating, float)):
            # Lower bound of integration
            a = 1. / np.sqrt(4 * alpha * time)
            h = (0.5 / borefield_i.H * quad_vec(f, a, np.inf, epsabs=1e-4, epsrel=1e-6)[0].T).T
        else:
            # Lower bound of integration
            a = 1.0 / np.sqrt(4.0 * alpha * time)
            # Upper bound of integration
            b = np.concatenate(([np.inf], a[:-1]))
            h = np.cumsum(np.stack(
                [(0.5 / borefield_i.H * quad_vec(f, a_i, b_i, epsabs=1e-4, epsrel=1e-6)[0].T).T
                 for t, a_i, b_i in zip(time, a, b)],
                axis=-1), axis=-1)

    # Return a 1d array if only Borehole objects were provided
    if single_pair:
        if outer:
            h = h[0, 0, ...]
        else:
            h = h[0, ...]
        # Return a float if time is also a float
        if isinstance(time, (np.floating, float)):
            h = float(h)
    return h


def _finite_line_source_vertical_approximation(
        time: Union[float, np.ndarray],
        alpha: float,
        borefield_j: Borefield,
        borefield_i: Borefield,
        distances: Union[None, np.ndarray] = None,
        weights: Union[None, np.ndarray] = None,
        reaSource: bool = True,
        imgSource: bool = True,
        outer: bool = True,
        N: int = 10
        ) -> np.ndarray:
    """
    Evaluate the Finite Line Source (FLS) solution using the approximation
    of Cimmino (2021) [#FLSApprox-Cimmin2021]_.

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
    distances : float or (nDistances,) array, optional
        If None, distances are evaluated from distances between boreholes in
        borefield_j and borefield_i. If not None, distances between boreholes
        in boreholes_j and boreholes_i are overwritten and an array of shape
        (nBoreholes_i, nBoreholes_j, nDistances, nTime,) (if outer == True)
        or (nBoreholes, nDistances, nTime,) (if outer == False) is returned.
        Default is None.
    weights : (nDistances,) or (nSums, nDistances,) array, optional
        If None, the FLS solution is unmodified. If not None, the FLS solution
        at each distance is multiplied by the corresponding weight and summed
        over the axis. An array of shape
        (nBoreholes_i, nBoreholes_j, nSums, nTime,) (if outer == True)
        or (nBoreholes, nSums, nTime,) (if outer == False) is returned. Only
        applies if distances is not None.
        Default is None.
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

    References
    ----------
    .. [#FLSApprox-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.

    """
    # Evaluate coefficients of the FLS solution
    p, q = _finite_line_source_vertical_coefficients(
            borefield_j, borefield_i, reaSource=reaSource, imgSource=imgSource,
            outer=outer)
    # The approximation of the error function is only valid for positive
    # arguments and f (= x * erf(x)) is an even function
    q = np.abs(q)
    q2 = q**2

    # Coefficients of the approximation of the error function
    a, b = _erf_coeffs(N)

    # Distances
    if distances is None:
        dis = borefield_j.distance(borefield_i, outer=outer)
        dis2 = dis**2
        sqrt_dis2_plus_q2 = np.sqrt(dis2 + q2)
        dis2_plus_b_q2 = dis2 + np.multiply.outer(b, q2)
    else:
        dis = distances
        dis2 = dis**2
        sqrt_dis2_plus_q2 = np.sqrt(np.add.outer(q2, dis2))
        dis2_plus_b_q2 = np.add.outer(
            np.multiply.outer(b, q2),
            dis2)

    # Temporal terms    
    four_alpha_time = 4 * alpha * time
    sqrt_four_alpha_time = np.sqrt(four_alpha_time)

    # Term G1 of Cimmino (2021)
    G1 = 0.5 * (q.T * (
            exp1(
                np.divide.outer(
                    dis2_plus_b_q2,
                    four_alpha_time
                    )
                ).T @ a) @ p).T / sqrt_four_alpha_time

    # Term G3 of Cimmino (2021)
    x3 = np.divide.outer(sqrt_dis2_plus_q2, sqrt_four_alpha_time)
    G3 = ((np.exp(-x3**2) / np.sqrt(np.pi) - x3 * erfc(x3)).T @ p).T

    # Approximation of the FLS solution
    h = 0.5 * ((G1 + G3).T / borefield_i.H).T * sqrt_four_alpha_time
    if distances is not None and weights is not None:
        if isinstance(time, (np.floating, float)):
            axis = -1
        else:
            axis = -2
        h = np.tensordot(h, weights, axes=(axis, 0))
    return h


def _finite_line_source_vertical_coefficients(
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


def _finite_line_source_vertical_integrand(
        borefield_j: Borefield,
        borefield_i: Borefield,
        distances: Union[None, np.ndarray] = None,
        weights: Union[None, np.ndarray] = None,
        reaSource: bool = True,
        imgSource: bool = True,
        outer: bool = True
        ) -> Callable:
    """
    Integrand of the finite line source solution.

    Parameters
    ----------
    borefield_j : Borehole or Borefield object
        Borehole or Borefield object of the boreholes extracting heat.
    borefield_i : Borehole or Borefield object
        Borehole or Borefield object object for which the FLS is evaluated.
    distances : float or (nDistances,) array, optional
        If None, distances are evaluated from distances between boreholes in
        borefield_j and borefield_i. If not None, distances between boreholes
        in boreholes_j and boreholes_i are overwritten and an array of shape
        (nBoreholes_i, nBoreholes_j, nDistances, nTime,) (if outer == True)
        or (nBoreholes, nDistances, nTime,) (if outer == False) is returned.
        Default is None.
    weights : (nDistances,) or (nSums, nDistances,) array, optional
        If None, the FLS solution is unmodified. If not None, the FLS solution
        at each distance is multiplied by the corresponding weight and summed
        over the axis. An array of shape
        (nBoreholes_i, nBoreholes_j, nSums, nTime,) (if outer == True)
        or (nBoreholes, nSums, nTime,) (if outer == False) is returned. Only
        applies if distances is not None.
        Default is None.
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
    f : callable
        Integrand of the finite line source solution. Can be vector-valued.

    """
    # Evaluate coefficients of the FLS solution
    p, q = _finite_line_source_vertical_coefficients(
            borefield_j, borefield_i, reaSource=reaSource, imgSource=imgSource,
            outer=outer)
    # Integrand of the finite line source solution
    if distances is None:
        dis = borefield_j.distance(borefield_i, outer=outer)
        f = lambda s: s**-2 * np.exp(-dis**2 * s**2) * (erfint(q * s).T @ p).T
    else:
        dis = distances
        if weights is None:
            f = lambda s: s**-2 * np.multiply.outer(
                (erfint(q * s).T @ p).T, np.exp(-dis**2 * s**2)
                )
        else:
            w = weights
            f = lambda s: s**-2 * np.multiply.outer(
                (erfint(q * s).T @ p).T, w @ np.exp(-dis**2 * s**2)
                )
    return f


def _finite_line_source_vertical_steady_state(
        borefield_j: Borefield,
        borefield_i: Borefield,
        distances: Union[None, float, np.ndarray] = None,
        weights: Union[None, np.ndarray] = None,
        reaSource: bool = True,
        imgSource: bool = True,
        outer: bool = True
        ) -> np.ndarray:
    """
    Steady-state finite line source solution.

    Parameters
    ----------
    borefield_j : Borehole or Borefield object
        Borehole or Borefield object of the boreholes extracting heat.
    borefield_i : Borehole or Borefield object
        Borehole or Borefield object object for which the FLS is evaluated.
    distances : float or (nDistances,) array, optional
        If None, distances are evaluated from distances between boreholes in
        borefield_j and borefield_i. If not None, distances between boreholes
        in boreholes_j and boreholes_i are overwritten and an array of shape
        (nBoreholes_i, nBoreholes_j, nDistances, nTime,) (if outer == True)
        or (nBoreholes, nDistances, nTime,) (if outer == False) is returned.
        Default is None.
    weights : (nDistances,) or (nSums, nDistances,) array, optional
        If None, the FLS solution is unmodified. If not None, the FLS solution
        at each distance is multiplied by the corresponding weight and summed
        over the axis. An array of shape
        (nBoreholes_i, nBoreholes_j, nSums, nTime,) (if outer == True)
        or (nBoreholes, nSums, nTime,) (if outer == False) is returned. Only
        applies if distances is not None.
        Default is None.
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
    h : float or array, shape (nBoreholes_i, nBoreholes_j) or (nBoreholes,)
        Value of the steady-state FLS solution. The average (over the length)
        temperature drop on the wall of borehole_i due to heat extracted from
        borehole_j is:

        .. math:: \\Delta T_{b,i} = T_g - \\frac{Q_j}{2\\pi k_s H_j} h

    """
    # Evaluate coefficients of the FLS solution
    p, q = _finite_line_source_vertical_coefficients(
            borefield_j, borefield_i, reaSource=reaSource, imgSource=imgSource,
            outer=outer)

    # Extract lengths and reshape if outer == True
    H_i = borefield_i.H

    # Steady-state finite line source solution
    if distances is None:
        dis = borefield_j.distance(borefield_i, outer=outer)
        q2_plus_dis2 = np.sqrt(q**2 + dis**2)
        log_q_q2_plus_dis2 = np.log(q + q2_plus_dis2)
    else:
        dis = distances
        q2_plus_dis2 = np.sqrt(np.add.outer(q**2, dis**2))
        if isinstance(dis, np.ndarray):
            q = q[..., np.newaxis]
        log_q_q2_plus_dis2 = np.log(q + q2_plus_dis2)
        if weights is not None:
            q2_plus_dis2 = q2_plus_dis2 @ weights.T
            log_q_q2_plus_dis2 = log_q_q2_plus_dis2 @ weights.T
    h = 0.5 * (((q * log_q_q2_plus_dis2 - q2_plus_dis2).T @ p) / H_i).T
    return h
