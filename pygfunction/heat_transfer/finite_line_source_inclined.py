# -*- coding: utf-8 -*-
from collections.abc import Callable
from typing import Union, List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad_vec
from scipy.special import erf, roots_legendre

from ..borefield import Borefield
from ..boreholes import Borehole
from ..utilities import exp1, _erf_coeffs


def finite_line_source_inclined(
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
    Evaluate the Finite Line Source (FLS) solution for inclined boreholes.

    This function uses a numerical quadrature to evaluate the one-integral form
    of the FLS solution. For inclined boreholes, the FLS solution was proposed
    by Lazzarotto [#FLSi-Lazzar2016]_ and Lazzarotto and Björk
    [#FLSi-LazBjo2016]_. The FLS solution is given by:

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
        (2021) [#FLSi-Cimmin2021]_. This approximation does not require
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
    .. [#FLSi-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.
    .. [#FLSi-Lazzar2016] Lazzarotto, A. (2016). A methodology for the
       calculation of response functions for geothermal fields with
       arbitrarily oriented boreholes – Part 1, Renewable Energy, 86,
       1380-1393.
    .. [#FLSi-LazBjo2016] Lazzarotto, A., & Björk, F. (2016). A methodology for
       the calculation of response functions for geothermal fields with
       arbitrarily oriented boreholes – Part 2, Renewable Energy, 86,
       1353-1361.

    """
    # Check if both bore fields are Borehole objects
    single_pair =  (
        isinstance(borefield_j, Borehole)
        and isinstance(borefield_i, Borehole)
        )
    # Convert bore fields to Borefield objects
    if isinstance(borefield_j, Borehole) or isinstance(borefield_j, list):
        borefield_j = Borefield.from_boreholes(borefield_j)
    if isinstance(borefield_i, Borehole) or isinstance(borefield_i, list):
        borefield_i = Borefield.from_boreholes(borefield_i)

    # Convert time to array if it is a list
    if isinstance(time, list):
        time = np.array(time)

    # Evaluate the finite line source solution
    if time is np.inf:
        # Steady-state finite line source solution
        h = _finite_line_source_inclined_steady_state(
            borefield_j, borefield_i, reaSource=reaSource,
            imgSource=imgSource, outer=outer, M=M)
    elif approximation:
        # Approximation of the finite line source solution
        h = _finite_line_source_inclined_approximation(
            time, alpha, borefield_j, borefield_i, reaSource=reaSource,
            imgSource=imgSource, outer=outer, M=M, N=N)
    else:
        # Integrand of the finite line source solution
        f = _finite_line_source_inclined_integrand(
            borefield_j, borefield_i, reaSource=reaSource,
            imgSource=imgSource, outer=outer, M=M)
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
        if isinstance(time, float):
            h = float(h)
    return h


def _finite_line_source_inclined_approximation(
        time: npt.ArrayLike,
        alpha: float,
        borefield_j: Borefield,
        borefield_i: Borefield,
        reaSource: bool = True,
        imgSource: bool = True,
        outer: bool = True,
        M: int = 11,
        N: int = 10
        ) -> np.ndarray:
    """
    Evaluate the inclined Finite Line Source (FLS) solution using the
    approximation method of Cimmino (2021) [#IncFLSApprox-Cimmin2021]_.

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

    References
    ----------
    .. [#IncFLSApprox-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.

    """
    # Evaluate coefficients of the FLS solution
    p, q, k = _finite_line_source_inclined_coefficients(
            borefield_j, borefield_i, reaSource=reaSource, imgSource=imgSource,
            outer=outer)

    # Coefficients of the approximation of the error function
    a, b = _erf_coeffs(N)

    # Roots for Gauss-Legendre quadrature
    x, w = roots_legendre(M)
    u = 0.5 * x + 0.5
    w = w / 2

    # Extract lengths and reshape if outer == True
    H_j = borefield_j.H
    H_i = borefield_i.H
    if outer:
        H_i = H_i[..., np.newaxis]
        H_ratio = np.divide.outer(borefield_j.H, borefield_i.H).T
    else:
        H_ratio = borefield_j.H / borefield_i.H

    # Additional coefficients for the approximation of the FLS solution
    s = 1. / (4 * alpha * time)
    d = [
        k[2] + np.multiply.outer(u, H_j * k[0]),
        k[2] - H_i + np.multiply.outer(u, H_j * k[0]),
        ]
    c = np.maximum(
        (q - d[0]**2 - k[1]**2)
        + (k[1] + np.multiply.outer(u, np.ones_like(k[1]) * H_j))**2,
        borefield_j.r_b**2)

    # Approximation of the FLS solution
    h = 0.25 * H_ratio * ((
        np.sign(
            np.multiply.outer(d[0], np.ones_like(s))
            ) * exp1(
                np.multiply.outer(c + np.multiply.outer(b, d[0]**2), s))
        - np.sign(
            np.multiply.outer(d[1], np.ones_like(s))
            ) * exp1(np.multiply.outer(c + np.multiply.outer(b, d[1]**2), s))
        ).T @ a @ w @ p).T
    return h


def _finite_line_source_inclined_integrand(
        borefield_j: Borefield,
        borefield_i: Borefield,
        reaSource: bool = True,
        imgSource: bool = True,
        outer: bool = True,
        M: int = 11
        ) -> Callable:
    """
    Integrand of the inclined Finite Line Source (FLS) solution.

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
    M : int, optional
        Number of Gauss-Legendre sample points for the quadrature over
        :math:`u`. This is only used for inclined boreholes.
        Default is 11.

    Returns
    -------
    f : callable
        Integrand of the finite line source solution. Can be vector-valued.

    """
    # Evaluate coefficients of the FLS solution
    p, q, k = _finite_line_source_inclined_coefficients(
            borefield_j, borefield_i, reaSource=reaSource, imgSource=imgSource,
            outer=outer)

    # Roots for Gauss-Legendre quadrature
    x, w = roots_legendre(M)
    u = 0.5 * x + 0.5
    w = w / 2

    # Extract lengths and reshape if outer == True
    H_j = borefield_j.H
    H_i = borefield_i.H
    if outer:
        H_i = H_i[..., np.newaxis]

    # Integrand of the inclined finite line source solution
    f = lambda s: \
        H_j / s * (((
            np.exp(
                s**2 * (
                -q
                + np.multiply.outer(u**2, H_j**2 * (k[0]**2 - 1))
                + 2 * np.multiply.outer(u, H_j * (k[0] * k[2] - k[1]))
                + k[2]**2
                )
                ) \
            * (
                erf(
                    s * (
                        np.multiply.outer(u, H_j * k[0])
                        + k[2]
                        )
                    )
                - erf(
                    s * (
                        np.multiply.outer(u, H_j * k[0])
                        + k[2]
                        - H_i
                        )
                    )
                )
            ).T @ w) @ p).T
    return f


def _finite_line_source_inclined_steady_state(
        borefield_j: Borefield,
        borefield_i: Borefield,
        reaSource: bool = True,
        imgSource: bool = True,
        outer: bool = True,
        M: int = 11
        ) -> np.ndarray:
    """
    Steady-state inclined Finite Line Source (FLS) solution.

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
    M : int, optional
        Number of Gauss-Legendre sample points for the quadrature over
        :math:`u`. This is only used for inclined boreholes.
        Default is 11.

    Returns
    -------
    h : float or array, shape (nBoreholes_i, nBoreholes_j) or (nBoreholes,)
        Value of the steady-state FLS solution. The average (over the length)
        temperature drop on the wall of borehole_i due to heat extracted from
        borehole_j is:

        .. math:: \\Delta T_{b,i} = T_g - \\frac{Q_j}{2\\pi k_s H_j} h

    """
    # Evaluate coefficients of the FLS solution
    p, q, k = _finite_line_source_inclined_coefficients(
            borefield_j, borefield_i, reaSource=reaSource, imgSource=imgSource,
            outer=outer)

    # Roots for Gauss-Legendre quadrature
    x, w = roots_legendre(M)
    u = 0.5 * x + 0.5
    w = w / 2

    # Extract lengths and reshape if outer == True
    H_j = borefield_j.H
    H_i = borefield_i.H
    if outer:
        H_i = H_i[..., np.newaxis]
        H_ratio = np.divide.outer(borefield_j.H, borefield_i.H).T
    else:
        H_ratio = borefield_j.H / borefield_i.H

    # Steady-state inclined finite line source solution
    h = 0.5 * H_ratio * (
        np.log(
            (
                2 * H_i * np.sqrt(
                    q
                    + np.multiply.outer(u**2, np.ones_like(k[0]) * H_j**2)
                    + H_i**2
                    + 2 * np.multiply.outer(u, H_j * k[1])
                    - 2 * H_i * k[2]
                    - 2 * H_i * np.multiply.outer(u, H_j * k[0])
                    )
                + 2 * H_i**2
                - 2 * H_i * k[2]
                - 2 * H_i * np.multiply.outer(u, H_j * k[0])
            ) / (
                2 * H_i * np.sqrt(
                    q
                    + np.multiply.outer(u**2, np.ones_like(k[0]) * H_j**2)
                    + 2 * np.multiply.outer(u, H_j * k[1])
                    )
                - 2 * H_i * k[2]
                - 2 * H_i * np.multiply.outer(u, H_j * k[0])
                )
                ).T @ w @ p).T
    return h


def _finite_line_source_inclined_coefficients(
        borefield_j: Borefield,
        borefield_i: Borefield,
        reaSource: bool = True,
        imgSource: bool = True,
        outer: bool = True
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coefficients for the finite line source solutions of inclined boreholes.

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
        Squared distance between top edges of line sources.
    k : array
        Terms used in arguments of exponential and error functions in the
        integrand of the inclined finite line source solution.

    """
    # Sines and cosines of tilt (b: beta) and orientation (t: theta)
    sb_j = np.sin(borefield_j.tilt)
    sb_i = np.sin(borefield_i.tilt)
    cb_j = np.cos(borefield_j.tilt)
    cb_i = np.cos(borefield_i.tilt)
    st_j = np.sin(borefield_j.orientation)
    st_i = np.sin(borefield_i.orientation)
    ct_j = np.cos(borefield_j.orientation)
    ct_i = np.cos(borefield_i.orientation)
    # Horizontal distances between boreholes
    dis_ij = borefield_j.distance(borefield_i, outer=outer)
    if outer:
        dx = np.subtract.outer(borefield_j.x, borefield_i.x).T
        dy = np.subtract.outer(borefield_j.y, borefield_i.y).T
        ct_ij = np.cos(
            np.subtract.outer(
                borefield_j.orientation,
                borefield_i.orientation)
            ).T
    else:
        dx = borefield_j.x - borefield_i.x
        dy = borefield_j.y - borefield_i.y
        ct_ij = np.cos(borefield_j.orientation - borefield_i.orientation)
        
    if reaSource and imgSource:
        # Full (real + image) FLS solution
        p = np.array([1, -1])
        if outer:
            dzRea = np.subtract.outer(borefield_j.D, borefield_i.D).T
            dzImg = np.add.outer(borefield_i.D, borefield_j.D)
            k = [
                np.stack(
                    [np.multiply.outer(sb_i, sb_j) * ct_ij + np.multiply.outer(cb_i, cb_j),
                     np.multiply.outer(sb_i, sb_j) * ct_ij - np.multiply.outer(cb_i, cb_j)],
                    axis=0),
                np.stack(
                    [sb_j * (ct_j * dx + st_j * dy) + cb_j * dzRea,
                     sb_j * (ct_j * dx + st_j * dy) + cb_j * dzImg],
                    axis=0),
                np.stack(
                    [(sb_i * (ct_i * dx.T + st_i * dy.T) + cb_i * dzRea.T).T,
                     (sb_i * (ct_i * dx.T + st_i * dy.T) - cb_i * dzImg.T).T],
                    axis=0),
                ]
        else:
            dzRea = borefield_j.D - borefield_i.D
            dzImg = borefield_j.D + borefield_i.D
            k = [
                np.stack(
                    [sb_j * sb_i * ct_ij + cb_j * cb_i,
                     sb_j * sb_i * ct_ij - cb_j * cb_i],
                    axis=0),
                np.stack(
                    [sb_j * (ct_j * dx + st_j * dy) + cb_j * dzRea,
                     sb_j * (ct_j * dx + st_j * dy) + cb_j * dzImg],
                    axis=0),
                np.stack(
                    [sb_i * (ct_i * dx + st_i * dy) + cb_i * dzRea,
                     sb_i * (ct_i * dx + st_i * dy) - cb_i * dzImg],
                    axis=0),
                ]
        q = dis_ij**2 + np.stack(
            [dzRea, dzImg],
            axis=0)**2
    elif reaSource:
        # Real FLS solution
        p = np.array([1])
        if outer:
            dzRea = np.subtract.outer(borefield_j.D, borefield_i.D).T
            k = [
                (np.multiply.outer(sb_i, sb_j) * ct_ij + np.multiply.outer(cb_i, cb_j))[np.newaxis, ...],
                (sb_j * (ct_j * dx + st_j * dy) + cb_j * dzRea)[np.newaxis, ...],
                (sb_i * (ct_i * dx.T + st_i * dy.T) + cb_i * dzRea.T).T[np.newaxis, ...],
                ]
        else:
            dzRea = borefield_j.D - borefield_i.D
            k = [
                (sb_j * sb_i * ct_ij + cb_j * cb_i)[np.newaxis, ...],
                (sb_j * (ct_j * dx + st_j * dy) + cb_j * dzRea)[np.newaxis, ...],
                (sb_i * (ct_i * dx + st_i * dy) + cb_i * dzRea)[np.newaxis, ...],
                ]
        q = (dis_ij**2 + dzRea**2)[np.newaxis, ...]
    elif imgSource:
        # Image FLS solution
        p = np.array([-1])
        if outer:
            dzImg = np.add.outer(borefield_i.D, borefield_j.D)
            k = [
                (np.multiply.outer(sb_i, sb_j) * ct_ij - np.multiply.outer(cb_i, cb_j))[np.newaxis, ...],
                (sb_j * (ct_j * dx + st_j * dy) + cb_j * dzImg)[np.newaxis, ...],
                (sb_i * (ct_i * dx.T + st_i * dy.T) - cb_i * dzImg.T).T[np.newaxis, ...],
                ]
        else:
            dzImg = borefield_j.D + borefield_i.D
            k = [
                (sb_j * sb_i * ct_ij - cb_j * cb_i)[np.newaxis, ...],
                (sb_j * (ct_j * dx + st_j * dy) + cb_j * dzImg)[np.newaxis, ...],
                (sb_i * (ct_i * dx + st_i * dy) - cb_i * dzImg)[np.newaxis, ...],
                ]
        q = (dis_ij**2 + dzImg**2)[np.newaxis, ...]
    else:
        # No heat source
        p = np.zeros(0)
        if outer:
            q = np.zeros((0, len(borefield_i), len(borefield_j)))
        else:
            q = np.zeros((0, len(borefield_i)))
        k = [q, q, q]
    return p, q, k
