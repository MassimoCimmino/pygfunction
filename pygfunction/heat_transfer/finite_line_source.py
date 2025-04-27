# -*- coding: utf-8 -*-
from typing import Union, List

import numpy as np
import numpy.typing as npt

from .finite_line_source_inclined import finite_line_source_inclined
from .finite_line_source_vertical import finite_line_source_vertical
from ..borefield import Borefield
from ..boreholes import Borehole


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
    # Convert bore fields to Borefield objects
    if isinstance(borefield_j, list):
        borefield_j = Borefield.from_boreholes(borefield_j)
    if isinstance(borefield_i, list):
        borefield_i = Borefield.from_boreholes(borefield_i)

    # Select the analytical solution based on borehole inclination
    if np.any(borefield_j.is_tilted) or np.any(borefield_i.is_tilted):
        # Inclined boreholes
        h = finite_line_source_inclined(
            time, alpha, borefield_j, borefield_i, reaSource=reaSource,
            imgSource=imgSource, approximation=approximation, outer=outer, M=M,
            N=N)
    else:
        # Vertical boreholes
        h = finite_line_source_vertical(
            time, alpha, borefield_j, borefield_i,
            reaSource=reaSource, imgSource=imgSource,
            approximation=approximation, outer=outer, N=N)
    return h
