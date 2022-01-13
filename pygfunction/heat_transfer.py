# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import erfc

from .boreholes import Borehole
from .utilities import erfint, exp1, _erf_coeffs


def finite_line_source(
        time, alpha, borehole1, borehole2, reaSource=True, imgSource=True,
        approximation=False, N=10):
    """
    Evaluate the Finite Line Source (FLS) solution.

    This function uses a numerical quadrature to evaluate the one-integral form
    of the FLS solution, as proposed by Claesson and Javed [#FLS-ClaJav2011]_
    and extended to boreholes with different vertical positions by Cimmino and
    Bernier [#FLS-CimBer2014]_. The FlS solution is given by:

        .. math::
            h_{1\\rightarrow2}(t) &= \\frac{1}{2H_2}
            \\int_{\\frac{1}{\\sqrt{4\\alpha t}}}^{\\infty}
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
                    h_{2\\rightarrow1}(t) = \\frac{H_2}{H_1}
                    h_{1\\rightarrow2}(t)

    Parameters
    ----------
    time : float or array, shape (K)
        Value of time (in seconds) for which the FLS solution is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    borehole1 : Borehole object or list of Borehole objects, length (N)
        Borehole object of the borehole extracting heat.
    borehole2 : Borehole object or list of Borehole objects, length (M)
        Borehole object for which the FLS is evaluated.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
        Default is True.
    imgSource : bool, optional
        True if the image part of the FLS solution is to be included.
        Default is True.
    approximation : bool, optional
        Set to true to use the approximation of the FLS solution of Cimmino
        (2021) [#FLS-Cimmin2021]_. This approximation does not require
        the numerical evaluation of any integral.
        Default is False.
    N : int, optional
        Number of terms in the approximation of the FLS solution. This
        parameter is unused if `approximation` is set to False.
        Default is 10. Maximum is 25.

    Returns
    -------
    h : float or array, shape (M, N, K), (M, N) or (K)
        Value of the FLS solution. The average (over the length) temperature
        drop on the wall of borehole2 due to heat extracted from borehole1 is:

        .. math:: \\Delta T_{b,2} = T_g - \\frac{Q_1}{2\\pi k_s H_2} h

    Notes
    -----
    The function returns a float if time is a float and borehole1 and borehole2
    are Borehole objects. If time is a float and any of borehole1 and borehole2
    are lists, the function returns an array, shape (M, N), If time is an array
    and borehole1 and borehole2 are Borehole objects, the function returns an
    array, shape (K).If time is an array and any of borehole1 and borehole2 are
    are lists, the function returns an array, shape (M, N, K).

    Examples
    --------
    >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
    >>> b2 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
    >>> h = gt.heat_transfer.finite_line_source(4*168*3600., 1.0e-6, b1, b2)
    h = 0.0110473635393
    >>> h = gt.heat_transfer.finite_line_source(
        4*168*3600., 1.0e-6, b1, b2, approximation=True, N=10)
    h = 0.0110474667731

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

    """
    if isinstance(borehole1, Borehole) and isinstance(borehole2, Borehole):
        # Unpack parameters
        dis = borehole1.distance(borehole2)
        H1, D1 = borehole1.H, borehole1.D
        H2, D2 = borehole2.H, borehole2.D
        # Integrand of the finite line source solution
        f = _finite_line_source_integrand(
            dis, H1, D1, H2, D2, reaSource, imgSource)

        # Evaluate integral
        if time == np.inf:
            h = _finite_line_source_steady_state(
                dis, H1, D1, H2, D2, reaSource, imgSource)
        elif isinstance(time, (np.floating, float)):
            if not approximation:
                # Lower bound of integration
                a = 1.0 / np.sqrt(4.0*alpha*time)
                h = 0.5 / H2 * quad(f, a, np.inf)[0]
            else:
                h = finite_line_source_approximation(
                    time, alpha, dis, H1, D1, H2, D2,
                    reaSource=reaSource, imgSource=imgSource, N=N)
        else:
            if not approximation:
                h = np.stack(
                    [0.5 / H2 * quad(f, 1.0 / np.sqrt(4.0*alpha*t), np.inf)[0] for t in time],
                    axis=-1)
            else:
                h = finite_line_source_approximation(
                    time, alpha, dis, H1, D1, H2, D2,
                    reaSource=reaSource, imgSource=imgSource, N=N)
    else:
        # Unpack parameters
        if isinstance(borehole1, Borehole): borehole1 = [borehole1]
        if isinstance(borehole2, Borehole): borehole2 = [borehole2]
        x1 = np.array([b.x for b in borehole1])
        y1 = np.array([b.y for b in borehole1])
        x2 = np.array([b.x for b in borehole2])
        y2 = np.array([b.y for b in borehole2])
        r_b = np.array([b.r_b for b in borehole1])
        dis = np.maximum(np.sqrt(np.add.outer(x2, -x1)**2 + np.add.outer(y2, -y1)**2), r_b)
        D1 = np.array([b.D for b in borehole1]).reshape(1, -1)
        H1 = np.array([b.H for b in borehole1]).reshape(1, -1)
        D2 = np.array([b.D for b in borehole2]).reshape(-1, 1)
        H2 = np.array([b.H for b in borehole2]).reshape(-1, 1)

        if time is np.inf:
            h = _finite_line_source_steady_state(
                dis, H1, D1, H2, D2, reaSource, imgSource)
        else:
            if not approximation:
                # Evaluate integral
                h = finite_line_source_vectorized(
                    time, alpha, dis, H1, D1, H2, D2,
                    reaSource=reaSource, imgSource=imgSource)
            else:
                h = finite_line_source_approximation(
                    time, alpha, dis, H1, D1, H2, D2,
                    reaSource=reaSource, imgSource=imgSource, N=N)
    return h


def finite_line_source_approximation(
        time, alpha, dis, H1, D1, H2, D2, reaSource=True, imgSource=True,
        N=10):
    """
    Evaluate the Finite Line Source (FLS) solution using the approximation
    of Cimmino (2021) [#FLSApprox-Cimmin2021]_.

    Parameters
    ----------
    time : float or array, shape (K)
        Value of time (in seconds) for which the FLS solution is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    dis : float or array
        Radial distances to evaluate the FLS solution.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
        Default is True.
    imgSource : bool
        True if the image part of the FLS solution is to be included.
        Default is True.
    N : int, optional
        Number of terms in the approximation of the FLS solution. This
        parameter is unused if `approximation` is set to False.
        Default is 10. Maximum is 25.

    Returns
    -------
    h : float
        Value of the FLS solution. The average (over the length) temperature
        drop on the wall of borehole2 due to heat extracted from borehole1 is:

        .. math:: \\Delta T_{b,2} = T_g - \\frac{Q_1}{2\\pi k_s H_2} h
    

    References
    ----------
    .. [#FLSApprox-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.

    """

    dis = np.divide.outer(dis, np.sqrt(4*alpha*time))
    H1 = np.divide.outer(H1, np.sqrt(4*alpha*time))
    D1 = np.divide.outer(D1, np.sqrt(4*alpha*time))
    H2 = np.divide.outer(H2, np.sqrt(4*alpha*time))
    D2 = np.divide.outer(D2, np.sqrt(4*alpha*time))
    if reaSource and imgSource:
        # Full (real + image) FLS solution
        p = np.array([1, -1, 1, -1, 1, -1, 1, -1])
        q = np.abs(
            np.stack([D2 - D1 + H2,
                      D2 - D1,
                      D2 - D1 - H1,
                      D2 - D1 + H2 - H1,
                      D2 + D1 + H2,
                      D2 + D1,
                      D2 + D1 + H1,
                      D2 + D1 + H2 + H1],
                     axis=-1))
    elif reaSource:
        # Real FLS solution
        p = np.array([1, -1, 1, -1])
        q = np.abs(
            np.stack([D2 - D1 + H2,
                      D2 - D1,
                      D2 - D1 - H1,
                      D2 - D1 + H2 - H1],
                     axis=-1))
    elif imgSource:
        # Image FLS solution
        p = np.array([1, -1, 1, -1])
        q = np.abs(
            np.stack([D2 + D1 + H2,
                      D2 + D1,
                      D2 + D1 + H1,
                      D2 + D1 + H2 + H1],
                     axis=-1))
    else:
        # No heat source
        p = np.zeros(1)
        q = np.zeros(1)
    # Coefficients of the approximation of the error function
    a, b = _erf_coeffs(N)

    dd = dis**2
    qq = q**2
    G1 = np.inner(
        p,
        q * np.inner(
            a,
            0.5*  exp1(np.expand_dims(dd, axis=(-1, -2)) + np.multiply.outer(qq, b))))
    x3 = np.sqrt(np.expand_dims(dd, axis=-1) + qq)
    G3 = np.inner(p, np.exp(-x3**2) / np.sqrt(np.pi) - x3 * erfc(x3))

    h = 0.5 / H2 * (G1 + G3)
    return h


def finite_line_source_vectorized(
        time, alpha, dis, H1, D1, H2, D2, reaSource=True, imgSource=True,
        approximation=False, N=10):
    """
    Evaluate the Finite Line Source (FLS) solution.

    This function uses a numerical quadrature to evaluate the one-integral form
    of the FLS solution, as proposed by Claesson and Javed
    [#FLSVec-ClaJav2011]_ and extended to boreholes with different vertical
    positions by Cimmino and Bernier [#FLSVec-CimBer2014]_. The FlS solution
    is given by:

        .. math::
            h_{1\\rightarrow2}(t) &= \\frac{1}{2H_2}
            \\int_{\\frac{1}{\\sqrt{4\\alpha t}}}^{\\infty}
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
                    h_{2\\rightarrow1}(t) = \\frac{H_2}{H_1}
                    h_{1\\rightarrow2}(t)

    Parameters
    ----------
    time : float or array, shape (K)
        Value of time (in seconds) for which the FLS solution is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    dis : float or array
        Radial distances to evaluate the FLS solution.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
        Default is True.
    imgSource : bool
        True if the image part of the FLS solution is to be included.
        Default is True.
    approximation : bool, optional
        Set to true to use the approximation of the FLS solution of Cimmino
        (2021) [#FLSVec-Cimmin2021]_. This approximation does not require
        the numerical evaluation of any integral.
        Default is False.
    N : int, optional
        Number of terms in the approximation of the FLS solution. This
        parameter is unused if `approximation` is set to False.
        Default is 10. Maximum is 25.

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
    .. [#FLSVec-ClaJav2011] Claesson, J., & Javed, S. (2011). An analytical
       method to calculate borehole fluid temperatures for time-scales from
       minutes to decades. ASHRAE Transactions, 117(2), 279-288.
    .. [#FLSVec-CimBer2014] Cimmino, M., & Bernier, M. (2014). A
       semi-analytical method to generate g-functions for geothermal bore
       fields. International Journal of Heat and Mass Transfer, 70, 641-650.
    .. [#FLSVec-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.

    """
    if not approximation:
        # Integrand of the finite line source solution
        f = _finite_line_source_integrand(
            dis, H1, D1, H2, D2, reaSource, imgSource)
    
        # Evaluate integral
        if isinstance(time, (np.floating, float)):
            # Lower bound of integration
            a = 1.0 / np.sqrt(4.0*alpha*time)
            h = 0.5 / H2 * quad_vec(f, a, np.inf)[0]
        else:
            h = np.stack(
                [0.5 / H2 * quad_vec(f, 1.0 / np.sqrt(4.0*alpha*t), np.inf)[0]
                 for t in time],
                axis=-1)
    else:
        h = finite_line_source_approximation(
            time, alpha, dis, H1, D1, H2, D2, reaSource=reaSource,
            imgSource=imgSource, N=N)
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
        h = np.stack(
            [0.5 / (N2*H2) * quad_vec(f, 1.0 / np.sqrt(4.0*alpha*t), np.inf)[0]
             for t in time],
            axis=-1)
    return h


def _finite_line_source_integrand(dis, H1, D1, H2, D2, reaSource, imgSource):
    """
    Integrand of the finite line source solution.

    Parameters
    ----------
    dis : float or array
        Radial distances to evaluate the FLS solution.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
    imgSource : bool
        True if the image part of the FLS solution is to be included.

    Returns
    -------
    f : callable
        Integrand of the finite line source solution. Can be vector-valued.

    Notes
    -----
    All arrays (dis, H1, D1, H2, D2) must follow numpy array broadcasting
    rules.

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
        f = lambda s: s**-2 * np.exp(-dis**2*s**2) * np.inner(p, erfint(q*s))
    elif reaSource:
        # Real FLS solution
        p = np.array([1, -1, 1, -1])
        q = np.stack([D2 - D1 + H2,
                      D2 - D1,
                      D2 - D1 - H1,
                      D2 - D1 + H2 - H1],
                     axis=-1)
        f = lambda s: s**-2 * np.exp(-dis**2*s**2) * np.inner(p, erfint(q*s))
    elif imgSource:
        # Image FLS solution
        p = np.array([1, -1, 1, -1])
        q = np.stack([D2 + D1 + H2,
                      D2 + D1,
                      D2 + D1 + H1,
                      D2 + D1 + H2 + H1],
                     axis=-1)
        f = lambda s: s**-2 * np.exp(-dis**2*s**2) * np.inner(p, erfint(q*s))
    else:
        # No heat source
        f = lambda s: 0.
    return f


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

    Notes
    -----
    All arrays (dis, H1, D1, H2, D2) must follow numpy array broadcasting
    rules.

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
        f = lambda s: 0.
    return f


def _finite_line_source_steady_state(dis, H1, D1, H2, D2, reaSource, imgSource):
    """
    Integrand of the finite line source solution.

    Parameters
    ----------
    dis : float or array
        Radial distances to evaluate the FLS solution.
    H1 : float or array
        Lengths of the emitting heat sources.
    D1 : float or array
        Buried depths of the emitting heat sources.
    H2 : float or array
        Lengths of the receiving heat sources.
    D2 : float or array
        Buried depths of the receiving heat sources.
    reaSource : bool
        True if the real part of the FLS solution is to be included.
    imgSource : bool
        True if the image part of the FLS solution is to be included.

    Returns
    -------
    h : Steady-state finite line source solution.

    Notes
    -----
    All arrays (dis, H1, D1, H2, D2) must follow numpy array broadcasting
    rules.

    """
    # Steady-state solution
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
        dis = np.expand_dims(dis, axis=-1)
        h = 0.5 / H2 * np.inner(p, q * np.log((q + np.sqrt(q**2 + dis**2)) / dis) - np.sqrt(q**2 + dis**2))
    elif reaSource:
        # Real FLS solution
        p = np.array([1, -1, 1, -1])
        q = np.stack([D2 - D1 + H2,
                      D2 - D1,
                      D2 - D1 - H1,
                      D2 - D1 + H2 - H1,],
                      axis=-1)
        dis = np.expand_dims(dis, axis=-1)
        h = 0.5 / H2 * np.inner(p, q * np.log((q + np.sqrt(q**2 + dis**2)) / dis) - np.sqrt(q**2 + dis**2))
    elif imgSource:
        # Image FLS solution
        p = np.array([1, -1, 1, -1])
        q = np.stack([D2 + D1 + H2,
                      D2 + D1,
                      D2 + D1 + H1,
                      D2 + D1 + H2 + H1],
                      axis=-1)
        dis = np.expand_dims(dis, axis=-1)
        h = 0.5 / H2 * np.inner(p, q * np.log((q + np.sqrt(q**2 + dis**2)) / dis) - np.sqrt(q**2 + dis**2))
    else:
        # No heat source
        h = 0.
    return h
