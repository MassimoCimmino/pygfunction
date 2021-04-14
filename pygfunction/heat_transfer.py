from __future__ import absolute_import, division, print_function

import time as tim
from functools import partial
from multiprocessing import Pool

import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import erf


def finite_line_source(
        time, alpha, borehole1, borehole2, reaSource=True, imgSource=True):
    """
    Evaluate the Finite Line Source (FLS) solution.

    This function uses a numerical quadrature to evaluate the one-integral form
    of the FLS solution, as proposed by Claesson and Javed
    [#ClaessonJaved2011]_ and extended to boreholes with different vertical
    positions by Cimmino and Bernier [#CimminoBernier2014]_. The FlS solution
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
    time : float
        Value of time (in seconds) for which the FLS solution is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    borehole1 : Borehole object
        Borehole object of the borehole extracting heat.
    borehole2 : Borehole object
        Borehole object for which the FLS is evaluated.
    reaSource : boolean, defaults to True
        True if the real part of the FLS solution is to be included.
    imgSource : boolean, defaults to True
        True if the image part of the FLS solution is to be included.

    Returns
    -------
    h : float
        Value of the FLS solution. The average (over the length) temperature
        drop on the wall of borehole2 due to heat extracted from borehole1 is:

        .. math:: \\Delta T_{b,2} = T_g - \\frac{Q_1}{2\\pi k_s H_2} h

    Examples
    --------
    >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
    >>> b2 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
    >>> h = gt.heat_transfer.finite_line_source(4*168*3600., 1.0e-6, b1, b2)
    h = 0.0110473635393

    References
    ----------
    .. [#ClaessonJaved2011] Claesson, J., & Javed, S. (2011). An analytical
       method to calculate borehole fluid temperatures for time-scales from
       minutes to decades. ASHRAE Transactions, 117(2), 279-288.
    .. [#CimminoBernier2014] Cimmino, M., & Bernier, M. (2014). A
       semi-analytical method to generate g-functions for geothermal bore
       fields. International Journal of Heat and Mass Transfer, 70, 641-650.

    """
    def _Ils(s, b1, b2, reaSource, imgSource):
        r = b1.distance(b2)
        func = 0.
        # Function to integrate
        if reaSource:
            # Real part of the FLS solution
            func += _erfint((b2.D - b1.D + b2.H)*s)
            func += -_erfint((b2.D - b1.D)*s)
            func += _erfint((b2.D - b1.D - b1.H)*s)
            func += -_erfint((b2.D - b1.D + b2.H - b1.H)*s)
        if imgSource:
            # Image part of the FLS solution
            func += _erfint((b2.D + b1.D + b2.H)*s)
            func += -_erfint((b2.D + b1.D)*s)
            func += _erfint((b2.D + b1.D + b1.H)*s)
            func += -_erfint((b2.D + b1.D + b2.H + b1.H)*s)
        return 0.5 / (b2.H*s**2) * func * np.exp(-r**2*s**2)

    def _erfint(x):
        # Integral of error function
        return x * erf(x) - 1.0/np.sqrt(np.pi) * (1.0-np.exp(-x**2))

    # Lower bound of integration
    a = 1.0 / np.sqrt(4.0*alpha*time)
    # Evaluate integral using Gauss-Kronrod
    h, err = quad(
        _Ils, a, np.inf, args=(borehole1, borehole2, reaSource, imgSource))
    return h


def finite_line_source_vectorized(
        time, alpha, dis, H1, D1, H2, D2, reaSource=True, imgSource=True):
    """

    """
    def _erfint(x):
        # Integral of error function
        return x * erf(x) - 1.0/np.sqrt(np.pi) * (1.0-np.exp(-x**2))

    if reaSource and imgSource:
        # Full (real + image) FLS solution
        func = lambda s: 0.5 / (H2*s**2) * np.exp(-dis**2*s**2) * (
            _erfint((D2 - D1 + H2)*s) - _erfint((D2 - D1)*s)
            + _erfint((D2 - D1 - H1)*s) - _erfint((D2 - D1 + H2 - H1)*s)
            + _erfint((D2 + D1 + H2)*s) - _erfint((D2 + D1)*s)
            + _erfint((D2 + D1 + H1)*s) - _erfint((D2 + D1 + H2 + H1)*s) )
    elif reaSource:
        # Real FLS solution
        func = lambda s: 0.5 / (H2*s**2) * np.exp(-dis**2*s**2) * (
            _erfint((D2 - D1 + H2)*s) - _erfint((D2 - D1)*s)
            + _erfint((D2 - D1 - H1)*s) - _erfint((D2 - D1 + H2 - H1)*s) )
    elif imgSource:
        # Image FLS solution
        func = lambda s: 0.5 / (H2*s**2) * np.exp(-dis**2*s**2) * (
            _erfint((D2 + D1 + H2)*s) - _erfint((D2 + D1)*s)
            + _erfint((D2 + D1 + H1)*s) - _erfint((D2 + D1 + H2 + H1)*s) )
    else:
        # No heat source
        func = lambda s: 0.

    # Lower bound of integration
    a = 1.0 / np.sqrt(4.0*alpha*time)
    # Evaluate integral using Gauss-Kronrod
    h, err = quad_vec(func, a, np.inf)
    return h

