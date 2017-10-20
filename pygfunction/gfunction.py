from __future__ import division, print_function, absolute_import

from functools import partial
import numpy as np
from multiprocessing import Pool
from scipy.interpolate import interp1d as interp1d
from scipy.constants import pi
import time as tim

from .boreholes import Borehole
from .heat_transfer import finite_line_source as FLS
from .heat_transfer import similarities


def uniform_heat_extraction(boreholes, time, alpha, use_similarities=True,
                            disTol=0.1, tol=1.0e-6, processes=None,
                            disp=False):
    """
    Evaluate the g-function with uniform heat extraction along boreholes.

    This function superimposes the finite line source (FLS) solution to
    estimate the g-function of a geothermal bore field.

    Parameters
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    use_similarities : bool, optional
        True if similarities are used to limit the number of FLS evaluations.
        Default is True.
    disTol : float, optional
        Absolute tolerance (in meters) on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
        Default is 0.1.
    tol : float, optional
        Relative tolerance on length and depth. Two lenths H1, H2
        (or depths D1, D2) are considered equal if abs(H1 - H2)/H2 < tol.
        Default is 1.0e-6.
    processes : int, optional
        Number of processors to use in calculations. If the value is set to
        None, a number of processors equal to cpu_count() is used.
        Default is None.
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.

    Returns
    -------
    gFunction : float or array
        Values of the g-function

    Examples
    --------
    >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
    >>> b2 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
    >>> alpha = 1.0e-6
    >>> time = np.array([1.0*10**i for i in range(4, 12)])
    >>> gt.gfunction.uniform_heat_extraction([b1, b2], time, alpha)
    array([ 0.75978163,  1.84860837,  2.98861057,  4.33496051,  6.29199383,
        8.13636888,  9.08401497,  9.20736188])

    """
    if disp:
        print(60*'-')
        print('Calculating g-function for uniform heat extraction rate')
        print(60*'-')
    # Initialize chrono
    tic = tim.time()
    # Number of boreholes
    nBoreholes = len(boreholes)
    # Number of time values
    nt = len(np.atleast_1d(time))
    # Initialize heat extraction rates
    Q = np.ones(nBoreholes)
    # Initialize g-function
    gFunction = np.zeros_like(np.atleast_1d(time))
    # Borehole lengths
    H = np.array([b.H for b in boreholes])

    # Calculate borehole to borehole thermal response factors
    h_ij = thermal_response_factors(
        boreholes, time, alpha, use_similarities=use_similarities,
        splitRealAndImage=False, disTol=disTol, tol=tol, processes=processes,
        disp=disp)
    toc1 = tim.time()

    # Evaluate g-function at all times
    if disp:
        print('Building and solving system of equations ...')
    for i in range(nt):
        Tb = h_ij[:,:,i].dot(Q)
        # The g-function is the average of all borehole wall temperatures
        gFunction[i] = np.dot(Tb, H) / sum(H)
    toc2 = tim.time()

    if disp:
        print('{} sec'.format(toc2 - toc1))
        print('Total time for g-function evaluation: {} sec'.format(
                toc2 - tic))
        print(60*'-')

    # Return float if time is a scalar
    if np.isscalar(time):
        gFunction = np.asscalar(gFunction)

    return gFunction


def uniform_temperature(boreholes, time, alpha, nSegments=12, method='linear',
                        use_similarities=True, disTol=0.1, tol=1.0e-6,
                        processes=None, disp=False):
    """
    Evaluate the g-function with uniform borehole wall temperature.

    This function superimposes the finite line source (FLS) solution to
    estimate the g-function of a geothermal bore field. Each borehole is
    modeled as a series of finite line source segments, as proposed in
    [#CimminoBernier2014]_.

    Parameters
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    nSegments : int, optional
        Number of line segments used per borehole.
        Default is 12.
    method : string, optional
        Interpolation method used for segment-to-segment thermal response
        factors. See documentation for scipy.interpolate.interp1d.
        Default is linear.
    use_similarities : bool, optional
        True if similarities are used to limit the number of FLS evaluations.
        Default is True.
    disTol : float, optional
        Absolute tolerance (in meters) on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
        Default is 0.1.
    tol : float, optional
        Relative tolerance on length and depth. Two lenths H1, H2
        (or depths D1, D2) are considered equal if abs(H1 - H2)/H2 < tol.
        Default is 1.0e-6.
    processes : int, optional
        Number of processors to use in calculations. If the value is set to
        None, a number of processors equal to cpu_count() is used.
        Default is None.
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.

    Returns
    -------
    gFunction : float or array
        Values of the g-function

    Examples
    --------
    >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
    >>> b2 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
    >>> alpha = 1.0e-6
    >>> time = np.array([1.0*10**i for i in range(4, 12)])
    >>> gt.gfunction.uniform_temperature([b1, b2], time, alpha)
    array([ 0.75978079,  1.84859851,  2.98852756,  4.33406497,  6.27830732,
        8.05746656,  8.93697282,  9.04925079])

    References
    ----------
    .. [#CimminoBernier2014] Cimmino, M., & Bernier, M. (2014). A
       semi-analytical method to generate g-functions for geothermal bore
       fields. International Journal of Heat and Mass Transfer, 70, 641-650.

    """
    if disp:
        print(60*'-')
        print('Calculating g-function for uniform borehole wall temperature')
        print(60*'-')
    # Initialize chrono
    tic = tim.time()
    # Number of boreholes
    nBoreholes = len(boreholes)
    # Total number of line sources
    nSources = nSegments*nBoreholes
    # Number of time values
    nt = len(np.atleast_1d(time))
    # Initialize g-function
    gFunction = np.zeros_like(np.atleast_1d(time))
    # Initialize segment heat extraction rates
    Q = np.zeros((nSources, nt))

    # Split boreholes into segments
    boreSegments = _borehole_segments(boreholes, nSegments)
    # Vector of time values
    t = np.atleast_1d(time).flatten()
    # Calculate segment to segment thermal response factors
    h_ij = thermal_response_factors(
        boreSegments, t, alpha, use_similarities=use_similarities,
        splitRealAndImage=True, disTol=disTol, tol=tol, processes=processes,
        disp=disp)
    toc1 = tim.time()

    if disp:
        print('Building and solving system of equations ...')
    # -------------------------------------------------------------------------
    # Build a system of equation [A]*[X] = [B] for the evaluation of the
    # g-function. [A] is a coefficient matrix, [X] = [Qb,Tb] is a state
    # space vector of the borehole heat extraction rates and borehole wall
    # temperature (equal for all segments), [B] is a coefficient vector.
    # -------------------------------------------------------------------------

    # Segment lengths
    Hb = np.array([b.H for b in boreSegments])
    # Vector of time steps
    dt = np.hstack((t[0], t[1:] - t[:-1]))
    # Spline object for thermal response factors
    h_dt = interp1d(t, h_ij, kind=method, axis=2)
    # Thermal response factors evaluated at t=dt
    h_dt = h_dt(dt)
    # Thermal response factor increments
    dh_ij = np.concatenate((h_ij[:,:,0:1], h_ij[:,:,1:]-h_ij[:,:,:-1]), axis=2)

    # Energy conservation: sum([Qb*Hb]) = sum([Hb])
    A_eq2 = np.hstack((Hb, 0.))
    B_eq2 = np.atleast_1d(np.sum(Hb))

    # Build and solve the system of equations at all times
    for p in range(nt):
        # Current thermal response factor matrix
        h_ij_dt = h_dt[:,:,p]
        # Reconstructed load history
        Q_reconstructed = load_history_reconstruction(t[0:p+1], Q[:,0:p+1])
        # Borehole wall temperature for zero heat extraction at current step
        Tb_0 = _temporal_superposition(dh_ij, Q_reconstructed)
        # Spatial superposition: [Tb] = [Tb0] + [h_ij_dt]*[Qb]
        A_eq1 = np.hstack((h_ij_dt, -np.ones((nSources, 1))))
        B_eq1 = -Tb_0
        # Assemble equations
        A = np.vstack((A_eq1, A_eq2))
        B = np.hstack((B_eq1, B_eq2))
        # Solve the system of equations
        X = np.linalg.solve(A, B)
        # Store calculated heat extraction rates
        Q[:,p] = X[0:nSources]
        # The borehole wall temperatures are equal for all segments
        Tb = X[-1]
        gFunction[p] = Tb

    toc2 = tim.time()
    if disp:
        print('{} sec'.format(toc2 - toc1))
        print('Total time for g-function evaluation: {} sec'.format(
                toc2 - tic))
        print(60*'-')

    # Return float if time is a scalar
    if np.isscalar(time):
        gFunction = np.asscalar(gFunction)

    return gFunction


def equal_inlet_temperature(boreholes, UTubes, m_flow, cp, time, alpha,
                            method='linear', nSegments=12,
                            use_similarities=True, disTol=0.1, tol=1.0e-6,
                            processes=None, disp=False):
    """
    Evaluate the g-function with equal inlet fluid temperatures.

    This function superimposes the finite line source (FLS) solution to
    estimate the g-function of a geothermal bore field. Each borehole is
    modeled as a series of finite line source segments, as proposed in
    [#Cimmino2015]_.

    Parameters
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    UTubes : list of pipe objects
        Model for pipes inside each borehole.
    m_flow : float or array
        Fluid mass flow rate per borehole (in kg/s).
    cp : fluid specific isobaric heat capacity (in J/kg.K)
        Model with fluid properties.
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    nSegments : int, optional
        Number of line segments used per borehole.
        Default is 12.
    method : string, optional
        Interpolation method used for segment-to-segment thermal response
        factors. See documentation for scipy.interpolate.interp1d.
        Default is 'linear'.
    use_similarities : bool, optional
        True if similarities are used to limit the number of FLS evaluations.
        Default is True.
    disTol : float, optional
        Absolute tolerance (in meters) on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
        Default is 0.1.
    tol : float, optional
        Relative tolerance on length and depth. Two lenths H1, H2
        (or depths D1, D2) are considered equal if abs(H1 - H2)/H2 < tol.
        Default is 1.0e-6.
    processes : int, optional
        Number of processors to use in calculations. If the value is set to
        None, a number of processors equal to cpu_count() is used.
        Default is None.
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.

    Returns
    -------
    gFunction : float or array
        Values of the g-function

    Examples
    --------
    >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
    >>> b2 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
    >>> alpha = 1.0e-6
    >>> time = np.array([1.0*10**i for i in range(4, 12)])
    >>> gt.gfunction.uniform_temperature([b1, b2], time, alpha)
    array([ 0.75978079,  1.84859851,  2.98852756,  4.33406497,  6.27830732,
        8.05746656,  8.93697282,  9.04925079])

    References
    ----------
    .. [#Cimmino2015] Cimmino, M. (2015). The effects of borehole thermal
       resistances and fluid flow rate on the g-functions of geothermal bore
       fields. International Journal of Heat and Mass Transfer, 91, 1119-1127.

    """
    if disp:
        print(60*'-')
        print('Calculating g-function for equal inlet fluid temperature')
        print(60*'-')
    # Initialize chrono
    tic = tim.time()
    # Number of boreholes
    nBoreholes = len(boreholes)
    # Total number of line sources
    nSources = nSegments*nBoreholes
    # Number of time values
    nt = len(np.atleast_1d(time))
    # Initialize g-function
    gFunction = np.zeros_like(np.atleast_1d(time))
    # Initialize segment heat extraction rates
    Q = np.zeros((nSources, nt))

    # If m_flow is supplied as float, apply m_flow to all boreholes
    if np.isscalar(m_flow):
        m_flow = np.tile(m_flow, nBoreholes)

    # Split boreholes into segments
    boreSegments = _borehole_segments(boreholes, nSegments)
    # Vector of time values
    t = np.atleast_1d(time).flatten()
    # Calculate segment to segment thermal response factors
    h_ij = thermal_response_factors(
        boreSegments, t, alpha, use_similarities=use_similarities,
        splitRealAndImage=True, disTol=disTol, tol=tol, processes=processes,
        disp=disp)
    toc1 = tim.time()

    if disp:
        print('Building and solving system of equations ...')
    # -------------------------------------------------------------------------
    # Build a system of equation [A]*[X] = [B] for the evaluation of the
    # g-function. [A] is a coefficient matrix, [X] = [Qb,Tb,Tf_in] is a state
    # space vector of the borehole heat extraction rates, borehole wall
    # temperatures and inlet fluid temperature (equal for all boreholes),
    # [B] is a coefficient vector.
    # -------------------------------------------------------------------------

    # Segment lengths
    Hb = np.array([b.H for b in boreSegments])
    # Vector of time steps
    dt = np.hstack((t[0], t[1:] - t[:-1]))
    # Spline object for thermal response factors
    h_dt = interp1d(t, h_ij, kind=method, axis=2)
    # Thermal response factors evaluated at t=dt
    h_dt = h_dt(dt)
    # Thermal response factor increments
    dh_ij = np.concatenate((h_ij[:,:,0:1], h_ij[:,:,1:]-h_ij[:,:,:-1]), axis=2)

    # Energy balance on borehole segments:
    # [Q_{b,i}] = [a_in]*[T_{f,in}] + [a_{b,i}]*[T_{b,i}]
    A_eq2 = np.hstack((-np.eye(nSources), np.zeros((nSources, nSources + 1))))
    B_eq2 = np.zeros(nSources)
    for i in range(nBoreholes):
        # Coefficients for current borehole
        a_in, a_b = UTubes[i].coefficients_borehole_heat_extraction_rate(
                m_flow[i], cp, nSegments)
        # Matrix coefficients ranges
        # Rows
        j1 = i*nSegments
        j2 = (i+1) * nSegments
        # Columns
        n1 = j1 + nSources
        n2 = j2 + nSources
        # Segment length
        Hi = boreholes[i].H / nSegments
        A_eq2[j1:j2, -1:] = a_in / (-2.0*pi*UTubes[i].k_s*Hi)
        A_eq2[j1:j2, n1:n2] = a_b / (-2.0*pi*UTubes[i].k_s*Hi)

    # Energy conservation: sum([Qb*Hb]) = sum([Hb])
    A_eq3 = np.hstack((Hb, np.zeros(nSources + 1)))
    B_eq3 = np.atleast_1d(np.sum(Hb))

    # Build and solve the system of equations at all times
    for p in range(nt):
        # Current thermal response factor matrix
        h_ij_dt = h_dt[:,:,p]
        # Reconstructed load history
        Q_reconstructed = load_history_reconstruction(t[0:p+1], Q[:,0:p+1])
        # Borehole wall temperature for zero heat extraction at current step
        Tb_0 = _temporal_superposition(dh_ij, Q_reconstructed)
        # Spatial superposition: [Tb] = [Tb0] + [h_ij_dt]*[Qb]
        A_eq1 = np.hstack((h_ij_dt,
                           -np.eye(nSources),
                           np.zeros((nSources, 1))))
        B_eq1 = -Tb_0
        # Assemble equations
        A = np.vstack((A_eq1, A_eq2, A_eq3))
        B = np.hstack((B_eq1, B_eq2, B_eq3))
        # Solve the system of equations
        X = np.linalg.solve(A, B)
        # Store calculated heat extraction rates
        Q[:,p] = X[0:nSources]
        # The gFunction is equal to the average borehole wall temperature
        Tb = X[nSources:2*nSources]
        gFunction[p] = Tb.dot(Hb) / np.sum(Hb)

    toc2 = tim.time()
    if disp:
        print('{} sec'.format(toc2 - toc1))
        print('Total time for g-function evaluation: {} sec'.format(
                toc2 - tic))
        print(60*'-')

    # Return float if time is a scalar
    if np.isscalar(time):
        gFunction = np.asscalar(gFunction)

    return gFunction


def mixed_inlet_temperature(boreholes, UTubes, bore_connectivity, m_flow, cp,
                            time, alpha, method='linear', nSegments=12,
                            use_similarities=True, disTol=0.1, tol=1.0e-6,
                            processes=None, disp=False):
    """
    Evaluate the g-function with equal inlet fluid temperatures.

    This function superimposes the finite line source (FLS) solution to
    estimate the g-function of a geothermal bore field. Each borehole is
    modeled as a series of finite line source segments, as proposed in
    [#Cimmino2015]_.

    Parameters
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    UTubes : list of pipe objects
        Model for pipes inside each borehole.
    bore_connectivity : list
        Index of fluid inlet into each borehole. -1 corresponds to a borehole
        connected to the bore field inlet.
    m_flow : array
        Fluid mass flow rate in each borehole (in kg/s).
    cp : fluid specific isobaric heat capacity (in J/kg.K)
        Model with fluid properties.
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    nSegments : int, optional
        Number of line segments used per borehole.
        Default is 12.
    method : string, optional
        Interpolation method used for segment-to-segment thermal response
        factors. See documentation for scipy.interpolate.interp1d.
        Default is 'linear'.
    use_similarities : bool, optional
        True if similarities are used to limit the number of FLS evaluations.
        Default is True.
    disTol : float, optional
        Absolute tolerance (in meters) on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
        Default is 0.1.
    tol : float, optional
        Relative tolerance on length and depth. Two lenths H1, H2
        (or depths D1, D2) are considered equal if abs(H1 - H2)/H2 < tol.
        Default is 1.0e-6.
    processes : int, optional
        Number of processors to use in calculations. If the value is set to
        None, a number of processors equal to cpu_count() is used.
        Default is None.
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.

    Returns
    -------
    gFunction : float or array
        Values of the g-function

    Examples
    --------
    >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
    >>> b2 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
    >>> alpha = 1.0e-6
    >>> time = np.array([1.0*10**i for i in range(4, 12)])
    >>> gt.gfunction.uniform_temperature([b1, b2], time, alpha)
    array([ 0.75978079,  1.84859851,  2.98852756,  4.33406497,  6.27830732,
        8.05746656,  8.93697282,  9.04925079])

    References
    ----------
    .. [#Cimmino2015] Cimmino, M. (2015). The effects of borehole thermal
       resistances and fluid flow rate on the g-functions of geothermal bore
       fields. International Journal of Heat and Mass Transfer, 91, 1119-1127.

    """
    if disp:
        print(60*'-')
        print('Calculating g-function for mixed inlet fluid temperatures')
        print(60*'-')
    # Initialize chrono
    tic = tim.time()
    # Number of boreholes
    nBoreholes = len(boreholes)
    # Total number of line sources
    nSources = nSegments*nBoreholes
    # Number of time values
    nt = len(np.atleast_1d(time))
    # Initialize g-function
    gFunction = np.zeros_like(np.atleast_1d(time))
    # Initialize segment heat extraction rates
    Q = np.zeros((nSources, nt))

    # Verify that borehole connectivity is valid
    _verify_bore_connectivity(bore_connectivity, nBoreholes)

    # Split boreholes into segments
    boreSegments = _borehole_segments(boreholes, nSegments)
    # Vector of time values
    t = np.atleast_1d(time).flatten()
    # Calculate segment to segment thermal response factors
    h_ij = thermal_response_factors(
        boreSegments, t, alpha, use_similarities=use_similarities,
        splitRealAndImage=True, disTol=disTol, tol=tol, processes=processes,
        disp=disp)
    toc1 = tim.time()

    if disp:
        print('Building and solving system of equations ...')
    # -------------------------------------------------------------------------
    # Build a system of equation [A]*[X] = [B] for the evaluation of the
    # g-function. [A] is a coefficient matrix, [X] = [Qb,Tb,Tf_in] is a state
    # space vector of the borehole heat extraction rates, borehole wall
    # temperatures and inlet fluid temperature (into the bore field),
    # [B] is a coefficient vector.
    # -------------------------------------------------------------------------

    # Segment lengths
    Hb = np.array([b.H for b in boreSegments])
    # Vector of time steps
    dt = np.hstack((t[0], t[1:] - t[:-1]))
    # Spline object for thermal response factors
    h_dt = interp1d(t, h_ij, kind=method, axis=2)
    # Thermal response factors evaluated at t=dt
    h_dt = h_dt(dt)
    # Thermal response factor increments
    dh_ij = np.concatenate((h_ij[:,:,0:1], h_ij[:,:,1:]-h_ij[:,:,:-1]), axis=2)

    # Energy balance on borehole segments:
    # [Q_{b,i}] = [a_in]*[T_{f,in}] + [a_{b,i}]*[T_{b,i}]
    A_eq2 = np.hstack((-np.eye(nSources), np.zeros((nSources, nSources + 1))))
    B_eq2 = np.zeros(nSources)
    for i in range(nBoreholes):
        # Segment length
        Hi = boreholes[i].H / nSegments
        # Rows of equation matrix
        j1 = i*nSegments
        j2 = (i + 1)*nSegments

        # Coefficients for current borehole
        a_in, a_b = UTubes[i].coefficients_borehole_heat_extraction_rate(
                m_flow[i], cp, nSegments)
        # [a_b] is the coefficient matrix for [T_{b,i}]
        n1 = i*nSegments + nSources
        n2 = (i + 1)*nSegments + nSources
        A_eq2[j1:j2, n1:n2] = a_b / (-2.0*pi*UTubes[i].k_s*Hi)

        # Assemble matrix coefficient for [T_{f,in}] and all [T_b]
        path = _path_from_inlet(bore_connectivity, i)
        b_in = a_in
        if len(path) > 0:
            for j in path[::-1]:
                # Coefficients for borehole j
                c_in, c_b = UTubes[j].coefficients_outlet_temperature(
                        m_flow[j], cp, nSegments)
                # Assign the coefficient matrix for [T_{b,j}]
                n1 = j*nSegments + nSources
                n2 = (j + 1)*nSegments + nSources
                A_eq2[j1:j2, n1:n2] = b_in.dot(c_b)/(-2.0*pi*UTubes[i].k_s*Hi)
                # Keep on building coefficient for [T_{f,in}]
                b_in = b_in.dot(c_in)
        A_eq2[j1:j2, -1:] = b_in / (-2.0*pi*UTubes[i].k_s*Hi)

    # Energy conservation: sum([Qb*Hb]) = sum([Hb])
    A_eq3 = np.hstack((Hb, np.zeros(nSources + 1)))
    B_eq3 = np.atleast_1d(np.sum(Hb))

    # Build and solve the system of equations at all times
    for p in range(nt):
        # Current thermal response factor matrix
        h_ij_dt = h_dt[:,:,p]
        # Reconstructed load history
        Q_reconstructed = load_history_reconstruction(t[0:p+1], Q[:,0:p+1])
        # Borehole wall temperature for zero heat extraction at current step
        Tb_0 = _temporal_superposition(dh_ij, Q_reconstructed)
        # Spatial superposition: [Tb] = [Tb0] + [h_ij_dt]*[Qb]
        A_eq1 = np.hstack((h_ij_dt,
                           -np.eye(nSources),
                           np.zeros((nSources, 1))))
        B_eq1 = -Tb_0
        # Assemble equations
        A = np.vstack((A_eq1, A_eq2, A_eq3))
        B = np.hstack((B_eq1, B_eq2, B_eq3))
        # Solve the system of equations
        X = np.linalg.solve(A, B)
        # Store calculated heat extraction rates
        Q[:,p] = X[0:nSources]
        # The gFunction is equal to the average borehole wall temperature
        Tb = X[nSources:2*nSources]
        gFunction[p] = Tb.dot(Hb) / np.sum(Hb)

    toc2 = tim.time()
    if disp:
        print('{} sec'.format(toc2 - toc1))
        print('Total time for g-function evaluation: {} sec'.format(
                toc2 - tic))
        print(60*'-')

    # Return float if time is a scalar
    if np.isscalar(time):
        gFunction = np.asscalar(gFunction)

    return gFunction, Tb, Q[:,p], X[-1]


def thermal_response_factors(
        boreSegments, time, alpha, use_similarities=True,
        splitRealAndImage=True, disTol=0.1, tol=1.0e-6, processes=None,
        disp=False):
    """
    Evaluate segment-to-segment thermal response factors.

    This function goes through the list of borehole segments and evaluates
    the segments-to-segment response factors for all times in time.

    Parameters
    ----------
    boreSegments : list of Borehole objects
        List of borehole segments.
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    use_similarities : bool, optional
        True if similarities are used to limit the number of FLS evaluations.
        Default is True.
    splitRealAndImage : bool, optional
        Set to True if similarities are evaluated separately for real and image
        sources. Set to False if similarities are evaluated for the sum of the
        real and image sources.
        Default is True.
    disTol : float, optional
        Absolute tolerance (in meters) on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
        Default is 0.1.
    tol : float, optional
        Relative tolerance on length and depth. Two lenths H1, H2
        (or depths D1, D2) are considered equal if abs(H1 - H2)/H2 < tol.
        Default is 1.0e-6.
    processes : int, optional
        Number of processors to use in calculations. If the value is set to
        None, a number of processors equal to cpu_count() is used.
        Default is None.
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.

    Returns
    -------
    h_ij : array
        Segment-to-segment thermal response factors.

    """
    # Total number of line sources
    nSources = len(boreSegments)
    # Number of time values
    nt = len(np.atleast_1d(time))
    # Prepare pool of workers for parallel computation
    pool = Pool(processes=processes)
    # Initialize chrono
    tic = tim.time()

    # Initialize segment-to-segment response factors
    h_ij = np.zeros((nSources, nSources, nt))
    # Calculation is based on the choice of use_similarities
    if use_similarities:
        # Calculations with similarities
        if disp:
            print('Identifying similarities ...')
        (nSimPos, simPos, disSimPos, HSimPos, DSimPos,
         nSimNeg, simNeg, disSimNeg, HSimNeg, DSimNeg) = \
            similarities(boreSegments,
                         splitRealAndImage=splitRealAndImage,
                         disTol=disTol,
                         tol=tol,
                         processes=processes)

        toc1 = tim.time()
        if disp:
            print('{} sec'.format(toc1 - tic))
            print('Calculating segment to segment response factors ...')

        # Initialize FLS solution for the real source
        hPos = np.zeros(nt)
        # Initialize FLS solution for the image source
        hNeg = np.zeros(nt)

        # Similarities for real sources
        for s in range(nSimPos):
            n1 = simPos[s][0][0]
            n2 = simPos[s][0][1]
            b1 = boreSegments[n1]
            b2 = boreSegments[n2]
            if splitRealAndImage:
                # FLS solution for real source only
                func = partial(FLS, alpha=alpha, borehole1=b1, borehole2=b2,
                               reaSource=True, imgSource=False)
            else:
                # FLS solution for combined real and image sources
                func = partial(FLS, alpha=alpha, borehole1=b1, borehole2=b2,
                               reaSource=True, imgSource=True)
            # Evaluate the FLS solution at all times in parallel
            hPos = np.array(pool.map(func, np.atleast_1d(time)))
            # Assign thermal response factors to similar segment pairs
            for (i, j) in simPos[s]:
                h_ij[j, i, :] = hPos
                h_ij[i, j, :] = b2.H/b1.H * hPos

        # Similarities for image sources (only if splitRealAndImage=True)
        if splitRealAndImage:
            for s in range(nSimNeg):
                n1 = simNeg[s][0][0]
                n2 = simNeg[s][0][1]
                b1 = boreSegments[n1]
                b2 = boreSegments[n2]
                # FLS solution for image source only
                func = partial(FLS, alpha=alpha, borehole1=b1, borehole2=b2,
                               reaSource=False, imgSource=True)
                # Evaluate the FLS solution at all times in parallel
                hNeg = np.array(pool.map(func, time))
                # Assign thermal response factors to similar segment pairs
                for (i, j) in simNeg[s]:
                    h_ij[j, i, :] = h_ij[j, i, :] + hNeg
                    h_ij[i, j, :] = b2.H/b1.H * h_ij[j, i, :]

    else:
        # Calculations without similarities
        if disp:
            print('Calculating segment to segment response factors ...')
        for i in range(nSources):
            # Segment to same-segment thermal response factor
            # FLS solution for combined real and image sources
            b2 = boreSegments[i]
            func = partial(FLS, alpha=alpha, borehole1=b2, borehole2=b2)
            # Evaluate the FLS solution at all times in parallel
            h = np.array(pool.map(func, time))
            h_ij[i, i, :] = h

            # Segment to other segments thermal response factor
            for j in range(i+1, nSources):
                b1 = boreSegments[j]
                # Evaluate the FLS solution at all times in parallel
                func = partial(FLS, alpha=alpha, borehole1=b1, borehole2=b2)
                h = np.array(pool.map(func, time))
                h_ij[i, j, :] = h
                h_ij[j, i, :] = b2.H / b1.H * h_ij[i, j, :]

    toc2 = tim.time()
    if disp:
        print('{} sec'.format(toc2 - tic))

    # Close pool of workers
    pool.close()
    pool.join()

    # Return 2d array if time is a scalar
    if np.isscalar(time):
        h_ij = h_ij[:,:,0]

    return h_ij


def load_history_reconstruction(time, Q):
    """
    Reconstructs the load history.

    This function calculates an equivalent load history for an inverted order
    of time step sizes.

    Parameters
    ----------
    time : array
        Values of time (in seconds) in the load history.
    Q : array
        Heat extraction rates (in Watts) of all segments at all times.

    Returns
    -------
    Q_reconstructed : array
        Reconstructed load history.

    """
    # Number of heat sources
    nSources = Q.shape[0]
    # Time step sizes
    dt = np.hstack((time[0], time[1:]-time[:-1]))
    # Time vector
    t = np.hstack((0., time))
    # Inverted time step sizes
    dt_reconstructed = dt[::-1]
    # Reconstructed time vector
    t_reconstructed = np.hstack((0., np.cumsum(dt_reconstructed)))
    # Accumulated heat extracted
    f = np.hstack((np.zeros((nSources, 1)), np.cumsum(Q*dt, axis=1)))
    # Create interpolation object for accumulated heat extracted
    sf = interp1d(t, f, kind='linear', axis=1)
    # Reconstructed load history
    Q_reconstructed = (sf(t_reconstructed[1:]) - sf(t_reconstructed[:-1])) \
        / dt_reconstructed

    return Q_reconstructed


def _borehole_segments(boreholes, nSegments):
    """
    Split boreholes into segments.

    This function goes through the list of boreholes and builds a new list,
    with each borehole split into nSegments.

    Parameters
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    nSegments : int
        Number of line segments used per borehole.

    Returns
    -------
    boreSegments : list
        List of borehole segments.

    """
    boreSegments = []
    for b in boreholes:
        for i in range(nSegments):
            # Divide borehole into segments of equal length
            H = b.H / nSegments
            # Burried depth of the i-th segment
            D = b.D + i * b.H / nSegments
            # Add to list of segments
            boreSegments.append(Borehole(H, D, b.r_b, b.x, b.y))
    return boreSegments


def _temporal_superposition(dh_ij, Q):
    """
    Temporal superposition for inequal time steps.

    Parameters
    ----------
    dh_ij : array
        Values of the segment-to-segment thermal response factor increments at
        the given time step.
    Q_reconstructed : array
        Heat extraction rates of all segments at all times.

    Returns
    -------
    Tb_0 : array
        Current values of borehole wall temperatures assuming no heat
        extraction during current time step.

    """
    # Number of heat sources
    nSources = Q.shape[0]
    # Number of time steps
    nt = Q.shape[1]
    # Borehole wall temperature
    Tb_0 = np.zeros(nSources)
    # Spatial and temporal superpositions
    for it in range(nt):
        Tb_0 += dh_ij[:,:,it].dot(Q[:,nt-it-1])
    return Tb_0


def _path_from_inlet(bore_connectivity, bore_index):
    """
    Verifies that borehole connectivity is valid.

    This function raises an error if the supplied borehole connectivity is
    invalid.

    Parameters
    ----------
    bore_connectivity : list
        Index of fluid inlet into each borehole. -1 corresponds to a borehole
        connected to the bore field inlet.
    bore_index : int
        Index of borehole to evaluate path.

    Returns
    -------
    path : list
        List of boreholes leading to the borehole. An empty list is return if
        the borehole is connected to the bore field inlet
        (bore_connectivity[bore_index] == -1).

    """
    # Initialize path
    path = []
    # Index of borehole feeding into borehole (bore_index)
    index_in = bore_connectivity[bore_index]
    # Stop when bore field inlet is reached (index_in == -1)
    while not index_in == -1:
        # Add index of upstream borehole to front of path
        path.insert(0, index_in)
        # Get index of next upstream borehole
        index_in = bore_connectivity[index_in]

    return path


def _verify_bore_connectivity(bore_connectivity, nBoreholes):
    """
    Verifies that borehole connectivity is valid.

    This function raises an error if the supplied borehole connectivity is
    invalid.

    Parameters
    ----------
    bore_connectivity : list
        Index of fluid inlet into each borehole. -1 corresponds to a borehole
        connected to the bore field inlet.
    nBoreholes : int
        Number of boreholes in the bore field.

    """
    if not len(bore_connectivity) == nBoreholes:
        raise ValueError(
            'The length of the borehole connectivity list does not correspond '
            'to the number of boreholes in the bore field.')
    # Cycle through each borehole and verify that connections lead to -1
    # (-1 is the bore field inlet)
    for i in range(nBoreholes):
        n = 0 # Initialize step counter
        # Index of borehole feeding into borehole i
        index_in = bore_connectivity[i]
        # Stop when bore field inlet is reached (index_in == -1)
        while not index_in == -1:
            index_in = bore_connectivity[index_in]
            n += 1 # Increment step counter
            # Raise error if n exceeds the number of boreholes
            if n > nBoreholes:
                raise ValueError(
                    'The borehole connectivity list is invalid.')
    return
