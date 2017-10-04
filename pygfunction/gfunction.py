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
                            disTol=0.1, tol=1.0e-6, processes=None):
    """
    Evaluate the g-function with uniform heat extraction along boreholes.

    This function superimposes the finite line source (FLS) solution to
    estimate the g-function of a geothermal bore field.

    Parameters
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    time : array
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    use_similarities : boolean, defaults to True
        True if similarities are used to limit the number of FLS evaluations.
    disTol : float, defaults to 0.1
        Absolute tolerance (in meters) on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
    tol : float, defaults to 1.0e-6
        Relative tolerance on length and depth. Two lenths H1, H2
        (or depths D1, D2) are considered equal if abs(H1 - H2)/H2 < tol.

    Returns
    -------
    gFunction : array
        Values of the g-function

    Examples
    --------
    >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
    >>> b2 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
    >>> alpha = 1.0e-6
    >>> time = [1.0*10**i for i in range(4, 12)]
    >>> gt.gfunction.uniform_heat_extraction([b1, b2], time, alpha)
    [0.75978163  1.84860837  2.98861057  4.33496051  6.29199383  8.13636888
     9.08401497  9.20736188]

    """
    print('------------------------------------------------------------------')
    # Number of boreholes
    nBoreholes = len(boreholes)
    # Initialize borehole-to-borehole response factors
    h_ij = np.zeros((nBoreholes, nBoreholes))
    # Initialize g-function
    gFunction = np.zeros(len(time))
    # Initialize heat extraction rates
    Q = np.ones(nBoreholes)
    # Borehole lengths
    H = np.array([b.H for b in boreholes])
    t0 = tim.time()
    if use_similarities:
        print('Identifying similarities ...')
        # Identify symmetries
        (nSimPos, simPos, disSimPos, HSimPos, DSimPos,
         nSimNeg, simNeg, disSimNeg, HSimNeg, DSimNeg) = \
            similarities(boreholes,
                       splitRealAndImage=False,
                       disTol=disTol,
                       tol=tol,
                       processes=processes)

        t1 = tim.time()
        print('{} sec (elapsed: {} sec)'.format(t1 - t0, t1 - t0))
        print('Calculating segment to segment response factors ...')
        # Calculate borehole to borehole thermal response factors
        h_ij = _segment_to_segment_thermal_response_factors_symmetries(
                boreholes, time, alpha, nSimPos, simPos, disSimPos, HSimPos,
                DSimPos, nSimNeg, simNeg, disSimNeg, HSimNeg, DSimNeg,
                splitRealAndImage=False, processes=processes)
        t2 = tim.time()
        print('{} sec (elapsed: {} sec)'.format(t2 - t1, t2 - t0))
    else:
        print('Calculating segment to segment response factors ...')
        # Symmetries unused, go through all pairs of boreholes
        h_ij = _segment_to_segment_thermal_response_factors(
                boreholes, time, alpha, processes=processes)
        t2 = tim.time()
        print('{} sec (elapsed: {} sec)'.format(t2 - t0, t2 - t0))
    print('Building and solving system of equations ...')
    Nt = len(time)
    for i in range(Nt):
        Tb = h_ij[:,:,i].dot(Q)
        # The g-function is the average of all borehole wall temperatures
        gFunction[i] = np.dot(Tb, H) / sum(H)
    t3 = tim.time()
    print('{} sec (elapsed: {} sec)'.format(t3 - t2, t3 - t0))
    print('------------------------------------------------------------------')

    return gFunction


def uniform_temperature(boreholes, time, alpha, nSegments=12, method='linear',
                        use_similarities=True, disTol=0.1, tol=1.0e-6,
                        processes=None):
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
    time : array
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    nSegments : int, defaults to 12
        Number of line segments used per borehole.
    method : string, defaults to 'linear'
        Interpolation method used for segment-to-segment thermal response
        factors. See documentation for scipy.interpolate.interp1d.
    use_similarities : boolean, defaults to True
        True if similarities are used to limit the number of FLS evaluations.
    disTol : float, defaults to 0.1
        Absolute tolerance (in meters) on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
    tol : float, defaults to 1.0e-6
        Relative tolerance on length and depth. Two lenths H1, H2
        (or depths D1, D2) are considered equal if abs(H1 - H2)/H2 < tol.

    Returns
    -------
    gFunction : array
        Values of the g-function

    Examples
    --------
    >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
    >>> b2 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
    >>> alpha = 1.0e-6
    >>> time = [1.0*10**i for i in range(4, 12)]
    >>> uniform_temperature([b1, b2], time, alpha)
    [0.75978079  1.84859851  2.98852756  4.33406497  6.27830732  8.05746656
     8.93697282  9.04925079]

    References
    ----------
    .. [#CimminoBernier2014] Cimmino, M., & Bernier, M. (2014). A
       semi-analytical method to generate g-functions for geothermal bore
       fields. International Journal of Heat and Mass Transfer, 70, 641-650.

    """
    print('------------------------------------------------------------------')
    # Number of boeholes
    nBoreholes = len(boreholes)
    # Total number of line sources
    nSources = nSegments*nBoreholes
    # Initialize g-function
    gFunction = np.zeros_like(time)
    # Number of time values
    Nt = len(time)
    # Split boreholes into segments
    boreSegments = _borehole_segments(boreholes, nSegments)
    t0 = tim.time()
    if use_similarities:
        print('Identifying similarities ...')
        # Evaluate real and image symmetries in parallel
        (nSimPos, simPos, disSimPos, HSimPos, DSimPos,
         nSimNeg, simNeg, disSimNeg, HSimNeg, DSimNeg) = \
            similarities(boreSegments,
                       splitRealAndImage=True,
                       disTol=disTol,
                       tol=tol,
                       processes=processes)

        t1 = tim.time()
        print('{} sec (elapsed: {} sec)'.format(t1 - t0, t1 - t0))
        print('Calculating segment to segment response factors ...')
        # Evaluate segment-to-segment thermal response factors
        h_ij = _segment_to_segment_thermal_response_factors_symmetries(
                boreSegments, time, alpha, nSimPos, simPos, disSimPos, HSimPos,
                DSimPos, nSimNeg, simNeg, disSimNeg, HSimNeg, DSimNeg,
                splitRealAndImage=True, processes=processes)
        t2 = tim.time()
        print('{} sec (elapsed: {} sec)'.format(t2 - t1, t2 - t0))
    else:
        print('Calculating segment to segment response factors ...')
        # Evaluate segment-to-segment thermal response factors
        h_ij = _segment_to_segment_thermal_response_factors(
                boreSegments, time, alpha, processes=processes)
        t2 = tim.time()
        print('{} sec (elapsed: {} sec)'.format(t2 - t0, t2 - t0))
    print('Building and solving system of equations ...')
    # Initialize segment heat extraction rates
    Q = np.zeros((nSources, Nt))
    # Vector [0.]
    Z = np.zeros(1)
    # Time values for interpolation of past  loads
    t_Q = np.concatenate((Z, time))
    # Time differences for interpolation of past  loads
    dt = t_Q[1:] - t_Q[:-1]
    # Cummulative values of heat extracted
    Qdt = np.zeros((nSources, Nt + 1))
    # Segment lengths
    H = np.array([[b.H for b in boreSegments] + [0.]])
    One = np.ones((nSources, 1))
    # Borehole wall temperature for zero heat extraction rate at latest time
    # step
    Tb0 = np.zeros(nSources)
    # Create list of spline objects for segment-to-segment thermal response
    # factors
    S_dt = interp1d(time,
                    h_ij,
                    kind=method,
                    axis=2)
    # Segment-to-segment heat extraction rate increments
    h_dt = S_dt(dt)
    dh = np.concatenate((h_ij[:,:,0:1], h_ij[:,:,1:]-h_ij[:,:,:-1]), axis=2)
    # Solve the system of equations at all times
    for p in range(Nt):
        H_dt = h_dt[:,:,p]
        # Superpose heat extraction rates, if necessary
        if p > 0:
            # Reconstruct load history
            # ------------------------
            # Add heat extracted last time step to cummulative heat extracted
            Qdt[:,p] = Qdt[:,p-1] + Q[:,p-1] * dt[p-1]
            Qdt[:,p+1] = Qdt[:,p]
            # Create interpolation object for cummulative heat extracted
            SQdt = interp1d(t_Q[:p+2],
                            Qdt[:,:p+2],
                            kind='linear', axis=1)
            # Times needed for load reconstruction
            tS = np.cumsum(dt[p::-1])
            # Interpolate cummulative heat extracted
            Qdt_reconstructed = SQdt(tS)
            # Split heat extracted per time step
            Qdt_reconstructed[:,1:] -= Qdt_reconstructed[:,:-1].copy()
            # Calculate heat extraction rate
            Q_reconstructed = Qdt_reconstructed / dt[p::-1]
            # Superpose past loads
            Tb0 = _temporal_superposition(time, p, dh, nSources,
                                          Q_reconstructed[:,::-1])

        # Matrix system of equations
        A = np.concatenate((np.concatenate((H_dt, -One), axis=1), H), axis=0)
        B = np.concatenate((-Tb0, np.array([np.sum(H)])), axis=0)
        X = np.linalg.solve(A, B)

        # Store calculated heat extraction rates
        Q[:,p] = X[0:nSources]
        # The borehole wall temperatures are equal for all segments
        Tb = X[-1]
        gFunction[p] = Tb

    t3 = tim.time()
    print('{} sec (elapsed: {} sec)'.format(t3 - t2, t3 - t0))
    print('------------------------------------------------------------------')
    return gFunction


def equal_inlet_temperature(boreholes, UTubes, m_flow, cp, time, alpha,
                            method='linear', nSegments=12,
                            use_similarities=True, disTol=0.1, tol=1.0e-6,
                            processes=None):
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
    time : array
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    method : string, defaults to 'linear'
        Interpolation method used for segment-to-segment thermal response
        factors. See documentation for scipy.interpolate.interp1d.
    nSegments : int, defaults to 12
        Number of line segments used per borehole.
    use_similarities : boolean, defaults to True
        True if symmetries are used to limit the number of FLS evaluations.
    disTol : float, defaults to 0.1
        Absolute tolerance (in meters) on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
    tol : float, defaults to 1.0e-6
        Relative tolerance on length and depth. Two lenths H1, H2
        (or depths D1, D2) are considered equal if abs(H1 - H2)/H2 < tol.

    Returns
    -------
    gFunction : array
        Values of the g-function

    Examples
    --------
    >>> b1 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=0., y=0.)
    >>> b2 = gt.boreholes.Borehole(H=150., D=4., r_b=0.075, x=5., y=0.)
    >>> alpha = 1.0e-6
    >>> time = [1.0*10**i for i in range(4, 12)]
    >>> gfunction.uniform_temperature([b1, b2], time, alpha)
    [0.75978079  1.84859851  2.98852756  4.33406497  6.27830732  8.05746656
     8.93697282  9.04925079]

    References
    ----------
    .. [#Cimmino2015] Cimmino, M. (2015). The effects of borehole thermal
       resistances and fluid flow rate on the g-functions of geothermal bore
       fields. International Journal of Heat and Mass Transfer, 91, 1119-1127.

    """
    print('------------------------------------------------------------------')
    # Number of boreholes
    nBoreholes = len(boreholes)
    # Total number of line sources
    nSources = nSegments*nBoreholes
    # If m_flow is supplied as float, apply m_flow to all boreholes
    if np.isscalar(m_flow):
        m_flow = np.tile(m_flow, nBoreholes)
    # Initialize g-function
    gFunction = np.zeros_like(time)
    # Initialize segment heat extraction rates
    Nt = len(time)
    # Split boreholes into segments
    boreSegments = _borehole_segments(boreholes, nSegments)
    t0 = tim.time()
    if use_similarities:
        print('Identifying similarities ...')
        # Evaluate real and image symmetries in parallel
        (nSimPos, simPos, disSimPos, HSimPos, DSimPos,
         nSimNeg, simNeg, disSimNeg, HSimNeg, DSimNeg) = \
            similarities(boreSegments,
                       splitRealAndImage=True,
                       disTol=disTol,
                       tol=tol,
                       processes=processes)

        t1 = tim.time()
        print('{} sec (elapsed: {} sec)'.format(t1 - t0, t1 - t0))
        print('Calculating segment to segment response factors ...')
        # Evaluate segment-to-segment thermal response factors
        h_ij = _segment_to_segment_thermal_response_factors_symmetries(
                boreSegments, time, alpha, nSimPos, simPos, disSimPos, HSimPos,
                DSimPos, nSimNeg, simNeg, disSimNeg, HSimNeg, DSimNeg,
                splitRealAndImage=True, processes=processes)
        t2 = tim.time()
        print('{} sec (elapsed: {} sec)'.format(t2 - t1, t2 - t0))
    else:
        print('Calculating segment to segment response factors ...')
        # Evaluate segment-to-segment thermal response factors
        h_ij = _segment_to_segment_thermal_response_factors(
                boreSegments, time, alpha, processes=processes)
        t2 = tim.time()
        print('{} sec (elapsed: {} sec)'.format(t2 - t0, t2 - t0))
    print('Building and solving system of equations ...')
    # Initialize segment heat extraction rates
    Q = np.zeros((nSources, Nt))
    # Vector [0.]
    Z = np.zeros(1)
    # Time values for interpolation of past  loads
    t_Q = np.concatenate((Z, time))
    # Time differences for interpolation of past  loads
    dt = t_Q[1:] - t_Q[:-1]
    # Cummulative values of heat extracted
    Qdt = np.zeros((nSources, Nt + 1))
    # Segment lengths
    Hu = np.array([b.H for b in boreSegments])
    # Borehole wall temperature for zero heat extraction rate at latest time
    # step
    Tb0 = np.zeros(nSources)
    # Create list of spline objects for segment-to-segment thermal response
    # factors
    S_dt = interp1d(time,
                    h_ij,
                    kind=method,
                    axis=2)
    # Segment-to-segment heat extraction rate increments
    h_dt = S_dt(dt)
    dh = np.concatenate((h_ij[:,:,0:1], h_ij[:,:,1:]-h_ij[:,:,:-1]), axis=2)
    # Equations for segment heat extraction rates
    Eq2 = np.concatenate((-np.eye(nSources),
                          np.zeros((nSources, nSources + 1))), axis=1)
    # Include sub-matrices for relation between Tin, Tb and Qb
    for i in range(nBoreholes):
        dEin, dEb = UTubes[i].coefficients_borehole_heat_extraction_rate(
                m_flow[i], cp, nSegments)
        j1 = i*nSegments
        j2 = (i+1) * nSegments
        n1 = j1 + nSources
        n2 = j2 + nSources
        Hi = boreholes[i].H / nSegments
        Eq2[j1:j2, -1] = -dEin.flatten() / (2.0*pi*UTubes[i].k_s*Hi)
        Eq2[j1:j2, n1:n2] = -dEb / (2.0*pi*UTubes[i].k_s*Hi)
    # Equation for total heat extraction rate
    Eq3 = np.concatenate((np.array([[b.H for b in boreSegments]]),
                          np.zeros((1, nSources + 1))), axis=1)
    # Solve the system of equations at all times
    for p in range(Nt):
        H_dt = h_dt[:,:,p]
        # Superpose heat extraction rates, if necessary
        if p > 0:
            # Reconstruct load history
            # Add heat extracted last time step to cummulative heat extracted
            Qdt[:,p] = Qdt[:,p-1] + Q[:,p-1] * dt[p-1]
            Qdt[:,p+1] = Qdt[:,p]
            # Create interpolation object for cummulative heat extracted
            SQdt = interp1d(t_Q[:p+2],
                            Qdt[:,:p+2],
                            kind='linear', axis=1)
            # Times needed for load reconstruction
            tS = np.cumsum(dt[p::-1])
            # Interpolate cummulative heat extracted
            Qdt_reconstructed = SQdt(tS)
            # Split heat extracted per time step
            Qdt_reconstructed[:,1:] -= Qdt_reconstructed[:,:-1].copy()
            # Calculate heat extraction rate
            Q_reconstructed = Qdt_reconstructed / dt[p::-1]
            # Superpose past loads
            Tb0 = _temporal_superposition(time, p, dh, nSources,
                                          Q_reconstructed[:,::-1])
        # Equations for spatial superposition
        Eq1 = np.concatenate((H_dt,
                              -np.eye(nSources),
                              np.zeros((nSources, 1))), axis=1)
        # Matrix system of equations
        A = np.concatenate((Eq1, Eq2, Eq3), axis=0)
        B = np.concatenate((-Tb0,
                            np.zeros(nSources),
                            np.array([np.sum(Eq3)])), axis=0)

        X = np.linalg.solve(A, B)

        # Store calculated heat extraction rates
        Q[:,p] = X[0:nSources]

        # The gFunction is equal to the average borehole wall temperature
        Tb = X[nSources:2*nSources].dot(Hu) / np.sum(Hu)
        gFunction[p] = Tb
    t3 = tim.time()
    print('{} sec (elapsed: {} sec)'.format(t3 - t2, t3 - t0))
    print('------------------------------------------------------------------')
    return gFunction


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


def _segment_to_segment_thermal_response_factors(boreSegments,
                                                 time,
                                                 alpha,
                                                 processes=None):
    """
    Evaluate segment-to-segment thermal response factors.

    This function goes through the list of borehole segments and evaluates
    the segments-to-segment response factors for all times in time.

    Parameters
    ----------
    boreSegments : list of Borehole objects
        List of borehole segments.
    time : array
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    processes : int, defaults to cpu_count()
        Number of processors to use in calculations.

    Returns
    -------
    h_ij : array
        Segment-to-segment thermal response factors.

    """
    # Prepare pool of workers for parallel computation
    pool = Pool(processes=processes)
    # Total number of line sources
    nSources = len(boreSegments)
    # Initialize segment-to-segment response factors
    h_ij = np.array([[[0. for t in time] for j in range(nSources)]
                    for i in range(nSources)])
    for i in range(nSources):
        b2 = boreSegments[i]
        # Evaluate the FLS solution at all times in parallel
        func = partial(FLS, alpha=alpha, borehole1=b2, borehole2=b2)
        h = pool.map(func, time)
        for p in range(len(time)):
            h_ij[i, i, p] = h[p]
        for j in range(i+1, nSources):
            b1 = boreSegments[j]
            # Evaluate the FLS solution at all times in parallel
            func = partial(FLS, alpha=alpha, borehole1=b1, borehole2=b2)
            h = pool.map(func, time)
            for p in range(len(time)):
                h_ij[i, j, p] = h[p]
                h_ij[j, i, p] = b2.H / b1.H * h_ij[i, j, p]
    pool.close()
    pool.join()
    return h_ij


def _segment_to_segment_thermal_response_factors_symmetries(
        boreSegments, time, alpha, nSimPos, simPos, disSimPos, HSimPos,
        DSimPos, nSimNeg, simNeg, disSimNeg, HSimNeg, DSimNeg,
        splitRealAndImage=True, processes=None):
    """
    Evaluate segment-to-segment thermal response factors using similarities.

    This function goes through the list of symmetries and evaluates
    the segments-to-segment response factors for all times in time.

    Parameters
    ----------
    boreSegments : list of Borehole objects
        List of borehole segments.
    time : list
        Values of time (in seconds) for which the g-function is evaluated.
    alpha : float
        Soil thermal diffusivity (in m2/s).
    nSimPos : integer
        Number of similarities in the evaluation of real sources
        (if splitRealAndImage=True) or sum of real and image sources
        (if splitRealAndImage=False).
    simPos : list of list of tuples
        For each similarity, a list of pairs (tuple) of borehole indexes is
        returned.
    disSimPos : list of floats
        List of distances between boreholes for each similarity.
    HSimPos : list of tuples
        List of lengths of the pairs of boreholes in each similarity.
    DSimPos : list of tuples
        List of depth of the pairs of boreholes in each similarity.
    nSimNeg : integer
        Number of similarities in the evaluation of image sources
        (if splitRealAndImage=True), equals 0 if (splitRealAndImage=False).
    simNeg : list of list of tuples
        For each similarity, a list of pairs (tuple) of borehole indexes is
        returned.
    disSimNeg : list of floats
        List of distances between boreholes for each similarity.
    HSimNeg : list of tuples
        List of lengths of the pairs of boreholes in each similarity.
    DSimNeg : list of tuples
        List of depth of the pairs of boreholes in each similarity.
    splitRealAndImage : boolean, defaults to True
        True if the evaluation of the finite line source solution should be
        split between real and image sources.
    processes : int, defaults to cpu_count()
        Number of processors to use in calculations.

    Returns
    -------
    h_ij : array
        Segment-to-segment thermal response factors.

    """
    # Prepare pool of workers for parallel computation
    pool = Pool(processes=processes)
    # Total number of line sources
    nSources = len(boreSegments)
    # Initialize segment-to-segment response factors
    h_ij = np.array([[[0. for t in time] for j in range(nSources)]
                    for i in range(nSources)])
    # Initialize FLS solution for the real source
    hPos = np.zeros_like(time)
    # Initialize FLS solution for the image source
    hNeg = np.zeros_like(time)
    # Symmetries for real sources
    for s in range(nSimPos):
        n1 = simPos[s][0][0]
        n2 = simPos[s][0][1]
        b1 = boreSegments[n1]
        b2 = boreSegments[n2]
        # Evaluate the FLS solution at all times in parallel
        if splitRealAndImage:
            func = partial(FLS, alpha=alpha, borehole1=b1, borehole2=b2,
                           reaSource=True, imgSource=False)
        else:
            func = partial(FLS, alpha=alpha, borehole1=b1, borehole2=b2,
                           reaSource=True, imgSource=True)
        hPos = pool.map(func, time)
        for (i, j) in simPos[s]:
            for p in range(len(time)):
                h_ij[j, i, p] = hPos[p]
                h_ij[i, j, p] = b2.H/b1.H * hPos[p]
    # Symmetries for image sources
    if splitRealAndImage:
        for s in range(nSimNeg):
            n1 = simNeg[s][0][0]
            n2 = simNeg[s][0][1]
            b1 = boreSegments[n1]
            b2 = boreSegments[n2]
            # Evaluate the FLS solution at all times in parallel
            func = partial(FLS, alpha=alpha, borehole1=b1, borehole2=b2,
                           reaSource=False, imgSource=True)
            hNeg = pool.map(func, time)
            for (i, j) in simNeg[s]:
                for p in range(len(time)):
                    h_ij[j, i, p] = h_ij[j, i, p] + hNeg[p]
                    h_ij[i, j, p] = b2.H/b1.H * h_ij[j, i, p]
    pool.close()
    pool.join()
    return h_ij


def _temporal_superposition(time, nt, dh, nSources, Q):
    """
    Temporal superposition for inequal time steps.

    Parameters
    ----------
    time : array
        Values of time (in seconds) for which the g-function is evaluated.
    nt : int
        index of the latest time value.
    dh : array
        Values of the segment-to-segment thermal response factor increments at
        the given time step.
    nSources : int
        Total number of line sources.
    Q : array
        Heat extraction rates of all segments at all times.

    Returns
    -------
    Tb0 : array
        Current values of borehole wall temperatures assuming no heat
        extraction during current time step.

    """
    # Borehole wall temperature assuming no heat extraction
    Tb0 = 0.
    # Spatial and temporal superpositions
    for it in range(nt+1):
        Tb0 += np.dot(dh[:,:,it], Q[:,it])
    return Tb0
