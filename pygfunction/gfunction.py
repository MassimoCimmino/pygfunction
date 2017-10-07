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
    >>> time = [1.0*10**i for i in range(4, 12)]
    >>> gt.gfunction.uniform_heat_extraction([b1, b2], time, alpha)
    [0.75978163  1.84860837  2.98861057  4.33496051  6.29199383  8.13636888
     9.08401497  9.20736188]

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
    method : string, defaults to 'linear'
        Interpolation method used for segment-to-segment thermal response
        factors. See documentation for scipy.interpolate.interp1d.
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
        if disp: print('Identifying similarities ...')
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
