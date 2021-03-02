from __future__ import absolute_import, division, print_function

import time as tim

import numpy as np
from scipy.constants import pi
from scipy.interpolate import interp1d as interp1d

from .boreholes import Borehole
from .heat_transfer import thermal_response_factors
from .networks import Network, network_thermal_resistance

class gFunction:
    def __init__(self, boreholes_or_network, alpha, time=None,
                 method='similarities', boundary_condition=None, options=None):
        # Check if the input is a Network object
        if isinstance(boreholes_or_network, Network):
            self.network = boreholes_or_network
            self.boreholes = boreholes_or_network.b
            if boundary_condition is None:
                self.boundary_condition = 'MIFT'
        else:
            self.network = None
            self.boreholes = boreholes_or_network
            if boundary_condition is None:
                self.boundary_condition = 'UBWT'
        self.alpha = alpha
        self.time = time
        self.method = method
        if self.method.lower()=='similarities':
            self.use_similarities = True
        else:
            self.use_similarities = False
        self.options=options
        if self.time is not None:
            self.evaluate_g_function(self.time)

    def evaluate_g_function(self, time):
        """
        Compute the g-function based on the boundary condition supplied
        Parameters
        ----------
        time : float or array, optional
            Values of time (in seconds) for which the g-function is evaluated.

        Returns
        -------
        gFunction : float or array
            Values of the g-function
        """
        self.time = time
        # self.check_assertions()  # check to make sure none of the instances in the class has an undesired type
        # provide a list of acceptable boundary conditions
        acceptable_boundary_conditions = ['UHTR', 'UBWT', 'MIFT']
        # if the boundary condition specified is not one of the acceptable ones, then warn the user
        if self.boundary_condition not in acceptable_boundary_conditions:
            raise ValueError('Boundary condition specified is not an acceptable boundary condition. \n'
                             'Please provide one of the following inputs for boundary condition: {}'.\
                             format(acceptable_boundary_conditions))

        if self.boundary_condition == 'UHTR':
            # compute g-function for uniform heat flux boundary condition
            self.gFunc = uniform_heat_extraction(self.boreholes,
                                        self.time,
                                        self.alpha,
                                        use_similarities=self.use_similarities,
                                        **self.options)
        elif self.boundary_condition == 'UBWT':
            # compute g-function for uniform borehole wall temperature boundary condition
            self.gFunc = uniform_temperature(self.boreholes,
                                    self.time,
                                    self.alpha,
                                    use_similarities=self.use_similarities,
                                    **self.options)
        elif self.boundary_condition == 'MIFT':
            # compute g-function for uniform inlet fluid temperature boundary condition
            self.gFunc = mixed_inlet_temperature(self.network,
                                        self.network.m_flow,
                                        self.network.cp,
                                        self.time,
                                        self.alpha,
                                        **self.options)
        else:
            raise ValueError('The exact error is questionable. Please double check your inputs.')

        return self.gFunc

    def check_assertions(self):
        """
        This method ensures that the instances filled in the gFunction object are what is expected.
        Returns
        -------
        None
        """
        assert isinstance(self.boreholes, list)     # boreholes must be in a list
        assert len(self.boreholes) > 0              # there must be atleast one borehole location
        assert type(self.boreholes[0] is Borehole)  # the list of boreholes must be made up of borehole objects
        assert type(self.time) is np.ndarray or type(self.time) is float
        assert type(self.alpha) is float
        assert type(self.nSegments) is int
        assert type(self.method) is str
        assert type(self.use_similarities) is bool
        assert type(self.disTol) is float
        assert type(self.tol) is float
        assert type(self.processes) is int
        assert type(self.disp) is bool
        return


class _BaseSolver(object):
    def __init__(self, boreholes, network, time, boundary_condition, nSegments=12, processes=None, disp=False, profiles=False, **other_options):
        self.boreholes = boreholes
        self.network = network
        self.time = np.atleast_1d(time).flatten()
        self.boundary_condition = boundary_condition
        self.nSegments = nSegments
        self.processes = processes
        self.disp = disp
        self.profiles = profiles
        self.nSources = self.initialize(**other_options)
        # provide a list of acceptable boundary conditions
        acceptable_boundary_conditions = ['UHTR', 'UBWT', 'MIFT']
        # if the boundary condition specified is not one of the acceptable ones, then warn the user
        if self.boundary_condition not in acceptable_boundary_conditions:
            raise ValueError('Boundary condition specified is not an acceptable boundary condition. \n'
                             'Please provide one of the following inputs for boundary condition: {}'.\
                             format(acceptable_boundary_conditions))
        return

    def solve(self, time, alpha):
        # Number of time values
        nt = len(self.time)
        # Initialize g-function
        gFunc = np.zeros(nt)
        # Initialize segment heat extraction rates
        if self.boundary_condition == 'UHTR':
            Q = 1
        else:
            Q = np.zeros((self.nSources, nt))
        if self.boundary_condition == 'UBWT':
            Tb = np.zeros(nt)
        else:
            Tb = np.zeros((self.nSources, nt))
        # Calculate segment to segment thermal response factors
        h_ij = self.thermal_response_factors(time, alpha)
        # Segment lengths
        Hb = self.segment_lengths()
        Htot = np.sum(Hb)
        if self.disp: print('Building and solving the system of equations ...')
        # Initialize chrono
        tic = tim.time()

        # Build and solve the system of equations at all times
        for p in range(nt):
            if self.boundary_condition == 'UHTR':
                # compute g-function for uniform heat flux boundary condition
                h_dt = h_ij.y[p+1]
                Tb[:,p] = np.sum(h_dt, axis=1).flatten()
                gFunc[p] = Tb.dot(Hb)/np.sum(Htot)
            else:
                # Current thermal response factor matrix
                if p > 0:
                    dt = self.time[p] - self.time[p-1]
                else:
                    dt = self.time[p]
                h_dt = h_ij(dt)
                # Reconstructed load history
                Q_reconstructed = self.load_history_reconstruction(self.time[0:p+1], Q[:,0:p+1])
                # Borehole wall temperature for zero heat extraction at current step
                Tb_0 = self.temporal_superposition(h_ij.y[:,:,1:], Q_reconstructed)
    
                if self.boundary_condition == 'UBWT':
                    # compute g-function for uniform borehole wall temperature boundary condition
                    # Spatial superposition: [Tb] = [Tb0] + [h_ij_dt]*[Qb]
                    # Energy conservation: sum([Q*Hb]) = sum([Hb])
                    A = np.block([[h_dt, -np.ones((self.nSources, 1))],
                                  [Hb, 0.]])
                    B = np.hstack((-Tb_0, Htot))
                    # Solve the system of equations
                    X = np.linalg.solve(A, B)
                    # Store calculated heat extraction rates
                    Q[:,p] = X[0:self.nSources]
                    # The borehole wall temperatures are equal for all segments
                    Tb[p] = X[-1]
                    gFunc[p] = Tb[p]
                elif self.boundary_condition == 'MIFT':
                    # compute g-function for uniform inlet fluid temperature boundary condition
                    # Spatial superposition: [Tb] = [Tb0] + [h_ij_dt]*[Qb]
                    # [Q_{b,i}] = [a_in]*[T_{f,in}] + [a_{b,i}]*[T_{b,i}]
                    # Energy conservation: sum([Q*Hb]) = sum([Hb])
                    a_in, a_b = self.network.coefficients_borehole_heat_extraction_rate(
                            self.network.m_flow, self.network.cp, self.nSegments)
                    k_s = self.network.p[0].k_s
                    A = np.block([[h_dt, -np.eye(self.nSources), np.zeros((self.nSources, 1))],
                                  [np.eye(self.nSources), a_b/(2.0*pi*k_s*np.atleast_2d(Hb).T), a_in/(2.0*pi*k_s*np.atleast_2d(Hb).T)],
                                  [Hb, np.zeros(self.nSources + 1)]])
                    B = np.hstack((-Tb_0, np.zeros(self.nSources), Htot))
                    # Solve the system of equations
                    X = np.linalg.solve(A, B)
                    # Store calculated heat extraction rates
                    Q[:,p] = X[0:self.nSources]
                    Tb[:,p] = X[self.nSources:2*self.nSources]
                    Tf_in = X[-1]
                    Tf_out = Tf_in - 2*pi*self.network.p[0].k_s*Htot/(self.network.m_flow*self.network.cp)
                    Tf = 0.5*(Tf_in + Tf_out)
                    Rfield = network_thermal_resistance(self.network, self.network.m_flow, self.network.cp)
                    Tb_eff = Tf - 2*pi*self.network.p[0].k_s*Rfield
                    gFunc[p] = Tb_eff
                if self.profiles:
                    self.Q = Q
                    self.Tb = Tb
        toc = tim.time()
        if self.disp: print('{} sec'.format(toc - tic))
        return gFunc

    def segment_lengths(self):
        # Borehole lengths
        H = np.array([b.H for b in self.boreSegments])
        return H

    def borehole_segments(self):
        boreSegments = []
        for b in self.boreholes:
            for i in range(self.nSegments):
                # Divide borehole into segments of equal length
                H = b.H / self.nSegments
                # Buried depth of the i-th segment
                D = b.D + i * b.H / self.nSegments
                # Add to list of segments
                boreSegments.append(Borehole(H, D, b.r_b, b.x, b.y))
        return boreSegments

    def temporal_superposition(self, h_ij, Q_reconstructed):
        # Number of heat sources
        nSources = Q_reconstructed.shape[0]
        # Number of time steps
        nt = Q_reconstructed.shape[1]
        # Borehole wall temperature
        Tb_0 = np.zeros(nSources)
        # Spatial and temporal superpositions
        dQ = np.concatenate((Q_reconstructed[:,0:1],
                             Q_reconstructed[:,1:]-Q_reconstructed[:,0:-1]),
                            axis=1)
        for it in range(nt):
            Tb_0 += h_ij[:,:,it].dot(dQ[:,nt-it-1])
        return Tb_0

    def load_history_reconstruction(self, time, Q):
        # Number of heat sources
        nSources = Q.shape[0]
        # Time step sizes
        dt = np.hstack((time[0], time[1:]-time[:-1]))
        # Time vector
        t = np.hstack((0., time, time[-1] + time[0]))
        # Inverted time step sizes
        dt_reconstructed = dt[::-1]
        # Reconstructed time vector
        t_reconstructed = np.hstack((0., np.cumsum(dt_reconstructed)))
        # Accumulated heat extracted
        f = np.hstack((np.zeros((nSources, 1)), np.cumsum(Q*dt, axis=1)))
        f = np.hstack((f, f[:,-1:]))
        # Create interpolation object for accumulated heat extracted
        sf = interp1d(t, f, kind='linear', axis=1)
        # Reconstructed load history
        Q_reconstructed = (sf(t_reconstructed[1:]) - sf(t_reconstructed[:-1])) \
            / dt_reconstructed
    
        return Q_reconstructed


def uniform_heat_extraction(boreholes, time, alpha, use_similarities=True,
                            disTol=0.01, tol=1.0e-6, processes=None,
                            disp=False, **kwargs):
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
        Relative tolerance on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
        Default is 0.01.
    tol : float, optional
        Relative tolerance on length and depth. Two lengths H1, H2
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
    gFunction = np.zeros(nt)
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
                        use_similarities=True, disTol=0.01, tol=1.0e-6,
                        processes=None, disp=False, **kwargs):
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
        Relative tolerance on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
        Default is 0.01.
    tol : float, optional
        Relative tolerance on length and depth. Two lengths H1, H2
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
    gFunction = np.zeros(nt)
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
    if not np.isscalar(time) and len(time) > 1:
        # Spline object for thermal response factors
        h_dt = interp1d(np.hstack((0., t)),
                        np.dstack((np.zeros((nSources,nSources)), h_ij)),
                        kind=method, axis=2)
        # Thermal response factors evaluated at t=dt
        h_dt = h_dt(dt)
    else:
        h_dt = h_ij
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
                            use_similarities=True, disTol=0.01, tol=1.0e-6,
                            processes=None, disp=False, **kwargs):
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
    cp : float
        Fluid specific isobaric heat capacity (in J/kg.K).
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
        Relative tolerance on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
        Default is 0.01.
    tol : float, optional
        Relative tolerance on length and depth. Two lengths H1, H2
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
    gFunction = np.zeros(nt)
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
    if not np.isscalar(time) and len(time) > 1:
        # Spline object for thermal response factors
        h_dt = interp1d(np.hstack((0., t)),
                        np.dstack((np.zeros((nSources,nSources)), h_ij)),
                        kind=method, axis=2)
        # Thermal response factors evaluated at t=dt
        h_dt = h_dt(dt)
    else:
        h_dt = h_ij
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


def mixed_inlet_temperature(network, m_flow, cp,
                            time, alpha, method='linear', nSegments=12,
                            use_similarities=True, disTol=0.01, tol=1.0e-6,
                            processes=None, disp=False, **kwargs):
    """
    Evaluate the g-function with mixed inlet fluid temperatures.

    This function superimposes the finite line source (FLS) solution to
    estimate the g-function of a geothermal bore field. Each borehole is
    modeled as a series of finite line source segments, as proposed in
    [#Cimmino2018]_. The piping configurations between boreholes can be any
    combination of series and parallel connections.

    Parameters
    ----------
    network : Network objects
        List of boreholes included in the bore field.
    m_flow : float or array
        Total mass flow rate into the network or inlet mass flow rates
        into each circuit of the network (in kg/s). If a float is supplied,
        the total mass flow rate is split equally into all circuits.
    cp : float or array
        Fluid specific isobaric heat capacity (in J/kg.degC).
        Must be the same for all circuits (a single float can be supplied).
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
        Relative tolerance on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
        Default is 0.01.
    tol : float, optional
        Relative tolerance on length and depth. Two lengths H1, H2
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
    >>> Utube1 = gt.pipes.SingleUTube(pos=[(-0.05, 0), (0, -0.05)],
                                      r_in=0.015, r_out=0.02,
                                      borehole=b1,k_s=2, k_g=1, R_fp=0.1)
    >>> Utube2 = gt.pipes.SingleUTube(pos=[(-0.05, 0), (0, -0.05)],
                                      r_in=0.015, r_out=0.02,
                                      borehole=b1,k_s=2, k_g=1, R_fp=0.1)
    >>> bore_connectivity = [-1, 0]
    >>> network = gt.networks.Network([b1, b2], [Utube1, Utube2], bore_connectivity)
    >>> time = np.array([1.0*10**i for i in range(4, 12)])
    >>> m_flow = 0.25
    >>> cp = 4000.
    >>> alpha = 1.0e-6
    >>> gt.gfunction.mixed_inlet_temperature(network, m_flow, cp, time, alpha)
    array([0.63782415, 1.63304116, 2.72191316, 4.04091713, 5.98240458,
       7.77216202, 8.66195828, 8.77567215])

    References
    ----------
    .. [#Cimmino2018] Cimmino, M. (2018). g-Functions for bore fields with
       mixed parallel and series connections considering the axial fluid
       temperature variations. Proceedings of the IGSHPA Sweden Research Track
       2018. Stockholm, Sweden. pp. 262-270.

    """
    if disp:
        print(60*'-')
        print('Calculating g-function for mixed inlet fluid temperatures')
        print(60*'-')
    # Initialize chrono
    tic = tim.time()
    # Number of boreholes
    nBoreholes = network.nBoreholes
    # Total number of line sources
    nSources = nSegments*nBoreholes
    # Number of time values
    nt = len(np.atleast_1d(time))
    # Initialize g-function
    gFunction = np.zeros(nt)
    # Initialize segment heat extraction rates
    Q = np.zeros((nSources, nt))

    # Ground thermal conductivity
    k_s = network.p[0].k_s
    # Split boreholes into segments
    boreSegments = _borehole_segments(network.b, nSegments)
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
    Hb = np.array([[b.H for b in boreSegments]])
    # Vector of time steps
    dt = np.hstack((t[0], t[1:] - t[:-1]))
    if not np.isscalar(time) and len(time) > 1:
        # Spline object for thermal response factors
        h_dt = interp1d(np.hstack((0., t)),
                        np.dstack((np.zeros((nSources,nSources)), h_ij)),
                        kind=method, axis=2)
        # Thermal response factors evaluated at t=dt
        h_dt = h_dt(dt)
    else:
        h_dt = h_ij
    # Thermal response factor increments
    dh_ij = np.concatenate((h_ij[:,:,0:1], h_ij[:,:,1:]-h_ij[:,:,:-1]), axis=2)

    # Energy balance on borehole segments:
    # [Q_{b,i}] = [a_in]*[T_{f,in}] + [a_{b,i}]*[T_{b,i}]
    a_in, a_b = network.coefficients_borehole_heat_extraction_rate(
            m_flow, cp, nSegments)
    A_eq2 = np.hstack((np.eye(nSources), a_b/(2.0*pi*k_s*Hb.T), a_in/(2.0*pi*k_s*Hb.T)))
    B_eq2 = np.zeros(nSources)

    # Energy conservation: sum([Qb*Hb]) = sum([Hb])
    A_eq3 = np.hstack((Hb, np.zeros((1, nSources + 1))))
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
        B = np.hstack((B_eq1, B_eq2, B_eq3))
        A = np.vstack((A_eq1, A_eq2, A_eq3))
        # Solve the system of equations
        X = np.linalg.solve(A, B)
        # Store calculated heat extraction rates
        Q[:,p] = X[0:nSources]
        # The gFunction is equal to the average borehole wall temperature
        Tf_in = X[-1]
        Tf_out = Tf_in - 2*pi*k_s*np.sum(Hb)/(m_flow*cp)
        Tf = 0.5*(Tf_in + Tf_out)
        Rfield = network_thermal_resistance(network, m_flow, cp)
        Tb_eff = Tf - 2*pi*k_s*Rfield
        gFunction[p] = Tb_eff

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
    t = np.hstack((0., time, time[-1] + time[0]))
    # Inverted time step sizes
    dt_reconstructed = dt[::-1]
    # Reconstructed time vector
    t_reconstructed = np.hstack((0., np.cumsum(dt_reconstructed)))
    # Accumulated heat extracted
    f = np.hstack((np.zeros((nSources, 1)), np.cumsum(Q*dt, axis=1)))
    f = np.hstack((f, f[:,-1:]))
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
            # Buried depth of the i-th segment
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
    Q : array
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
