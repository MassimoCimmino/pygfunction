# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Union, List
from time import perf_counter

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d as interp1d

from ..borefield import Borefield
from ..networks import (
    Network,
    network_thermal_resistance
    )
from .. import utilities


class _BaseSolver(ABC):
    """
    Template for solver classes.

    Solver classes inherit from this class.

    Attributes
    ----------
    borefield : Borefield object
        The bore field.
    network : network object
        The network.
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    boundary_condition : str
        Boundary condition for the evaluation of the g-function. Should be one
        of

            - 'UHTR' :
                Uniform heat transfer rate.
            - 'UBWT' :
                Uniform borehole wall temperature.
            - 'MIFT' :
                Mixed inlet fluid temperatures.

    nSegments : int or list, optional
        Number of line segments used per borehole, or list of number of
        line segments used for each borehole.
        Default is 8.
    segment_ratios : array, list of arrays, or callable, optional
        Ratio of the borehole length represented by each segment. The
        sum of ratios must be equal to 1. The shape of the array is of
        (nSegments,) or list of (nSegments[i],). If segment_ratios==None,
        segments of equal lengths are considered. If a callable is provided, it
        must return an array of size (nSegments,) when provided with nSegments
        (of type int) as an argument, or an array of size (nSegments[i],) when
        provided with an element of nSegments (of type list).
        Default is :func:`utilities.segment_ratios`.
    m_flow_borehole : (nInlets,) array or (nMassFlow, nInlets,) array, optional
        Fluid mass flow rate into each circuit of the network. If a
        (nMassFlow, nInlets,) array is supplied, the
        (nMassFlow, nMassFlow,) variable mass flow rate g-functions
        will be evaluated using the method of Cimmino (2024)
        [#gFunction-CimBer2024]_. Only required for the 'MIFT' boundary
         condition. Only one of 'm_flow_borehole' and 'm_flow_network' can be
         provided.
        Default is None.
    m_flow_network : float or (nMassFlow,) array, optional
        Fluid mass flow rate into the network of boreholes. If an array
        is supplied, the (nMassFlow, nMassFlow,) variable mass flow
        rate g-functions will be evaluated using the method of Cimmino
        (2024) [#gFunction-CimBer2024]_. Only required for the 'MIFT' boundary
         condition. Only one of 'm_flow_borehole' and 'm_flow_network' can be
         provided.
        Default is None.
    cp_f : float, optional
        Fluid specific isobaric heat capacity (in J/kg.degC). Only required
        for the 'MIFT' boundary condition.
        Default is None.
    approximate_FLS : bool, optional
        Set to true to use the approximation of the FLS solution of Cimmino
        (2021). This approximation does not require the numerical evaluation of
        any integral. When using the 'equivalent' solver, the approximation is
        only applied to the thermal response at the borehole radius. Thermal
        interaction between boreholes is evaluated using the FLS solution.
        Default is False.
    nFLS : int, optional
        Number of terms in the approximation of the FLS solution. This
        parameter is unused if `approximate_FLS` is set to False.
        Default is 10. Maximum is 25.
    mQuad : int, optional
        Number of Gauss-Legendre sample points for the integral over :math:`u`
        in the inclined FLS solution.
        Default is 11.
    linear_threshold : float, optional
        Threshold time (in seconds) under which the g-function is
        linearized. The g-function value is then interpolated between 0
        and its value at the threshold. If linear_threshold==None, the
        g-function is linearized for times
        `t < r_b**2 / (25 * self.alpha)`.
        Default is None.
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.
    profiles : bool, optional
        Set to true to keep in memory the temperatures and heat extraction
        rates.
        Default is False.
    kind : string, optional
        Interpolation method used for segment-to-segment thermal response
        factors. See documentation for scipy.interpolate.interp1d.
        Default is 'linear'.
    dtype : numpy dtype, optional
        numpy data type used for matrices and vectors. Should be one of
        numpy.single or numpy.double.
        Default is numpy.double.

    """
    def __init__(
            self,
            borefield: Borefield,
            network: Union[Network, None],
            time: npt.ArrayLike,
            boundary_condition: str,
            m_flow_borehole: Union[npt.ArrayLike, None] = None,
            m_flow_network: Union[npt.ArrayLike, None] = None,
            cp_f: Union[float, None] = None,
            nSegments: int = 8,
            segment_ratios: Union[npt.ArrayLike, List[npt.ArrayLike], Callable[[int], npt.ArrayLike]] = utilities.segment_ratios,
            approximate_FLS: bool = False,
            mQuad: int = 11,
            nFLS: int = 10,
            linear_threshold: Union[float, None] = None,
            disp: bool = False,
            profiles: bool = False,
            kind: str = 'linear',
            dtype: npt.DTypeLike = np.double,
            **other_options):
        # Input attributes
        self.borefield = borefield
        self.network = network
        self.time = np.asarray(time)
        self.boundary_condition = boundary_condition
        self.m_flow_borehole = m_flow_borehole
        self.m_flow_network = m_flow_network
        self.cp_f = cp_f
        self.nSegments = nSegments
        self.segment_ratios = segment_ratios
        self.approximate_FLS = approximate_FLS
        self.mQuad = mQuad
        self.nFLS = nFLS
        self.linear_threshold = linear_threshold
        self.disp = disp
        self.profiles = profiles
        self.kind = kind
        self.dtype = dtype

        # Check the validity of inputs
        self._check_inputs()
        # Initialize the solver with solver-specific options
        self.nSources = self.initialize(**other_options)

        self.nMassFlow = 0
        if self.m_flow_borehole is not None:
            if not self.m_flow_borehole.ndim == 1:
                self.nMassFlow = np.size(self.m_flow_borehole, axis=0)
            self.m_flow_borehole = np.atleast_2d(self.m_flow_borehole)
            self.m_flow = self.m_flow_borehole
        if self.m_flow_network is not None:
            if not isinstance(self.m_flow_network, (np.floating, float)):
                self.nMassFlow = len(self.m_flow_network)
            self.m_flow_network = np.atleast_1d(self.m_flow_network)
            self.m_flow = self.m_flow_network

    @property
    def segment_lengths(self) -> np.ndarray:
        """
        Return the length of all segments in the bore field.

        The segments lengths are used for the energy balance in the calculation
        of the g-function.

        Returns
        -------
        H : array
            Array of segment lengths (in m).

        """
        return self.segments.H

    @abstractmethod
    def initialize(self, *kwargs) -> int:
        """
        Perform any calculation required at the initialization of the solver
        and returns the number of finite line heat sources in the borefield.

        Returns
        -------
        nSources : int
            Number of finite line heat sources in the borefield used to
            initialize the matrix of segment-to-segment thermal response
            factors (of size: nSources x nSources).

        """
        ...

    def solve(self, time: npt.ArrayLike, alpha: float) -> np.ndarray:
        """
        Build and solve the system of equations.

        Parameters
        ----------
        time : float or array
            Values of time (in seconds) for which the g-function is evaluated.
        alpha : float
            Soil thermal diffusivity (in m2/s).

        Returns
        -------
        gFunc : float or array
            Values of the g-function

        """
        # Number of time values
        scalar_time = isinstance(time, (float, np.floating, int, np.integer))
        if isinstance(time, list) or scalar_time:
            time = np.asarray(time)
        self.time = time
        nTimes = len(self.time)
        # Evaluate threshold time for g-function linearization
        if self.linear_threshold is None:
            time_threshold = np.max(self.borefield.r_b)**2 / (25 * alpha)
        else:
            time_threshold = self.linear_threshold
        # Find the number of g-function values to be linearized
        p_long = np.searchsorted(self.time, time_threshold, side='right')
        p0 = np.maximum(0, p_long - 1, dtype=int)
        if p_long > 0:
            time_long = np.concatenate([[time_threshold], self.time[p_long:]])
        else:
            time_long = self.time
        nTimes_long = len(time_long)
        # Calculate segment to segment thermal response factors
        h_ij = self.thermal_response_factors(time_long, alpha, kind=self.kind)
        # Segment lengths
        H_b = self.segment_lengths
        H_tot = np.sum(H_b)
        if self.disp: print('Building and solving the system of equations ...',
                            end='')
        # Initialize chrono
        tic = perf_counter()

        if self.boundary_condition == 'UHTR':
            # Initialize g-function
            gFunc = np.zeros(nTimes)
            # Initialize segment heat extraction rates
            Q_b = np.broadcast_to(1., (self.nSources, nTimes))
            # Initialize borehole wall temperatures
            T_b = np.zeros((self.nSources, nTimes), dtype=self.dtype)

            # Evaluate the g-function with uniform heat extraction along
            # boreholes
            T_b[:, p0:] = np.sum(h_ij.y[:, :, 1:], axis=1)
            gFunc[p0:] = (T_b[:, p0:].T @ H_b).T / H_tot
            # Linearize g-function for times under threshold
            if p_long > 0:
                gFunc[:p_long] = gFunc[p_long - 1] * self.time[:p_long] / time_threshold
                T_b[:, :p_long] = T_b[:, p_long - 1:p_long] * self.time[:p_long] / time_threshold

        elif self.boundary_condition == 'UBWT':
            # Initialize g-function
            gFunc = np.zeros(nTimes)
            # Initialize segment heat extraction rates
            Q_b = np.zeros((self.nSources, nTimes), dtype=self.dtype)
            T_b = np.zeros(nTimes, dtype=self.dtype)

            # Build and solve the system of equations at all times
            dt = np.concatenate(
                [time_long[0:1], time_long[1:] - time_long[:-1]])
            for p in range(nTimes_long):
                # Thermal response factors evaluated at t=dt
                h_dt = h_ij(dt[p])
                # Reconstructed load history
                Q_reconstructed = self.load_history_reconstruction(
                    time_long[0:p + 1], Q_b[:, p0:p + p0 + 1])
                # Borehole wall temperature for zero heat extraction at
                # current step
                T_b0 = self.temporal_superposition(
                    h_ij.y[:, :, 1:], Q_reconstructed)

                # Evaluate the g-function with uniform borehole wall
                # temperature
                # ---------------------------------------------------------
                # Build a system of equation [A]*[X] = [B] for the
                # evaluation of the g-function. [A] is a coefficient
                # matrix, [X] = [Q_b,T_b] is a state space vector of the
                # borehole heat extraction rates and borehole wall
                # temperature (equal for all segments), [B] is a
                # coefficient vector.
                #
                # Spatial superposition: [T_b] = [T_b0] + [h_ij_dt]*[Q_b]
                # Energy conservation: sum([Q_b*Hb]) = sum([Hb])
                # ---------------------------------------------------------
                A = np.block([[h_dt, -np.ones((self.nSources, 1),
                                              dtype=self.dtype)],
                              [H_b, 0.]])
                B = np.hstack((-T_b0, H_tot))
                # Solve the system of equations
                X = np.linalg.solve(A, B)
                # Store calculated heat extraction rates
                Q_b[:, p + p0] = X[0:self.nSources]
                # The borehole wall temperatures are equal for all segments
                T_b[p + p0] = X[-1]
                gFunc[p + p0] = T_b[p + p0]

            # Linearize g-function for times under threshold
            if p_long > 0:
                gFunc[:p_long] = gFunc[p_long - 1] * self.time[:p_long] / time_threshold
                Q_b[:,:p_long] = 1 + (Q_b[:, p_long - 1:p_long] - 1) * self.time[:p_long] / time_threshold
                T_b[:p_long] = T_b[p_long - 1] * self.time[:p_long] / time_threshold

            # Broadcast T_b to expected size
            T_b = np.broadcast_to(T_b, (self.nSources, nTimes))

        elif self.boundary_condition == 'MIFT':
            if self.nMassFlow == 0:
                # Initialize g-function
                gFunc = np.zeros((1, 1, nTimes))
                # Initialize segment heat extraction rates
                Q_b = np.zeros((1, self.nSources, nTimes), dtype=self.dtype)
                T_b = np.zeros((1, self.nSources, nTimes), dtype=self.dtype)
            else:
                # Initialize g-function
                gFunc = np.zeros((self.nMassFlow, self.nMassFlow, nTimes))
                # Initialize segment heat extraction rates
                Q_b = np.zeros(
                    (self.nMassFlow, self.nSources, nTimes), dtype=self.dtype)
                T_b = np.zeros(
                    (self.nMassFlow, self.nSources, nTimes), dtype=self.dtype)

            for j in range(np.maximum(self.nMassFlow, 1)):
                # Build and solve the system of equations at all times
                a_in_j, a_b_j = self.network.coefficients_borehole_heat_extraction_rate(
                        self.m_flow[j],
                        self.cp_f,
                        self.nSegments,
                        segment_ratios=self.segment_ratios)
                k_s = self.network.p[0].k_s
                for p in range(nTimes_long):
                    # Current thermal response factor matrix
                    if p > 0:
                        dt = time_long[p] - time_long[p-1]
                    else:
                        dt = time_long[p]
                    # Thermal response factors evaluated at t=dt
                    h_dt = h_ij(dt)
                    # Reconstructed load history
                    Q_reconstructed = self.load_history_reconstruction(
                        time_long[0:p+1], Q_b[j,:,p0:p+p0+1])
                    # Borehole wall temperature for zero heat extraction at
                    # current step
                    T_b0 = self.temporal_superposition(
                        h_ij.y[:,:,1:], Q_reconstructed)

                    # Evaluate the g-function with mixed inlet fluid
                    # temperatures
                    # ---------------------------------------------------------
                    # Build a system of equation [A]*[X] = [B] for the
                    # evaluation of the g-function. [A] is a coefficient
                    # matrix, [X] = [Q_b,T_b,Tf_in] is a state space vector of
                    # the borehole heat extraction rates, borehole wall
                    # temperatures and inlet fluid temperature (into the bore
                    # field), [B] is a coefficient vector.
                    #
                    # Spatial superposition: [T_b] = [T_b0] + [h_ij_dt]*[Q_b]
                    # Heat transfer inside boreholes:
                    # [Q_{b,i}] = [a_in]*[T_{f,in}] + [a_{b,i}]*[T_{b,i}]
                    # Energy conservation: sum([Q_b*H_b]) = sum([H_b])
                    # ---------------------------------------------------------
                    A = np.block(
                        [[h_dt,
                          -np.eye(self.nSources, dtype=self.dtype),
                          np.zeros((self.nSources, 1), dtype=self.dtype)],
                         [np.eye(self.nSources, dtype=self.dtype),
                          a_b_j / (2. * np.pi * k_s * np.atleast_2d(self.segments.H).T),
                          a_in_j / (2. * np.pi * k_s * np.atleast_2d(self.segments.H).T)],
                         [H_b, np.zeros(self.nSources + 1, dtype=self.dtype)]])
                    B = np.hstack(
                        (-T_b0,
                         np.zeros(self.nSources, dtype=self.dtype),
                         H_tot))
                    # Solve the system of equations
                    X = np.linalg.solve(A, B)
                    # Store calculated heat extraction rates
                    Q_b[j, :, p+p0] = X[0:self.nSources]
                    T_b[j, :, p+p0] = X[self.nSources:2 * self.nSources]
                    # Inlet fluid temperature
                    T_f_in = X[-1]
                    # The gFunction is equal to the effective borehole wall
                    # temperature
                    # Outlet fluid temperature
                    T_f_out = T_f_in - 2 * np.pi * k_s * H_tot / (
                        np.sum(np.abs(self.m_flow[j]) * self.cp_f))
                    # Average fluid temperature
                    T_f = 0.5 * (T_f_in + T_f_out)
                    # Borefield thermal resistance
                    R_field = network_thermal_resistance(
                        self.network, self.m_flow[j], self.cp_f)
                    # Effective borehole wall temperature
                    T_b_eff = T_f - 2 * np.pi * k_s * R_field
                    gFunc[j, j, p + p0] = T_b_eff

            for i in range(np.maximum(self.nMassFlow, 1)):
                for j in range(np.maximum(self.nMassFlow, 1)):
                    if not i == j:
                        # Inlet fluid temperature
                        a_in, a_b = self.network.coefficients_network_heat_extraction_rate(
                                self.m_flow[i],
                                self.cp_f,
                                self.nSegments,
                                segment_ratios=self.segment_ratios)
                        T_f_in = (-2 * np.pi * k_s * H_tot - a_b @ T_b[j, :, p0:]) / a_in
                        # The gFunction is equal to the effective borehole wall
                        # temperature
                        # Outlet fluid temperature
                        T_f_out = T_f_in - 2 * np.pi * k_s * H_tot / np.sum(np.abs(self.m_flow[i]) * self.cp_f)
                        # Borefield thermal resistance
                        R_field = network_thermal_resistance(
                            self.network, self.m_flow[i], self.cp_f)
                        # Effective borehole wall temperature
                        T_b_eff = 0.5 * (T_f_in + T_f_out) - 2 * np.pi * k_s * R_field
                        gFunc[i, j, p0:] = T_b_eff

            # Linearize g-function for times under threshold
            if p_long > 0:
                gFunc[:, :, :p_long] = gFunc[:, :, p_long-1] * self.time[:p_long] / time_threshold
                Q_b[:, :, :p_long] = 1 + (Q_b[:, :, p_long-1:p_long] - 1) * self.time[:p_long] / time_threshold
                T_b[:, :, :p_long] = T_b[:, :, p_long - 1:p_long] * self.time[:p_long] / time_threshold
            if self.nMassFlow == 0:
                gFunc = gFunc[0, 0, :]
                Q_b = Q_b[0, :, :]
                T_b = T_b[0, :, :]

        # Store temperature and heat extraction rate profiles
        if self.profiles:
            self.Q_b = Q_b
            self.T_b = T_b
        toc = perf_counter()
        if self.disp: print(f' {toc - tic:.3f} sec')
        return gFunc

    @staticmethod
    def load_history_reconstruction(
            time: np.ndarray, Q_b: np.ndarray) -> np.ndarray:
        """
        Reconstructs the load history.

        This function calculates an equivalent load history for an inverted
        order of time step sizes.

        Parameters
        ----------
        time : array
            Values of time (in seconds) in the load history.
        Q_b : array
            Heat extraction rates (in Watts) of all segments at all times.

        Returns
        -------
        Q_reconstructed : array
            Reconstructed load history.

        """
        # Number of heat sources
        nSources = Q_b.shape[0]
        # Time step sizes
        dt = np.hstack((time[0], time[1:] - time[:-1]))
        # Time vector
        t = np.hstack((0., time, time[-1] + time[0]))
        # Inverted time step sizes
        dt_reconstructed = dt[::-1]
        # Reconstructed time vector
        t_reconstructed = np.hstack((0., np.cumsum(dt_reconstructed)))
        # Accumulated heat extracted
        f = np.hstack(
            (np.zeros((nSources, 1)),
             np.cumsum(Q_b*dt, axis=1)))
        f = np.hstack((f, f[:, -1:]))
        # Create interpolation object for accumulated heat extracted
        sf = interp1d(t, f, kind='linear', axis=1)
        # Reconstructed load history
        Q_reconstructed = (
            sf(t_reconstructed[1:]) - sf(t_reconstructed[:-1])
            ) / dt_reconstructed

        return Q_reconstructed

    @staticmethod
    def temporal_superposition(
            h_ij: np.ndarray, Q_reconstructed: np.ndarray) -> np.ndarray:
        """
        Temporal superposition for inequal time steps.

        Parameters
        ----------
        h_ij : array
            Values of the segment-to-segment thermal response factor increments
            at the given time step.
        Q_reconstructed : array
            Reconstructed heat extraction rates of all segments at all times.

        Returns
        -------
        T_b0 : array
            Current values of borehole wall temperatures assuming no heat
            extraction during current time step.

        """
        # Number of time steps
        nTimes = Q_reconstructed.shape[1]
        # Spatial and temporal superpositions
        dQ = np.concatenate(
            (Q_reconstructed[:, 0:1],
             Q_reconstructed[:, 1:] - Q_reconstructed[:, 0:-1]),
            axis=1)[:,::-1]
        # Borehole wall temperature
        T_b0 = np.einsum('ijk,jk', h_ij[:,:,:nTimes], dQ)

        return T_b0

    def _check_inputs(self):
        """
        This method ensures that the instances filled in the Solver object
        are what is expected.

        """
        assert isinstance(self.borefield, Borefield), \
            "The borefield is not a valid 'Borefield' object."
        assert len(self.borefield) > 0, \
            "The number of boreholes must be 1 or greater."
        assert self.network is None or isinstance(self.network, Network), \
            "The network is not a valid 'Network' object."
        if self.boundary_condition == 'MIFT':
            assert not (self.m_flow_network is None and self.m_flow_borehole is None), \
                "The mass flow rate 'm_flow_borehole' or 'm_flow_network' must " \
                "be provided when using the 'MIFT' boundary condition."
            assert not (self.m_flow_network is not None and self.m_flow_borehole is not None), \
                "Only one of 'm_flow_borehole' or 'm_flow_network' can " \
                "be provided when using the 'MIFT' boundary condition."
            assert not self.cp_f is None, \
                "The heat capacity 'cp_f' must " \
                "be provided when using the 'MIFT' boundary condition."
            assert not (isinstance(self.m_flow_borehole, np.ndarray) and not np.size(self.m_flow_borehole, axis=1)==self.network.nInlets), \
                "The number of mass flow rates in 'm_flow_borehole' must " \
                "correspond to the number of circuits in the network."
        assert isinstance(self.time, np.ndarray), \
            "Time should be an array."
        assert (isinstance(self.nSegments, (int, np.integer))
                and self.nSegments >= 1) \
            or (isinstance(self.nSegments, (list, np.array))
                and len(self.nSegments) == len(self.borefield)
                and np.min(self.nSegments) >=1), \
            "The argument for number of segments `nSegments` should be " \
            "of type int or a list of integers. If passed as a list, the " \
            "length of the list should be equal to the number of boreholes" \
            "in the borefield. nSegments >= 1 is/are required."
        acceptable_boundary_conditions = ['UHTR', 'UBWT', 'MIFT']
        assert (isinstance(self.boundary_condition, str)
                and self.boundary_condition in acceptable_boundary_conditions), \
            f"Boundary condition '{self.boundary_condition}' is not an " \
            f"acceptable boundary condition. \n" \
            f"Please provide one of the following inputs : " \
            f"{acceptable_boundary_conditions}"
        assert isinstance(self.approximate_FLS, bool), \
            "The option 'approximate_FLS' should be set to True or False."
        assert isinstance(self.nFLS, int) and 1 <= self.nFLS <= 25, \
            "The option 'nFLS' should be a positive int and lower or equal " \
            "to 25."
        assert isinstance(self.disp, bool), \
            "The option 'disp' should be set to True or False."
        assert isinstance(self.profiles, bool), \
            "The option 'profiles' should be set to True or False."
        assert isinstance(self.kind, str), \
            "The option 'kind' should be set to a valid interpolation kind " \
            "in accordance with scipy.interpolate.interp1d options."
        acceptable_dtypes = (np.single, np.double)
        assert np.any([self.dtype is dtype for dtype in acceptable_dtypes]), \
            f"Data type '{self.dtype}' is not an acceptable data type. \n" \
            f"Please provide one of the following inputs : {acceptable_dtypes}"

        return
