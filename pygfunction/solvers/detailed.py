# -*- coding: utf-8 -*-
from time import perf_counter

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d as interp1d

from ._base_solver import _BaseSolver
from ..heat_transfer import finite_line_source


class Detailed(_BaseSolver):
    """
    Detailed solver for the evaluation of the g-function.

    This solver superimposes the finite line source (FLS) solution to
    estimate the g-function of a geothermal bore field. Each borehole is
    modeled as a series of finite line source segments, as proposed in
    [#Detailed-CimBer2014]_.

    Parameters
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
                **Uniform heat transfer rate**. This is corresponds to boundary
                condition *BC-I* as defined by Cimmino and Bernier (2014)
                [#Detailed-CimBer2014]_.
            - 'UBWT' :
                **Uniform borehole wall temperature**. This is corresponds to
                boundary condition *BC-III* as defined by Cimmino and Bernier
                (2014) [#Detailed-CimBer2014]_.
            - 'MIFT' :
                **Mixed inlet fluid temperatures**. This boundary condition was
                introduced by Cimmino (2015) [#Detailed-Cimmin2015]_ for
                parallel-connected boreholes and extended to mixed
                configurations by Cimmino (2019) [#Detailed-Cimmin2019]_.

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
        [#Detailed-Cimmin2024]_. Only required for the 'MIFT' boundary
        condition. Only one of 'm_flow_borehole' and 'm_flow_network' can be
        provided.
        Default is None.
    m_flow_network : float or (nMassFlow,) array, optional
        Fluid mass flow rate into the network of boreholes. If an array
        is supplied, the (nMassFlow, nMassFlow,) variable mass flow
        rate g-functions will be evaluated using the method of Cimmino
        (2024) [#Detailed-Cimmin2024]_. Only required for the 'MIFT' boundary
        condition. Only one of 'm_flow_borehole' and 'm_flow_network' can be
        provided.
        Default is None.
    cp_f : float, optional
        Fluid specific isobaric heat capacity (in J/kg.degC). Only required
        for the 'MIFT' boundary condition.
        Default is None.
    approximate_FLS : bool, optional
        Set to true to use the approximation of the FLS solution of Cimmino
        (2021) [#Detailed-Cimmin2021]_. This approximation does not require the
        numerical evaluation of any integral.
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

    References
    ----------
    .. [#Detailed-CimBer2014] Cimmino, M., & Bernier, M. (2014). A
       semi-analytical method to generate g-functions for geothermal bore
       fields. International Journal of Heat and Mass Transfer, 70, 641-650.
    .. [#Detailed-Cimmin2015] Cimmino, M. (2015). The effects of borehole
       thermal resistances and fluid flow rate on the g-functions of geothermal
       bore fields. International Journal of Heat and Mass Transfer, 91,
       1119-1127.
    .. [#Detailed-Cimmin2019] Cimmino, M. (2019). Semi-analytical method for
       g-function calculation of bore fields with series- and
       parallel-connected boreholes. Science and Technology for the Built
       Environment, 25 (8), 1007-1022.
    .. [#Detailed-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.
    .. [#Detailed-Cimmin2024] Cimmino, M. (2024). g-Functions for fields of
       series- and parallel-connected boreholes with variable fluid mass flow
       rate and reversible flow direction. Renewable Energy, 228, 120661.

    """
    def initialize(self, **kwargs) -> int:
        """
        Split boreholes into segments.

        Returns
        -------
        nSources : int
            Number of finite line heat sources in the borefield used to
            initialize the matrix of segment-to-segment thermal response
            factors (of size: nSources x nSources).

        """
        # Split boreholes into segments
        self.nBoreholes = len(self.borefield)
        self.segments = self.borefield.segments(
            self.nSegments, self.segment_ratios)
        self._i1Segments = np.cumsum(
            np.broadcast_to(self.nSegments, self.nBoreholes),
            dtype=int)
        self._i0Segments = np.concatenate(
            ([0], self._i1Segments[:-1]),
            dtype=int)
        return len(self.segments)

    def thermal_response_factors(
            self, time: npt.ArrayLike, alpha: float, kind: str = 'linear'
            ) -> interp1d:
        """
        Evaluate the segment-to-segment thermal response factors for all pairs
        of segments in the borefield at all time steps using the finite line
        source solution.

        This method returns a scipy.interpolate.interp1d object of the matrix
        of thermal response factors, containing a copy of the matrix accessible
        by h_ij.y[:nSources,:nSources,:nt+1]. The first index along the
        third axis corresponds to time t=0. The interp1d object can be used to
        obtain thermal response factors at any intermediate time by
        h_ij(t)[:nSources,:nSources].

        Attributes
        ----------
        time : float or array
            Values of time (in seconds) for which the g-function is evaluated.
        alpha : float
            Soil thermal diffusivity (in m2/s).
        kind : string, optional
            Interpolation method used for segment-to-segment thermal response
            factors. See documentation for scipy.interpolate.interp1d.
            Default is 'linear'.

        Returns
        -------
        h_ij : interp1d
            interp1d object (scipy.interpolate) of the matrix of
            segment-to-segment thermal response factors.

        """
        if self.disp:
            print('Calculating segment to segment response factors ...',
                  end='')
        # Number of time values
        nt = len(np.atleast_1d(time))
        # Initialize chrono
        tic = perf_counter()
        # Initialize segment-to-segment response factors
        h_ij = np.zeros((self.nSources, self.nSources, nt+1), dtype=self.dtype)
        nBoreholes = len(self.borefield)
        segment_lengths = self.segment_lengths

        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for same-borehole
        # thermal interactions
        # ---------------------------------------------------------------------
        h, i, j = \
            self._thermal_response_factors_borehole_to_self(time, alpha)
        # Broadcast values to h_ij matrix
        h_ij[i, j, 1:] = h
        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for
        # borehole-to-borehole thermal interactions
        # ---------------------------------------------------------------------
        i1 = self._i1Segments
        i0 = self._i0Segments
        for i, (_i0, _i1) in enumerate(zip(i0, i1)):
            # Segments of the receiving borehole
            b2 = self.segments[_i0:_i1]
            if i+1 < nBoreholes:
                # Segments of the emitting borehole
                b1 = self.segments[_i1:]
                h = finite_line_source(
                    time, alpha, b1, b2, approximation=self.approximate_FLS,
                    N=self.nFLS, M=self.mQuad)
                # Broadcast values to h_ij matrix
                h_ij[_i0:_i1, _i1:, 1:] = h
                h_ij[_i1:, _i0:_i1, 1:] = \
                    np.swapaxes(h, 0, 1) * np.divide.outer(
                        segment_lengths[_i0:_i1],
                        segment_lengths[_i1:]).T[:, :, np.newaxis]

        # Return 2d array if time is a scalar
        if np.isscalar(time):
            h_ij = h_ij[:,:,1]

        # Interp1d object for thermal response factors
        h_ij = interp1d(np.hstack((0., time)), h_ij,
                        kind=kind, copy=True, axis=2)
        toc = perf_counter()
        if self.disp: print(f' {toc - tic:.3f} sec')

        return h_ij

    def _thermal_response_factors_borehole_to_self(
            self, time: npt.ArrayLike, alpha: float) -> np.ndarray:
        """
        Evaluate the segment-to-segment thermal response factors for all pairs
        of segments between each borehole and itself.

        Attributes
        ----------
        time : float or array
            Values of time (in seconds) for which the g-function is evaluated.
        alpha : float
            Soil thermal diffusivity (in m2/s).

        Returns
        -------
        h : array
            Finite line source solution.
        i_segment : list
            Indices of the emitting segments in the bore field.
        j_segment : list
            Indices of the receiving segments in the bore field.
        """
        # Indices of the thermal response factors into h_ij
        i1 = self._i1Segments
        i0 = self._i0Segments
        i, j = zip(*[
            np.meshgrid(
                np.arange(_j0, _j1, dtype=int),
                np.arange(_j0, _j1, dtype=int),
                indexing='ij')
            for _j0, _j1 in zip(i0, i1)
            ])
        i = np.concatenate([_i.flatten() for _i in i])
        j = np.concatenate([_j.flatten() for _j in j])
        segments_i = self.segments[i]
        segments_j = self.segments[j]
        h = finite_line_source(
            time, alpha, segments_j, segments_i, outer=False,
            approximation=self.approximate_FLS, M=self.mQuad, N=self.nFLS)
        return h, i, j
