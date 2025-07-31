# -*- coding: utf-8 -*-
from time import perf_counter

import numpy as np
from scipy.interpolate import interp1d as interp1d

from ._base_solver import _BaseSolver
from ..heat_transfer import (
    finite_line_source_inclined_vectorized,
    finite_line_source_vectorized
    )


class Similarities(_BaseSolver):
    """
    Similarities solver for the evaluation of the g-function.

    This solver superimposes the finite line source (FLS) solution to
    estimate the g-function of a geothermal bore field. Each borehole is
    modeled as a series of finite line source segments, as proposed in
    [#Similarities-CimBer2014]_. The number of evaluations of the FLS solution
    is decreased by identifying similar pairs of boreholes, for which the same
    FLS value can be applied [#Similarities-Cimmin2018]_.

    Parameters
    ----------
    boreholes : list of Borehole objects
        List of boreholes included in the bore field.
    network : network object
        Model of the network.
    time : float or array
        Values of time (in seconds) for which the g-function is evaluated.
    boundary_condition : str
        Boundary condition for the evaluation of the g-function. Should be one
        of

            - 'UHTR' :
                **Uniform heat transfer rate**. This corresponds to boundary
                condition *BC-I* as defined by Cimmino and Bernier (2014)
                [#Similarities-CimBer2014]_.
            - 'UBWT' :
                **Uniform borehole wall temperature**. This corresponds to
                boundary condition *BC-III* as defined by Cimmino and Bernier
                (2014) [#Similarities-CimBer2014]_.
            - 'MIFT' :
                **Mixed inlet fluid temperatures**. This boundary condition was
                introduced by Cimmino (2015) [#Similarities-Cimmin2015]_ for
                parallel-connected boreholes and extended to mixed
                configurations by Cimmino (2019) [#Similarities-Cimmin2019]_.

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
        [#Similarities-Cimmin2024]_. Only required for the 'MIFT' boundary
        condition. Only one of 'm_flow_borehole' and 'm_flow_network' can be
        provided.
        Default is None.
    m_flow_network : float or (nMassFlow,) array, optional
        Fluid mass flow rate into the network of boreholes. If an array
        is supplied, the (nMassFlow, nMassFlow,) variable mass flow
        rate g-functions will be evaluated using the method of Cimmino
        (2024) [#Similarities-Cimmin2024]_. Only required for the 'MIFT' boundary
        condition. Only one of 'm_flow_borehole' and 'm_flow_network' can be
        provided.
        Default is None.
    cp_f : float, optional
        Fluid specific isobaric heat capacity (in J/kg.degC). Only required
        for the 'MIFT' boundary condition.
        Default is None.
    approximate_FLS : bool, optional
        Set to true to use the approximation of the FLS solution of Cimmino
        (2021) [#Similarities-Cimmin2021]_. This approximation does not require
        the numerical evaluation of any integral.
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
    disTol : float, optional
        Relative tolerance on radial distance. Two distances
        (d1, d2) between two pairs of boreholes are considered equal if the
        difference between the two distances (abs(d1-d2)) is below tolerance.
        Default is 0.01.
    tol : float, optional
        Relative tolerance on length and depth. Two lengths H1, H2
        (or depths D1, D2) are considered equal if abs(H1 - H2)/H2 < tol.
        Default is 1.0e-6.

    References
    ----------
    .. [#Similarities-CimBer2014] Cimmino, M., & Bernier, M. (2014). A
       semi-analytical method to generate g-functions for geothermal bore
       fields. International Journal of Heat and Mass Transfer, 70, 641-650.
    .. [#Similarities-Cimmin2015] Cimmino, M. (2015). The effects of borehole
       thermal resistances and fluid flow rate on the g-functions of geothermal
       bore fields. International Journal of Heat and Mass Transfer, 91,
       1119-1127.
    .. [#Similarities-Cimmin2018] Cimmino, M. (2018). Fast calculation of the
       g-functions of geothermal borehole fields using similarities in the
       evaluation of the finite line source solution. Journal of Building
       Performance Simulation, 11 (6), 655-668.
    .. [#Similarities-Cimmin2019] Cimmino, M. (2019). Semi-analytical method
       for g-function calculation of bore fields with series- and
       parallel-connected boreholes. Science and Technology for the Built
       Environment, 25 (8), 1007-1022.
    .. [#Similarities-Cimmin2021] Cimmino, M. (2021). An approximation of the
       finite line source solution to model thermal interactions between
       geothermal boreholes. International Communications in Heat and Mass
       Transfer, 127, 105496.
    .. [#Similarities-Cimmin2024] Cimmino, M. (2024). g-Functions for fields of
       series- and parallel-connected boreholes with variable fluid mass flow
       rate and reversible flow direction. Renewable Energy, 228, 120661.

    """
    def initialize(self, disTol=0.01, tol=1.0e-6, **kwargs):
        """
        Split boreholes into segments and identify similarities in the
        borefield.

        Returns
        -------
        nSources : int
            Number of finite line heat sources in the borefield used to
            initialize the matrix of segment-to-segment thermal response
            factors (of size: nSources x nSources).

        """
        self.disTol = disTol
        self.tol = tol
        # Check the validity of inputs
        self._check_solver_specific_inputs()
        # Split boreholes into segments
        self.boreSegments = self.borehole_segments()
        # Initialize similarities
        self.find_similarities()
        return len(self.boreSegments)

    def thermal_response_factors(self, time, alpha, kind='linear'):
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

        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for same-borehole thermal
        # interactions (vertical boreholes)
        # ---------------------------------------------------------------------
        # Evaluate FLS at all time steps
        h, i_segment, j_segment, k_segment = \
            self._thermal_response_factors_borehole_to_self_vertical(
                time, alpha)
        # Broadcast values to h_ij matrix
        h_ij[j_segment, i_segment, 1:] = h[k_segment, :]
        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for same-borehole thermal
        # interactions (inclined boreholes)
        # ---------------------------------------------------------------------
        # Evaluate FLS at all time steps
        h, i_segment, j_segment, k_segment = \
            self._thermal_response_factors_borehole_to_self_inclined(
                time, alpha)
        # Broadcast values to h_ij matrix
        h_ij[j_segment, i_segment, 1:] = h[k_segment, :]
        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for borehole-to-borehole
        # thermal interactions (vertical boreholes)
        # ---------------------------------------------------------------------
        for pairs, distances, distance_indices in zip(
                self.borehole_to_borehole_vertical,
                self.borehole_to_borehole_distances_vertical,
                self.borehole_to_borehole_indices_vertical):
            # Index of first borehole pair in group
            i, j = pairs[0]
            # Find segment-to-segment similarities
            H1, D1, H2, D2, i_pair, j_pair, k_pair = \
                self._map_axial_segment_pairs_vertical(i, j)
            # Locate thermal response factors in the h_ij matrix
            i_segment, j_segment, k_segment, l_segment = \
                self._map_segment_pairs_vertical(
                    i_pair, j_pair, k_pair, pairs, distance_indices)
            # Evaluate FLS at all time steps
            dis = np.reshape(distances, (-1, 1))
            H1 = H1.reshape(1, -1)
            H2 = H2.reshape(1, -1)
            D1 = D1.reshape(1, -1)
            D2 = D2.reshape(1, -1)
            h = finite_line_source_vectorized(
                time, alpha, dis, H1, D1, H2, D2,
                approximation=self.approximate_FLS, N=self.nFLS)
            # Broadcast values to h_ij matrix
            h_ij[j_segment, i_segment, 1:] = h[l_segment, k_segment, :]
            if (self._compare_boreholes(self.boreholes[j], self.boreholes[i]) and
                self.nBoreSegments[i] == self.nBoreSegments[j] and
                self._uniform_segment_ratios[i] and
                self._uniform_segment_ratios[j]):
                h_ij[i_segment, j_segment, 1:] = h[l_segment, k_segment, :]
            else:
                h_ij[i_segment, j_segment, 1:] = (h * H2.T / H1.T)[l_segment, k_segment, :]
        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for borehole-to-borehole
        # thermal interactions (inclined boreholes)
        # ---------------------------------------------------------------------
        # Evaluate FLS at all time steps
        h, hT, i_segment, j_segment, k_segment = \
            self._thermal_response_factors_borehole_to_borehole_inclined(
                time, alpha)
        # Broadcast values to h_ij matrix
        h_ij[j_segment, i_segment, 1:] = h[k_segment, :]
        h_ij[i_segment, j_segment, 1:] = hT[k_segment, :]

        # Return 2d array if time is a scalar
        if np.isscalar(time):
            h_ij = h_ij[:,:,1]

        # Interp1d object for thermal response factors
        h_ij = interp1d(
            np.hstack((0., time)), h_ij,
            kind=kind, copy=True, assume_sorted=True, axis=2)
        toc = perf_counter()
        if self.disp: print(f' {toc - tic:.3f} sec')

        return h_ij

    def _thermal_response_factors_borehole_to_borehole_inclined(
            self, time, alpha):
        """
        Evaluate the segment-to-segment thermal response factors for all pairs
        of inclined segments.

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
        hT : array
            Reciprocal finite line source solution.
        i_segment : list
            Indices of the emitting segments in the bore field.
        j_segment : list
            Indices of the receiving segments in the bore field.
        k_segment : list
            Indices of unique segment pairs in the (H1, D1, H2, D2) dimensions
            corresponding to all pairs in (i_pair, j_pair) in the bore field.
        """
        rb1 = np.array([])
        x1 = np.array([])
        y1 = np.array([])
        H1 = np.array([])
        D1 = np.array([])
        tilt1 = np.array([])
        orientation1 = np.array([])
        x2 = np.array([])
        y2 = np.array([])
        H2 = np.array([])
        D2 = np.array([])
        tilt2 = np.array([])
        orientation2 = np.array([])
        i_segment = np.array([], dtype=np.uint)
        j_segment = np.array([], dtype=np.uint)
        k_segment = np.array([], dtype=np.uint)
        k0 = 0
        for pairs in self.borehole_to_borehole_inclined:
            # Index of first borehole pair in group
            i, j = pairs[0]
            # Find segment-to-segment similarities
            rb1_i, x1_i, y1_i, H1_i, D1_i, tilt1_i, orientation1_i, \
                x2_i, y2_i, H2_i, D2_i, tilt2_i, orientation2_i, \
                i_pair, j_pair, k_pair = \
                    self._map_axial_segment_pairs_inclined(i, j)
            # Locate thermal response factors in the h_ij matrix
            i_segment_i, j_segment_i, k_segment_i = \
                self._map_segment_pairs_inclined(i_pair, j_pair, k_pair, pairs)
            # Append lists
            rb1 = np.append(rb1, rb1_i)
            x1 = np.append(x1, x1_i)
            y1 = np.append(y1, y1_i)
            H1 = np.append(H1, H1_i)
            D1 = np.append(D1, D1_i)
            tilt1 = np.append(tilt1, tilt1_i)
            orientation1 = np.append(orientation1, orientation1_i)
            x2 = np.append(x2, x2_i)
            y2 = np.append(y2, y2_i)
            H2 = np.append(H2, H2_i)
            D2 = np.append(D2, D2_i)
            tilt2 = np.append(tilt2, tilt2_i)
            orientation2 = np.append(orientation2, orientation2_i)
            i_segment = np.append(i_segment, i_segment_i)
            j_segment = np.append(j_segment, j_segment_i)
            k_segment = np.append(k_segment, k_segment_i + k0)
            k0 += len(k_pair)
        # Evaluate FLS at all time steps
        h = finite_line_source_inclined_vectorized(
            time, alpha, rb1, x1, y1, H1, D1, tilt1, orientation1,
            x2, y2, H2, D2, tilt2, orientation2, M=self.mQuad,
            approximation=self.approximate_FLS, N=self.nFLS)
        hT = (h.T * H2 / H1).T
        return h, hT, i_segment, j_segment, k_segment

    def _thermal_response_factors_borehole_to_self_inclined(self, time, alpha):
        """
        Evaluate the segment-to-segment thermal response factors for all pairs
        of segments between each inclined borehole and itself.

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
        k_segment : list
            Indices of unique segment pairs in the (H1, D1, H2, D2) dimensions
            corresponding to all pairs in (i_pair, j_pair) in the bore field.
        """
        rb1 = np.array([])
        x1 = np.array([])
        y1 = np.array([])
        H1 = np.array([])
        D1 = np.array([])
        tilt1 = np.array([])
        orientation1 = np.array([])
        x2 = np.array([])
        y2 = np.array([])
        H2 = np.array([])
        D2 = np.array([])
        tilt2 = np.array([])
        orientation2 = np.array([])
        i_segment = np.array([], dtype=np.uint)
        j_segment = np.array([], dtype=np.uint)
        k_segment = np.array([], dtype=np.uint)
        k0 = 0
        for group in self.borehole_to_self_inclined:
            # Index of first borehole in group
            i = group[0]
            # Find segment-to-segment similarities
            rb1_i, x1_i, y1_i, H1_i, D1_i, tilt1_i, orientation1_i, \
                x2_i, y2_i, H2_i, D2_i, tilt2_i, orientation2_i, \
                i_pair, j_pair, k_pair = \
                    self._map_axial_segment_pairs_inclined(i, i)
            # Locate thermal response factors in the h_ij matrix
            i_segment_i, j_segment_i, k_segment_i = \
                self._map_segment_pairs_inclined(
                    i_pair, j_pair, k_pair, [(n, n) for n in group])
            # Append lists
            rb1 = np.append(rb1, rb1_i)
            x1 = np.append(x1, x1_i)
            y1 = np.append(y1, y1_i)
            H1 = np.append(H1, H1_i)
            D1 = np.append(D1, D1_i)
            tilt1 = np.append(tilt1, tilt1_i)
            orientation1 = np.append(orientation1, orientation1_i)
            x2 = np.append(x2, x2_i)
            y2 = np.append(y2, y2_i)
            H2 = np.append(H2, H2_i)
            D2 = np.append(D2, D2_i)
            tilt2 = np.append(tilt2, tilt2_i)
            orientation2 = np.append(orientation2, orientation2_i)
            i_segment = np.append(i_segment, i_segment_i)
            j_segment = np.append(j_segment, j_segment_i)
            k_segment = np.append(k_segment, k_segment_i + k0)
            k0 += len(k_pair)
        # Evaluate FLS at all time steps
        h = finite_line_source_inclined_vectorized(
            time, alpha, rb1, x1, y1, H1, D1, tilt1, orientation1,
            x2, y2, H2, D2, tilt2, orientation2, M=self.mQuad,
            approximation=self.approximate_FLS, N=self.nFLS)
        return h, i_segment, j_segment, k_segment

    def _thermal_response_factors_borehole_to_self_vertical(self, time, alpha):
        """
        Evaluate the segment-to-segment thermal response factors for all pairs
        of segments between each vertical borehole and itself.

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
        k_segment : list
            Indices of unique segment pairs in the (H1, D1, H2, D2) dimensions
            corresponding to all pairs in (i_pair, j_pair) in the bore field.
        """
        H1 = np.array([])
        D1 = np.array([])
        H2 = np.array([])
        D2 = np.array([])
        dis = np.array([])
        i_segment = np.array([], dtype=np.uint)
        j_segment = np.array([], dtype=np.uint)
        k_segment = np.array([], dtype=np.uint)
        k0 = 0
        for group in self.borehole_to_self_vertical:
            # Index of first borehole in group
            i = group[0]
            # Find segment-to-segment similarities
            H1_i, D1_i, H2_i, D2_i, i_pair, j_pair, k_pair = \
                self._map_axial_segment_pairs_vertical(i, i)
            # Locate thermal response factors in the h_ij matrix
            i_segment_i, j_segment_i, k_segment_i, l_segment_i = \
                self._map_segment_pairs_vertical(
                    i_pair, j_pair, k_pair, [(n, n) for n in group], [0])
            # Append lists
            H1 = np.append(H1, H1_i)
            D1 = np.append(D1, D1_i)
            H2 = np.append(H2, H2_i)
            D2 = np.append(D2, D2_i)
            if len(self.borehole_to_self_vertical) > 1:
                dis = np.append(dis, np.full(len(H1_i), self.boreholes[i].r_b))
            else:
                dis = self.boreholes[i].r_b
            i_segment = np.append(i_segment, i_segment_i)
            j_segment = np.append(j_segment, j_segment_i)
            k_segment = np.append(k_segment, k_segment_i + k0)
            k0 += np.max(k_pair) + 1
        # Evaluate FLS at all time steps
        h = finite_line_source_vectorized(
            time, alpha, dis, H1, D1, H2, D2,
            approximation=self.approximate_FLS, N=self.nFLS)
        return h, i_segment, j_segment, k_segment

    def find_similarities(self):
        """
        Find similarities in the FLS solution for groups of boreholes.

        This function identifies pairs of boreholes for which the evaluation
        of the Finite Line Source (FLS) solution is equivalent.

        """
        if self.disp: print('Identifying similarities ...', end='')
        # Initialize chrono
        tic = perf_counter()

        # Find similar pairs of boreholes
        # Boreholes can only be similar if their segments are similar
        self.borehole_to_self_vertical, self.borehole_to_self_inclined, \
            self.borehole_to_borehole_vertical, self.borehole_to_borehole_inclined = \
                self._find_axial_borehole_pairs(self.boreholes)
        # Find distances for each similar pairs of vertical boreholes
        self.borehole_to_borehole_distances_vertical, self.borehole_to_borehole_indices_vertical = \
            self._find_distances(
                self.boreholes, self.borehole_to_borehole_vertical)

        # Stop chrono
        toc = perf_counter()
        if self.disp: print(f' {toc - tic:.3f} sec')

        return

    def _compare_boreholes(self, borehole1, borehole2):
        """
        Compare two boreholes and checks if they have the same dimensions :
        H, D, r_b, and tilt.

        Parameters
        ----------
        borehole1 : Borehole object
            First borehole.
        borehole2 : Borehole object
            Second borehole.

        Returns
        -------
        similarity : bool
            True if the two boreholes have the same dimensions.

        """
        # Compare lengths (H), buried depth (D) and radius (r_b)
        if (abs((borehole1.H - borehole2.H)/borehole1.H) < self.tol and
            abs((borehole1.r_b - borehole2.r_b)/borehole1.r_b) < self.tol and
            abs((borehole1.D - borehole2.D)/(borehole1.D + 1e-30)) < self.tol and
            abs(abs(borehole1.tilt) - abs(borehole2.tilt))/(abs(borehole1.tilt) + 1e-30) < self.tol):
            similarity = True
        else:
            similarity = False
        return similarity

    def _compare_real_pairs_vertical(self, pair1, pair2):
        """
        Compare two pairs of vertical boreholes or segments and return True if
        the two pairs have the same FLS solution for real sources.

        Parameters
        ----------
        pair1 : Tuple of Borehole objects
            First pair of boreholes or segments.
        pair2 : Tuple of Borehole objects
            Second pair of boreholes or segments.

        Returns
        -------
        similarity : bool
            True if the two pairs have the same FLS solution.

        """
        deltaD1 = pair1[1].D - pair1[0].D
        deltaD2 = pair2[1].D - pair2[0].D

        # Equality of lengths between pairs
        cond_H = (abs((pair1[0].H - pair2[0].H)/pair1[0].H) < self.tol
            and abs((pair1[1].H - pair2[1].H)/pair1[1].H) < self.tol)
        # Equality of lengths in each pair
        equal_H = abs((pair1[0].H - pair1[1].H)/pair1[0].H) < self.tol
        # Equality of buried depths differences
        cond_deltaD = abs(deltaD1 - deltaD2)/abs(deltaD1 + 1e-30) < self.tol
        # Equality of buried depths differences if all boreholes have the same
        # length
        cond_deltaD_equal_H = abs((abs(deltaD1) - abs(deltaD2))/(abs(deltaD1) + 1e-30)) < self.tol
        if cond_H and (cond_deltaD or (equal_H and cond_deltaD_equal_H)):
            similarity = True
        else:
            similarity = False
        return similarity

    def _compare_image_pairs_vertical(self, pair1, pair2):
        """
        Compare two pairs of vertical boreholes or segments and return True if
        the two pairs have the same FLS solution for mirror sources.

        Parameters
        ----------
        pair1 : Tuple of Borehole objects
            First pair of boreholes or segments.
        pair2 : Tuple of Borehole objects
            Second pair of boreholes or segments.

        Returns
        -------
        similarity : bool
            True if the two pairs have the same FLS solution.

        """
        sumD1 = pair1[1].D + pair1[0].D
        sumD2 = pair2[1].D + pair2[0].D

        # Equality of lengths between pairs
        cond_H = (abs((pair1[0].H - pair2[0].H)/pair1[0].H) < self.tol
            and abs((pair1[1].H - pair2[1].H)/pair1[1].H) < self.tol)
        # Equality of buried depths sums
        cond_sumD = abs((sumD1 - sumD2)/(sumD1 + 1e-30)) < self.tol
        if cond_H and cond_sumD:
            similarity = True
        else:
            similarity = False
        return similarity

    def _compare_realandimage_pairs_vertical(self, pair1, pair2):
        """
        Compare two pairs of vertical boreholes or segments and return True if
        the two pairs have the same FLS solution for both real and mirror
        sources.

        Parameters
        ----------
        pair1 : Tuple of Borehole objects
            First pair of boreholes or segments.
        pair2 : Tuple of Borehole objects
            Second pair of boreholes or segments.

        Returns
        -------
        similarity : bool
            True if the two pairs have the same FLS solution.

        """
        if (self._compare_real_pairs_vertical(pair1, pair2)
            and self._compare_image_pairs_vertical(pair1, pair2)):
            similarity = True
        else:
            similarity = False
        return similarity

    def _compare_real_pairs_inclined(self, pair1, pair2):
        """
        Compare two pairs of inclined boreholes or segments and return True if
        the two pairs have the same FLS solution for real sources.

        Parameters
        ----------
        pair1 : Tuple of Borehole objects
            First pair of boreholes or segments.
        pair2 : Tuple of Borehole objects
            Second pair of boreholes or segments.

        Returns
        -------
        similarity : bool
            True if the two pairs have the same FLS solution.

        """
        dx1 = pair1[0].x - pair1[1].x; dx2 = pair2[0].x - pair2[1].x
        dy1 = pair1[0].y - pair1[1].y; dy2 = pair2[0].y - pair2[1].y
        dis1 = np.sqrt(dx1**2 + dy1**2); dis2 = np.sqrt(dx2**2 + dy2**2)
        theta_12_1 = np.arctan2(dy1, dx1); theta_12_2 = np.arctan2(dy2, dx2)
        deltaD1 = pair1[0].D - pair1[1].D; deltaD2 = pair2[0].D - pair2[1].D
        # Equality of lengths between pairs
        cond_H = (abs((pair1[0].H - pair2[0].H)/pair1[0].H) < self.tol
            and abs((pair1[1].H - pair2[1].H)/pair1[1].H) < self.tol)
        # Equality of buried depths differences
        cond_deltaD = abs(deltaD1 - deltaD2)/(abs(deltaD1) + 1e-30) < self.tol
        # Equality of distances
        cond_dis = abs(dis1 - dis2)/(abs(dis1) + 1e-30) < self.disTol
        # Equality of tilts
        cond_beta = (
            abs(abs(pair1[0].tilt) - abs(pair2[0].tilt))/(abs(pair1[0].tilt) + 1e-30) < self.tol
            and abs(abs(pair1[1].tilt) - abs(pair2[1].tilt))/(abs(pair1[1].tilt) + 1e-30) < self.tol)
        # Equality of relative orientations
        sin_b1_cos_dt1_1 = np.sin(pair1[0].tilt) * np.cos(theta_12_1 - pair1[0].orientation)
        sin_b2_cos_dt2_1 = np.sin(pair1[1].tilt) * np.cos(theta_12_1 - pair1[1].orientation)
        sin_b1_cos_dt1_2 = np.sin(pair2[0].tilt) * np.cos(theta_12_2 - pair2[0].orientation)
        sin_b2_cos_dt2_2 = np.sin(pair2[1].tilt) * np.cos(theta_12_2 - pair2[1].orientation)
        cond_theta = (
            abs(sin_b1_cos_dt1_1 - sin_b1_cos_dt1_2) / (abs(sin_b1_cos_dt1_1) + 1e-30) > self.tol
            and abs(sin_b2_cos_dt2_1 - sin_b2_cos_dt2_2) / (abs(sin_b2_cos_dt2_1) + 1e-30) > self.tol)
        if cond_H and cond_deltaD and cond_dis and cond_beta and cond_theta:
            similarity = True
        else:
            similarity = False
        return similarity

    def _compare_image_pairs_inclined(self, pair1, pair2):
        """
        Compare two pairs of inclined boreholes or segments and return True if
        the two pairs have the same FLS solution for mirror sources.

        Parameters
        ----------
        pair1 : Tuple of Borehole objects
            First pair of boreholes or segments.
        pair2 : Tuple of Borehole objects
            Second pair of boreholes or segments.

        Returns
        -------
        similarity : bool
            True if the two pairs have the same FLS solution.

        """
        dx1 = pair1[0].x - pair1[1].x; dx2 = pair2[0].x - pair2[1].x
        dy1 = pair1[0].y - pair1[1].y; dy2 = pair2[0].y - pair2[1].y
        dis1 = np.sqrt(dx1**2 + dy1**2); dis2 = np.sqrt(dx2**2 + dy2**2)
        theta_12_1 = np.arctan2(dy1, dx1); theta_12_2 = np.arctan2(dy2, dx2)
        sumD1 = pair1[0].D + pair1[1].D; sumD2 = pair2[0].D + pair2[1].D
        # Equality of lengths between pairs
        cond_H = (abs((pair1[0].H - pair2[0].H)/pair1[0].H) < self.tol
            and abs((pair1[1].H - pair2[1].H)/pair1[1].H) < self.tol)
        # Equality of buried depths sums
        cond_sumD = abs(sumD1 - sumD2)/(abs(sumD1) + 1e-30) < self.tol
        # Equality of distances
        cond_dis = abs(dis1 - dis2)/(abs(dis1) + 1e-30) < self.disTol
        # Equality of tilts
        cond_beta = (
            abs(abs(pair1[0].tilt) - abs(pair2[0].tilt))/(abs(pair1[0].tilt) + 1e-30) < self.tol
            and abs(abs(pair1[1].tilt) - abs(pair2[1].tilt))/(abs(pair1[1].tilt) + 1e-30) < self.tol)
        # Equality of relative orientations
        sin_b1_cos_dt1_1 = np.sin(pair1[0].tilt) * np.cos(theta_12_1 - pair1[0].orientation)
        sin_b2_cos_dt2_1 = np.sin(pair1[1].tilt) * np.cos(theta_12_1 - pair1[1].orientation)
        sin_b1_cos_dt1_2 = np.sin(pair2[0].tilt) * np.cos(theta_12_2 - pair2[0].orientation)
        sin_b2_cos_dt2_2 = np.sin(pair2[1].tilt) * np.cos(theta_12_2 - pair2[1].orientation)
        cond_theta = (
            abs(sin_b1_cos_dt1_1 - sin_b1_cos_dt1_2) / (abs(sin_b1_cos_dt1_1) + 1e-30) > self.tol
            and abs(sin_b2_cos_dt2_1 - sin_b2_cos_dt2_2) / (abs(sin_b2_cos_dt2_1) + 1e-30) > self.tol)
        if cond_H and cond_sumD and cond_dis and cond_beta and cond_theta:
            similarity = True
        else:
            similarity = False
        return similarity

    def _compare_realandimage_pairs_inclined(self, pair1, pair2):
        """
        Compare two pairs of inclined boreholes or segments and return True if
        the two pairs have the same FLS solution for both real and mirror
        sources.

        Parameters
        ----------
        pair1 : Tuple of Borehole objects
            First pair of boreholes or segments.
        pair2 : Tuple of Borehole objects
            Second pair of boreholes or segments.

        Returns
        -------
        similarity : bool
            True if the two pairs have the same FLS solution.

        Notes
        -----
        For inclined boreholes the similarity condition is the same for real
        and image parts of the solution.

        """
        dx1 = pair1[0].x - pair1[1].x; dx2 = pair2[0].x - pair2[1].x
        dy1 = pair1[0].y - pair1[1].y; dy2 = pair2[0].y - pair2[1].y
        dis1 = np.sqrt(dx1**2 + dy1**2); dis2 = np.sqrt(dx2**2 + dy2**2)
        theta_12_1 = np.arctan2(dy1, dx1); theta_12_2 = np.arctan2(dy2, dx2)
        # Equality of lengths between pairs
        cond_H = (abs((pair1[0].H - pair2[0].H)/pair1[0].H) < self.tol
            and abs((pair1[1].H - pair2[1].H)/pair1[1].H) < self.tol)
        # Equality of buried depths
        cond_D = (
            abs(pair1[0].D - pair2[0].D)/(abs(pair1[0].D) + 1e-30) < self.tol
            and abs(pair1[1].D - pair2[1].D)/(abs(pair1[1].D) + 1e-30) < self.tol)
        # Equality of distances
        cond_dis = abs(dis1 - dis2)/(abs(dis1) + 1e-30) < self.disTol
        # Equality of tilts
        cond_beta = (
            abs(abs(pair1[0].tilt) - abs(pair2[0].tilt))/(abs(pair1[0].tilt) + 1e-30) < self.tol
            and abs(abs(pair1[1].tilt) - abs(pair2[1].tilt))/(abs(pair1[1].tilt) + 1e-30) < self.tol)
        # Equality of relative orientations
        sin_b1_cos_dt1_1 = np.sin(pair1[0].tilt) * np.cos(theta_12_1 - pair1[0].orientation)
        sin_b2_cos_dt2_1 = np.sin(pair1[1].tilt) * np.cos(theta_12_1 - pair1[1].orientation)
        sin_b1_cos_dt1_2 = np.sin(pair2[0].tilt) * np.cos(theta_12_2 - pair2[0].orientation)
        sin_b2_cos_dt2_2 = np.sin(pair2[1].tilt) * np.cos(theta_12_2 - pair2[1].orientation)
        cond_theta = (
            abs(sin_b1_cos_dt1_1 - sin_b1_cos_dt1_2) / (abs(sin_b1_cos_dt1_1) + 1e-30) < self.tol
            and abs(sin_b2_cos_dt2_1 - sin_b2_cos_dt2_2) / (abs(sin_b2_cos_dt2_1) + 1e-30) < self.tol)
        if cond_H and cond_D and cond_dis and cond_beta and cond_theta:
            similarity = True
        else:
            similarity = False
        return similarity

    def _find_axial_borehole_pairs(self, boreholes):
        """
        Find axial (i.e. disregarding the radial distance) similarities between
        borehole pairs to simplify the evaluation of the FLS solution.

        Parameters
        ----------
        boreholes : list of Borehole objects
            Boreholes in the bore field.

        Returns
        -------
        borehole_to_self : list
            Lists of borehole indexes for each unique set of borehole
            dimensions (H, D, r_b) in the bore field.
        borehole_to_borehole : list
            Lists of tuples of borehole indexes for each unique pair of
            boreholes that share the same (pairwise) dimensions (H, D).

        """
        nBoreholes = len(boreholes)
        borehole_to_self_vertical = []
        borehole_to_self_inclined = []
        # Only check for similarities if there is more than one borehole
        if nBoreholes > 1:
            borehole_to_borehole_vertical = []
            borehole_to_borehole_inclined = []
            for i, (borehole_i, nSegments_i, ratios_i) in enumerate(
                    zip(boreholes, self.nBoreSegments, self.segment_ratios)):
                # Compare the borehole to all known unique sets of dimensions
                if borehole_i.is_vertical():
                    borehole_to_self = borehole_to_self_vertical
                    compare_pairs = self._compare_realandimage_pairs_vertical
                else:
                    borehole_to_self = borehole_to_self_inclined
                    compare_pairs = self._compare_realandimage_pairs_inclined
                for k, borehole_set in enumerate(borehole_to_self):
                    m = borehole_set[0]
                    # Add the borehole to the group if a similar borehole is
                    # found
                    if (self._compare_boreholes(borehole_i, boreholes[m]) and
                        (self._equal_segment_ratios or
                         (nSegments_i == self.nBoreSegments[m] and
                          np.allclose(ratios_i,
                                      self.segment_ratios[m],
                                      rtol=self.tol)))):
                        borehole_set.append(i)
                        break
                else:
                    # If no similar boreholes are known, append the groups
                    borehole_to_self.append([i])

                for j, (borehole_j, nSegments_j, ratios_j) in enumerate(
                        zip(boreholes[i+1:],
                            self.nBoreSegments[i+1:],
                            self.segment_ratios[i+1:]),
                        start=i+1):
                    pair0 = (borehole_i, borehole_j) # pair
                    pair1 = (borehole_j, borehole_i) # reciprocal pair
                    # Compare pairs of boreholes to known unique pairs
                    if borehole_i.is_vertical() and borehole_j.is_vertical():
                        borehole_to_borehole = borehole_to_borehole_vertical
                        compare_pairs = self._compare_realandimage_pairs_vertical
                    else:
                        borehole_to_borehole = borehole_to_borehole_inclined
                        compare_pairs = self._compare_realandimage_pairs_inclined
                    for pairs in borehole_to_borehole:
                        m, n = pairs[0]
                        pair_ref = (boreholes[m], boreholes[n])
                        # Add the pair (or the reciprocal pair) to a group
                        # if a similar one is found
                        if (compare_pairs(pair0, pair_ref) and
                            (self._equal_segment_ratios or
                             (nSegments_i == self.nBoreSegments[m] and
                              nSegments_j == self.nBoreSegments[n] and
                              np.allclose(ratios_i,
                                          self.segment_ratios[m],
                                          rtol=self.tol) and
                              np.allclose(ratios_j,
                                          self.segment_ratios[n],
                                          rtol=self.tol)))):
                            pairs.append((i, j))
                            break
                        elif (compare_pairs(pair1, pair_ref) and
                              (self._equal_segment_ratios or
                               (nSegments_j == self.nBoreSegments[m] and
                                nSegments_i == self.nBoreSegments[n] and
                                np.allclose(ratios_j,
                                            self.segment_ratios[m],
                                            rtol=self.tol) and
                                np.allclose(ratios_i,
                                            self.segment_ratios[n],
                                            rtol=self.tol)))):
                            pairs.append((j, i))
                            break
                    # If no similar pairs are known, append the groups
                    else:
                        borehole_to_borehole.append([(i, j)])

        else:
            # Outputs for a single borehole
            if boreholes[0].is_vertical:
                borehole_to_self_vertical = [[0]]
                borehole_to_self_inclined = []
            else:
                borehole_to_self_vertical = []
                borehole_to_self_inclined = [[0]]
            borehole_to_borehole_vertical = []
            borehole_to_borehole_inclined = []
        return borehole_to_self_vertical, borehole_to_self_inclined, \
            borehole_to_borehole_vertical, borehole_to_borehole_inclined

    def _find_distances(self, boreholes, borehole_to_borehole):
        """
        Find unique distances between pairs of boreholes for each unique pair
        of boreholes in the bore field.

        Parameters
        ----------
        boreholes : list of Borehole objects
            Boreholes in the bore field.
        borehole_to_borehole : list
            Lists of tuples of borehole indexes for each unique pair of
            boreholes that share the same (pairwise) dimensions (H, D).

        Returns
        -------
        borehole_to_borehole_distances : list
            Sorted lists of borehole-to-borehole radial distances for each
            unique pair of boreholes.
        borehole_to_borehole_indices : list
            Lists of indexes of distances associated with each borehole pair.

        """
        nGroups = len(borehole_to_borehole)
        borehole_to_borehole_distances = [[] for i in range(nGroups)]
        borehole_to_borehole_indices = \
            [np.empty(len(group), dtype=np.uint) for group in borehole_to_borehole]
        # Find unique distances for each group
        for i, (pairs, distances, distance_indices) in enumerate(
                zip(borehole_to_borehole,
                    borehole_to_borehole_distances,
                    borehole_to_borehole_indices)):
            nPairs = len(pairs)
            # Array of all borehole-to-borehole distances within the group
            all_distances = np.array(
                [boreholes[pair[0]].distance(boreholes[pair[1]])
                 for pair in pairs])
            # Indices to sort the distance array
            i_sort = all_distances.argsort()
            # Sort the distance array
            distances_sorted = all_distances[i_sort]
            j0 = 0
            j1 = 1
            nDis = 0
            # For each increasing distance in the sorted array :
            # 1 - find all distances that are within tolerance
            # 2 - add the average distance in the list of unique distances
            # 3 - associate the distance index to all pairs for the identified
            #     distances
            # 4 - re-start at the next distance index not yet accounted for.
            while j0 < nPairs and j1 > 0:
                # Find the first distance outside tolerance
                j1 = np.argmax(
                    distances_sorted >= (1+self.disTol)*distances_sorted[j0])
                if j1 > j0:
                    # Average distance between pairs of boreholes
                    distances.append(np.mean(distances_sorted[j0:j1]))
                    # Apply distance index to borehole pairs
                    distance_indices[i_sort[j0:j1]] = nDis
                else:
                    # Average distance between pairs of boreholes
                    distances.append(np.mean(distances_sorted[j0:]))
                    # Apply distance index to borehole pairs
                    distance_indices[i_sort[j0:]] = nDis
                j0 = j1
                nDis += 1
        return borehole_to_borehole_distances, borehole_to_borehole_indices

    def _map_axial_segment_pairs_vertical(
            self, i, j, reaSource=True, imgSource=True):
        """
        Find axial (i.e. disregarding the radial distance) similarities between
        segment pairs along two boreholes to simplify the evaluation of the
        FLS solution.

        The returned H1, D1, H2, and D2 can be used to evaluate the segment-to-
        segment response factors using scipy.integrate.quad_vec.

        Parameters
        ----------
        i : int
            Index of the first borehole.
        j : int
            Index of the second borehole.

        Returns
        -------
        H1 : array
            Length of the emitting segments.
        D1 : array
            Array of buried depths of the emitting segments.
        H2 : array
            Length of the receiving segments.
        D2 : array
            Array of buried depths of the receiving segments.
        i_pair : list
            Indices of the emitting segments along a borehole.
        j_pair : list
            Indices of the receiving segments along a borehole.
        k_pair : list
            Indices of unique segment pairs in the (H1, D1, H2, D2) dimensions
            corresponding to all pairs in (i_pair, j_pair).

        """
        # Initialize local variables
        borehole1 = self.boreholes[i]
        borehole2 = self.boreholes[j]
        assert reaSource or imgSource, \
            "At least one of reaSource and imgSource must be True."
        if reaSource and imgSource:
            # Find segment pairs for the full (real + image) FLS solution
            compare_pairs = self._compare_realandimage_pairs_vertical
        elif reaSource:
            # Find segment pairs for the real FLS solution
            compare_pairs = self._compare_real_pairs_vertical
        elif imgSource:
            # Find segment pairs for the image FLS solution
            compare_pairs = self._compare_image_pairs_vertical
        # Dive both boreholes into segments
        segments1 = borehole1.segments(
            self.nBoreSegments[i], segment_ratios=self.segment_ratios[i])
        segments2 = borehole2.segments(
            self.nBoreSegments[j], segment_ratios=self.segment_ratios[j])
        # Prepare lists of segment lengths
        H1 = []
        H2 = []
        # Prepare lists of segment buried depths
        D1 = []
        D2 = []
        # All possible pairs (i, j) of indices between segments
        i_pair = np.repeat(np.arange(self.nBoreSegments[i], dtype=np.uint),
                           self.nBoreSegments[j])
        j_pair = np.tile(np.arange(self.nBoreSegments[j], dtype=np.uint),
                         self.nBoreSegments[i])
        # Empty list of indices for unique pairs
        k_pair = np.empty(self.nBoreSegments[i] * self.nBoreSegments[j],
                          dtype=np.uint)
        unique_pairs = []
        nPairs = 0

        p = 0
        for ii, segment_i in enumerate(segments1):
            for jj, segment_j in enumerate(segments2):
                pair = (segment_i, segment_j)
                # Compare the segment pairs to all known unique pairs
                for k, pair_k in enumerate(unique_pairs):
                    m, n = pair_k[0], pair_k[1]
                    pair_ref = (segments1[m], segments2[n])
                    # Stop if a similar pair is found and assign the index
                    if compare_pairs(pair, pair_ref):
                        k_pair[p] = k
                        break
                # If no similar pair is found : add a new pair, increment the
                # number of unique pairs, and extract the associated buried
                # depths
                else:
                    k_pair[p] = nPairs
                    H1.append(segment_i.H)
                    H2.append(segment_j.H)
                    D1.append(segment_i.D)
                    D2.append(segment_j.D)
                    unique_pairs.append((ii, jj))
                    nPairs += 1
                p += 1
        return np.array(H1), np.array(D1), np.array(H2), np.array(D2), i_pair, j_pair, k_pair

    def _map_axial_segment_pairs_inclined(
            self, i, j, reaSource=True, imgSource=True):
        """
        Find axial similarities between segment pairs along two boreholes to
        simplify the evaluation of the FLS solution.

        The returned H1, D1, H2, and D2 can be used to evaluate the segment-to-
        segment response factors using scipy.integrate.quad_vec.

        Parameters
        ----------
        i : int
            Index of the first borehole.
        j : int
            Index of the second borehole.

        Returns
        -------
        rb1 : array
            Radii of the emitting heat sources.
        x1 : array
            x-Positions of the emitting heat sources.
        y1 : array
            y-Positions of the emitting heat sources.
        H1 : array
            Lengths of the emitting heat sources.
        D1 : array
            Buried depths of the emitting heat sources.
        tilt1 : array
            Angles (in radians) from vertical of the emitting heat sources.
        orientation1 : array
            Directions (in radians) of the tilt the emitting heat sources.
        x2 : array
            x-Positions of the receiving heat sources.
        y2 : array
            y-Positions of the receiving heat sources.
        H2 : array
            Lengths of the receiving heat sources.
        D2 : array
            Buried depths of the receiving heat sources.
        tilt2 : array
            Angles (in radians) from vertical of the receiving heat sources.
        orientation2 : array
            Directions (in radians) of the tilt the receiving heat sources.
        i_pair : list
            Indices of the emitting segments along a borehole.
        j_pair : list
            Indices of the receiving segments along a borehole.
        k_pair : list
            Indices of unique segment pairs in the (H1, D1, H2, D2) dimensions
            corresponding to all pairs in (i_pair, j_pair).
        """
        # Initialize local variables
        borehole1 = self.boreholes[i]
        borehole2 = self.boreholes[j]
        assert reaSource or imgSource, \
            "At least one of reaSource and imgSource must be True."
        if reaSource and imgSource:
            # Find segment pairs for the full (real + image) FLS solution
            compare_pairs = self._compare_realandimage_pairs_inclined
        elif reaSource:
            # Find segment pairs for the real FLS solution
            compare_pairs = self._compare_real_pairs_inclined
        elif imgSource:
            # Find segment pairs for the image FLS solution
            compare_pairs = self._compare_image_pairs_inclined
        # Dive both boreholes into segments
        segments1 = borehole1.segments(
            self.nBoreSegments[i], segment_ratios=self.segment_ratios[i])
        segments2 = borehole2.segments(
            self.nBoreSegments[j], segment_ratios=self.segment_ratios[j])
        # Prepare lists of FLS-inclined arguments
        rb1 = []
        x1 = []
        y1 = []
        H1 = []
        D1 = []
        tilt1 = []
        orientation1 = []
        x2 = []
        y2 = []
        H2 = []
        D2 = []
        tilt2 = []
        orientation2 = []
        # All possible pairs (i, j) of indices between segments
        i_pair = np.repeat(np.arange(self.nBoreSegments[i], dtype=np.uint),
                           self.nBoreSegments[j])
        j_pair = np.tile(np.arange(self.nBoreSegments[j], dtype=np.uint),
                         self.nBoreSegments[i])
        # Empty list of indices for unique pairs
        k_pair = np.empty(self.nBoreSegments[i] * self.nBoreSegments[j],
                          dtype=np.uint)
        unique_pairs = []
        nPairs = 0

        p = 0
        for ii, segment_i in enumerate(segments1):
            for jj, segment_j in enumerate(segments2):
                pair = (segment_i, segment_j)
                # Compare the segment pairs to all known unique pairs
                for k, pair_k in enumerate(unique_pairs):
                    m, n = pair_k[0], pair_k[1]
                    pair_ref = (segments1[m], segments2[n])
                    # Stop if a similar pair is found and assign the index
                    if compare_pairs(pair, pair_ref):
                        k_pair[p] = k
                        break
                # If no similar pair is found : add a new pair, increment the
                # number of unique pairs, and extract the associated buried
                # depths
                else:
                    k_pair[p] = nPairs
                    rb1.append(segment_i.r_b)
                    x1.append(segment_i.x)
                    y1.append(segment_i.y)
                    H1.append(segment_i.H)
                    D1.append(segment_i.D)
                    tilt1.append(segment_i.tilt)
                    orientation1.append(segment_i.orientation)
                    x2.append(segment_j.x)
                    y2.append(segment_j.y)
                    H2.append(segment_j.H)
                    D2.append(segment_j.D)
                    tilt2.append(segment_j.tilt)
                    orientation2.append(segment_j.orientation)
                    unique_pairs.append((ii, jj))
                    nPairs += 1
                p += 1
        return np.array(rb1), np.array(x1), np.array(y1), np.array(H1), \
            np.array(D1), np.array(tilt1), np.array(orientation1), \
            np.array(x2), np.array(y2), np.array(H2), np.array(D2), \
            np.array(tilt2), np.array(orientation2), i_pair, j_pair, k_pair

    def _map_segment_pairs_vertical(
            self, i_pair, j_pair, k_pair, borehole_to_borehole,
            borehole_to_borehole_indices):
        """
        Return the maping of the unique segment-to-segment thermal response
        factors (h) to the complete h_ij array of the borefield, such that:

            h_ij[j_segment, i_segment, :nt] = h[:nt, l_segment, k_segment].T,

        where h is the array of unique segment-to-segment thermal response
        factors for a given unique pair of boreholes at all unique distances.

        Parameters
        ----------
        i_pair : list
            Indices of the emitting segments.
        j_pair : list
            Indices of the receiving segments.
        k_pair : list
            Indices of unique segment pairs in the (H1, D1, H2, D2) dimensions
            corresponding to all pairs in (i_pair, j_pair).
        borehole_to_borehole : list
            Tuples of borehole indexes.
        borehole_to_borehole_indices : list
            Indexes of distances.

        Returns
        -------
        i_segment : list
            Indices of the emitting segments in the bore field.
        j_segment : list
            Indices of the receiving segments in the bore field.
        k_segment : list
            Indices of unique segment pairs in the (H1, D1, H2, D2) dimensions
            corresponding to all pairs in (i_pair, j_pair) in the bore field.
        l_segment : list
            Indices of unique distances for all pairs in (i_pair, j_pair)
            in the bore field.

        """
        i_segment = np.concatenate(
            [i_pair + self._i0Segments[i] for (i, j) in borehole_to_borehole])
        j_segment = np.concatenate(
            [j_pair + self._i0Segments[j] for (i, j) in borehole_to_borehole])
        k_segment = np.tile(k_pair, len(borehole_to_borehole))
        l_segment = np.concatenate(
            [np.repeat(i, len(k_pair)) for i in borehole_to_borehole_indices])
        return i_segment, j_segment, k_segment, l_segment

    def _map_segment_pairs_inclined(
            self, i_pair, j_pair, k_pair, borehole_to_borehole):
        """
        Return the maping of the unique segment-to-segment thermal response
        factors (h) to the complete h_ij array of the borefield, such that:

            h_ij[j_segment, i_segment, :nt] = h[:nt, k_segment].T,

        where h is the array of unique segment-to-segment thermal response
        factors for a given unique pair of boreholes at all unique distances.

        Parameters
        ----------
        i_pair : list
            Indices of the emitting segments.
        j_pair : list
            Indices of the receiving segments.
        k_pair : list
            Indices of unique segment pairs in the (H1, D1, H2, D2) dimensions
            corresponding to all pairs in (i_pair, j_pair).
        borehole_to_borehole : list
            Tuples of borehole indexes.

        Returns
        -------
        i_segment : list
            Indices of the emitting segments in the bore field.
        j_segment : list
            Indices of the receiving segments in the bore field.
        k_segment : list
            Indices of unique segment pairs in the (H1, D1, H2, D2) dimensions
            corresponding to all pairs in (i_pair, j_pair) in the bore field.

        """
        i_segment = np.concatenate(
            [i_pair + self._i0Segments[i] for (i, j) in borehole_to_borehole])
        j_segment = np.concatenate(
            [j_pair + self._i0Segments[j] for (i, j) in borehole_to_borehole])
        k_segment = np.tile(k_pair, len(borehole_to_borehole))
        return i_segment, j_segment, k_segment

    def _check_solver_specific_inputs(self):
        """
        This method ensures that solver specific inputs to the Solver object
        are what is expected.

        """
        assert isinstance(self.disTol, (np.floating, float)) and self.disTol > 0., \
            "The distance tolerance 'disTol' should be a positive float."
        assert isinstance(self.tol, (np.floating, float)) and self.tol > 0., \
            "The relative tolerance 'tol' should be a positive float."
        return
