# -*- coding: utf-8 -*-
from collections.abc import Callable
from itertools import combinations_with_replacement
from time import perf_counter
from typing import Tuple, List, Union

import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d as interp1d

from ..boreholes import Borehole
from ..borefield import Borefield
from .base_solver import _BaseSolver
from ..heat_transfer import finite_line_source, finite_line_source_vertical


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
                [#Similarities-CimBer2014]_.
            - 'UBWT' :
                **Uniform borehole wall temperature**. This is corresponds to
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

    """
    def initialize(
            self, disTol: float = 0.01, tol: float = 1e-6, **kwargs) -> int:
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
        self.segments = self.borefield.segments(self.nSegments, self.segment_ratios)
        self._i1Segments = np.cumsum(
            np.broadcast_to(self.nSegments, len(self.borefield)),
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
        obtain thermal response factors at any intermediat time by
        h_ij(t)[:nSources,:nSources].

        Attributes
        ----------
        time : float or array
            Values of time (in seconds) for which the g-function is evaluated.
        alpha : float
            Soil thermal diffusivity (in m2/s).
        kind : str, optional
            Interpolation method used for segment-to-segment thermal response
            factors. See documentation for scipy.interpolate.interp1d.
            Default is 'linear'.

        Returns
        -------
        h_ij : interp1d object
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

        # Find unique boreholes geometries
        unique_boreholes, unique_borehole_indices, unique_nSegments, \
            unique_segment_ratios = \
                self._find_unique_borehole_geometries(
                    self.borefield, nSegments=self.nSegments,
                    segment_ratios=self.segment_ratios, rtol=self.tol)
        # Split vertical and inclined boreholes
        vertical_boreholes, vertical_nSegments, vertical_segment_ratios, \
            vertical_indices, inclined_boreholes, inclined_nSegments, \
            inclined_segment_ratios, inclined_indices = \
                self._split_vertical_and_inclined_boreholes(
                    unique_boreholes, nSegments=unique_nSegments,
                    segment_ratios=unique_segment_ratios,
                    indices=unique_borehole_indices)

        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for same-borehole thermal
        # interactions (vertical boreholes)
        # ---------------------------------------------------------------------
        if len(vertical_boreholes) > 0:
            segments_j, segments_i = \
                self._segment_pairs_same_borehole(
                    vertical_boreholes, vertical_nSegments,
                    segment_ratios=vertical_segment_ratios)
            h = finite_line_source(
                time, alpha, segments_j, segments_i, outer=False)
    
            # Local indices of segment pairs
            nVerticalBoreholes = len(vertical_boreholes)
            if isinstance(vertical_nSegments, (int, np.integer)):
                nSegments = np.broadcast_to(vertical_nSegments, nVerticalBoreholes)
            else:
                nSegments = vertical_nSegments
            local_segment_indices = \
                [np.triu_indices(nSegments[i]) for i in range(nVerticalBoreholes)]
            local_segment_indices_i = [indices[0] for indices in local_segment_indices]
            local_segment_indices_j = [indices[1] for indices in local_segment_indices]
            # Global indices of the first segments of boreholes
            m0 = [self._i0Segments[indices] for indices in vertical_indices]
            # Indices of the matrix h
            m = np.concatenate([np.add.outer(m0_i, indices_i).flatten() for m0_i, indices_i in zip(m0, local_segment_indices_i)])
            n = np.concatenate([np.add.outer(m0_j, indices_j).flatten() for m0_j, indices_j in zip(m0, local_segment_indices_j)])
            nSegmentPairs = np.array([len(indices) for indices in local_segment_indices_i])
            k1 = np.cumsum(nSegmentPairs)
            k0 = np.concatenate(([0], k1[:-1]))
            k = np.concatenate([np.tile(np.arange(l0, l1), len(indices)) for l0, l1, indices in zip(k0, k1, vertical_indices)])
            h_ij[m, n, 1:] = h[k, :]
            h_ij[n, m, 1:] = (h.T * segments_i.H / segments_j.H).T[k, :]

        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for same-borehole thermal
        # interactions (inclined boreholes)
        # ---------------------------------------------------------------------
        if len(inclined_boreholes) > 0:
            segments_j, segments_i = \
                self._segment_pairs_same_borehole(
                    inclined_boreholes, inclined_nSegments,
                    segment_ratios=inclined_segment_ratios)
            h = finite_line_source(
                time, alpha, segments_j, segments_i, outer=False)
    
            # Local indices of segment pairs
            nInclinedBoreholes = len(inclined_boreholes)
            if isinstance(inclined_nSegments, (int, np.integer)):
                nSegments = np.broadcast_to(inclined_nSegments, nInclinedBoreholes)
            else:
                nSegments = inclined_nSegments
            local_segment_indices = \
                [np.triu_indices(nSegments[i]) for i in range(nInclinedBoreholes)]
            local_segment_indices_i = [indices[0] for indices in local_segment_indices]
            local_segment_indices_j = [indices[1] for indices in local_segment_indices]
            # Global indices of the first segments of boreholes
            m0 = [self._i0Segments[indices] for indices in inclined_indices]
            # Indices of the matrix h
            m = np.concatenate([np.add.outer(m0_i, indices_i).flatten() for m0_i, indices_i in zip(m0, local_segment_indices_i)])
            n = np.concatenate([np.add.outer(m0_j, indices_j).flatten() for m0_j, indices_j in zip(m0, local_segment_indices_j)])
            nSegmentPairs = np.array([len(indices) for indices in local_segment_indices_i])
            k1 = np.cumsum(nSegmentPairs)
            k0 = np.concatenate(([0], k1[:-1]))
            k = np.concatenate([np.tile(np.arange(l0, l1), len(indices)) for l0, l1, indices in zip(k0, k1, inclined_indices)])
            h_ij[m, n, 1:] = h[k, :]
            h_ij[n, m, 1:] = (h.T * segments_i.H / segments_j.H).T[k, :]

        # ---------------------------------------------------------------------
        # Segment-to-segment thermal response factors for borehole-to-borehole
        # thermal interactions
        # ---------------------------------------------------------------------
        for (i, j) in combinations_with_replacement(
                range(len(unique_boreholes)), 2):
            unique_borehole_indices_j = unique_borehole_indices[j]
            unique_borehole_indices_i = unique_borehole_indices[i]
            unique_borehole_j = unique_boreholes[j]
            unique_borehole_i = unique_boreholes[i]
            borefield_j = self.borefield[unique_borehole_indices_j]
            borefield_i = self.borefield[unique_borehole_indices_i]
            nBoreholes_j = len(borefield_j)
            nBoreholes_i = len(borefield_i)

            if isinstance(unique_nSegments, (int, np.integer)):
                nSegments_j = unique_nSegments
                nSegments_i = unique_nSegments
            else:
                nSegments_j = unique_nSegments[j]
                nSegments_i = unique_nSegments[i]

            if not isinstance(unique_segment_ratios, list):
                segment_ratios_j = unique_segment_ratios
                segment_ratios_i = unique_segment_ratios
            else:
                segment_ratios_j = unique_segment_ratios[j]
                segment_ratios_i = unique_segment_ratios[i]

            segments_j, segments_i = \
                self._segment_pairs(
                    unique_borehole_j, unique_borehole_i,
                    nSegments_j, nSegments_i,
                    segment_ratios_j=segment_ratios_j,
                    segment_ratios_i=segment_ratios_i,
                    to_self=i==j)
            if unique_borehole_i.is_vertical and unique_borehole_j.is_vertical:
                # -------------------------------------------------------------
                # Vertical boreholes
                # -------------------------------------------------------------
                unique_distances, distance_indices = \
                    self._find_unique_distances_vertical(
                        borefield_j, borefield_i, rtol=self.disTol,
                        to_self=i==j)
                h = finite_line_source_vertical(
                    time, alpha, segments_j, segments_i, distances=unique_distances, outer=i!=j)
                # Broadcast values to h_ij matrix
                if i == j and nBoreholes_i > 1:
                    # TODO : Diagonal elements are repeated
                    # Local indices of segment pairs
                    local_segment_indices_i, local_segment_indices_j = \
                        np.triu_indices(nSegments_i)
                    # Local indices of borehole pairs
                    local_borehole_indices_i, local_borehole_indices_j = \
                        np.triu_indices(nBoreholes_i, k=1)
                    # Global indices of the first segments of boreholes
                    m0 = self._i0Segments[unique_borehole_indices_i]
                    n0 = m0

                    # Upper triangle indices of thermal response factors of
                    # segments (n) of borehole (j) onto segments (m) of
                    # borehole (i)
                    m_u = np.add.outer(
                        local_segment_indices_i,
                        m0)[..., local_borehole_indices_i]
                    n_u = np.add.outer(
                        local_segment_indices_j,
                        n0)[..., local_borehole_indices_j]
                    k_u = distance_indices
                    # Upper triangle indices of thermal response factors of
                    # segments (n) of borehole (i) onto segments (m) of
                    # borehole (j)
                    m_l = np.add.outer(
                        local_segment_indices_i,
                        m0)[..., local_borehole_indices_j]
                    n_l = np.add.outer(
                        local_segment_indices_j,
                        n0)[..., local_borehole_indices_i]
                    k_l = distance_indices
                    # Concatenate indices
                    m = np.concatenate((m_u, m_l), axis=1)
                    n = np.concatenate((n_u, n_l), axis=1)
                    k = np.concatenate((k_u, k_l), axis=0)

                    # Assign h_ij matrix elements
                    h_ij[m, n, 1:] = h[:, k, :]
                    h_ij[n, m, 1:] = (h.T * segments_i.H / segments_j.H).T[:, k, :]
                elif i != j:
                    # Local indices of segment pairs
                    local_segment_indices_i, local_segment_indices_j = \
                        np.meshgrid(
                            np.arange(nSegments_i, dtype=int),
                            np.arange(nSegments_j, dtype=int),
                            indexing='ij')
                    # Local indices of borehole pairs
                    local_borehole_indices_i, local_borehole_indices_j = \
                        np.meshgrid(
                            np.arange(nBoreholes_i, dtype=int),
                            np.arange(nBoreholes_j, dtype=int),
                            indexing='ij')
                    local_borehole_indices_i = local_borehole_indices_i.flatten()
                    local_borehole_indices_j = local_borehole_indices_j.flatten()
                    # Global indices of the first segments of boreholes
                    m0 = self._i0Segments[unique_borehole_indices_i]
                    n0 = self._i0Segments[unique_borehole_indices_j]

                    m = np.add.outer(
                        local_segment_indices_i,
                        m0)[..., local_borehole_indices_i]
                    n = np.add.outer(
                        local_segment_indices_j,
                        n0)[..., local_borehole_indices_j]
                    k = distance_indices.flatten()

                    # Assign h_ij matrix elements
                    h_ij[m, n, 1:] = h[:, :, k, :]
                    h_ij[n, m, 1:] = (h.T * segments_i.H / segments_j.H[..., np.newaxis]).T[:, :, k, :]
        # Interp1d object for thermal response factors
        h_ij = interp1d(
            np.hstack((0., time)), h_ij,
            kind=kind, copy=True, assume_sorted=True, axis=2)

        toc = perf_counter()
        if self.disp:
            print(f' {toc - tic:.3f} sec')

        return h_ij

    @classmethod
    def _find_unique_distances_vertical(
            cls, borefield_j: Borefield, borefield_i: Borefield,
            rtol: float = 0.01, to_self: Union[None, bool] = None
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds unique distances (within tolerance) between boreholes of
        borefield (j) and boreholes of borefield (i).

        Parameters
        ----------
        borefield_j : Borefield object
            Borefield object of the boreholes extracting heat.
        borefield_i : Borefield object
            Borefield object of the boreholes for which borehole wall
            temperatures are to be evaluated.
        rtol : float, optional
            Relative tolerance on the distances for them to be considered
            equal.
            Default is 0.01.
        to_self : bool, optional
            True if borefield_j and borefield_i are the same borefield. In this
            case, a condensed array of distances between boreholes is
            evaluated, corresponding to the upper triangular (non-diagonal)
            elements of the borehole-to-borehole distance matrix. If false,
            the full borehole-to-borehole distance matrix is evaluated. If
            None, this parameter is evaluated by comparison between borefield_j
            and borefield_i.
            Default is None.

        Returns
        -------
        distances : (nDistances,) array
            Array of unique distances between boreholes (in meters).
        indices : array
            Indices of unique distances corresponding to each borehole pair. If
            to_self==True, a 1d array is returned, corresponding to the
            condensed (non-diagonal) array of borehole pairs. If
            to_self==False, a (nBoreholes_i, nBoreholes_j,) array is returned.

        """
        if to_self is None:
            to_self = borefield_j == borefield_i

        # Find all distances between the boreholes, sorted and flattened
        if to_self:
            distances = borefield_j.distance_to_self(outer=False)
            indices = np.zeros(
                int(len(borefield_j) * (len(borefield_j) - 1) / 2),
                dtype=int)
        else:
            distances = borefield_j.distance(borefield_i, outer=True).flatten()
            indices_i, indices_j = np.meshgrid(
                np.arange(len(borefield_i), dtype=int),
                np.arange(len(borefield_j), dtype=int),
                indexing='ij')
            indices_i = indices_i.flatten()
            indices_j = indices_j.flatten()
            indices = np.zeros(
                (len(borefield_i), len(borefield_j)),
                dtype=int)

        index_array = np.argsort(distances)
        sorted_distances = distances[index_array]
        nDis = len(distances)
        labels = np.zeros(nDis, dtype=int)

        # Find unique distances within tolerance
        unique_distances = []
        n = 0
        # Start the search at the first distance
        j0 = 0
        j1 = 1
        while j0 < nDis and j1 > 0:
            # Find the index of the first distance for which the distance is
            # outside tolerance to the current distance
            j1 = np.searchsorted(
                sorted_distances,
                (1 + rtol) * sorted_distances[j0])
            # Add the average of the distances within tolerance to the
            # list of unique distances and store the number of distances
            unique_distances.append(np.mean(sorted_distances[j0:j1]))
            if to_self:
                indices[index_array[j0:j1]] = n
            else:
                indices[indices_i[index_array[j0:j1]], indices_j[index_array[j0:j1]]] = n
            labels[j0:j1] = n
            n = n + 1
            j0 = j1
        return np.array(unique_distances), indices

    @classmethod
    def _find_unique_borehole_geometries(
            cls,
            borefield: Borefield,
            nSegments: Union[None, npt.ArrayLike] = None,
            segment_ratios: Union[None, Callable[[int], npt.ArrayLike], List[np.ndarray], np.ndarray] = None,
            rtol: float = 1e-6
            ) -> Tuple[
                Borefield,
                List[np.ndarray],
                Union[None, int, np.ndarray],
                Union[None, Callable, List[np.ndarray], np.ndarray]]:
        """
        Finds unique borehole geometries (H, D, r_b, tilt).

        Parameters
        ----------
        borefield : Borefield object
            The borefield.
        nSegments : int or (nBoreholes,) array of int, optional
            Number of segments per borehole. If nSegments is None, the number
            of segments is not used to compare boreholes.
            Default is None.
        segment_ratios : (nSegments,) array, (nBoreholes,) list of (nSegments_i,) arrays or callable, optional
            Segment ratios for the discretization along boreholes. If
            segment_ratios is None, the segment ratios are not used to
            compare boreholes.
            Default is None.
        rtol : float, optional
            Relative tolerance on geometric parameters under which they are
            considered equal.
            Default is 1e-6.

        Returns
        -------
        unique_boreholes : Borefield object
            Borefield object of unique borehole geometries, all located at
            the origin (x=0, y=0) and with orientation=0.
        unique_borehole_indices : list of arrays of int
            Indices of boreholesi in the borefied corresponding to each
            unique borehole geometry.
        unique_nSegments : None, int or (nUniqueBoreholes,) array of int
            Number of segments along each unique borehole geometry. None is
            returned if nSegments is None.
        unique_segment_ratios : (nSegments,) array, (nUniqueBoreholes,) list of (nSegments_i,) arrays or callable
            Segment ratios for the discretization along unique borehole
            geometries. None is returned if segment_ratios is None.

        """
        # Convert nSegments to array if list
        if isinstance(nSegments, list):
            np.asarray(nSegments, dtype=int)
        # All remaining boreholes in borefield
        remaining_borehole_indices = np.arange(
            len(borefield),
            dtype=int)
        remaining_boreholes = borefield[:]
        # Fin unique borehole geometries
        unique_borehole_indices = []
        n = 0
        while len(remaining_borehole_indices) > 0:
            # Compare all remaining boreholes to the first remaining borehole
            reference_borehole = remaining_boreholes[0]
            reference_borehole_index = remaining_borehole_indices[0]
            # Geometric parameters
            similar_boreholes = np.all(
                np.stack(
                    [np.abs(remaining_boreholes.H - reference_borehole.H) < rtol * reference_borehole.H,
                     np.abs(remaining_boreholes.D - reference_borehole.D) < rtol * reference_borehole.D,
                     np.abs(remaining_boreholes.r_b - reference_borehole.r_b) < rtol * reference_borehole.r_b,
                     np.abs(remaining_boreholes.tilt - reference_borehole.tilt) < rtol * reference_borehole.tilt + 1e-12
                        ],
                    axis=0),
                axis=0
                )
            # Also compare nSegments if it was provided as a list or an array
            if isinstance(nSegments, np.ndarray):
                similar_boreholes = np.logical_and(
                    similar_boreholes,
                    np.equal(
                        nSegments[remaining_borehole_indices],
                        nSegments[reference_borehole_index])
                    )
            # Also compare segment_ratios if it was provided as a list
            if isinstance(segment_ratios, list):
                similar_boreholes = np.logical_and(
                    similar_boreholes,
                    [len(segment_ratios[i]) == len(segment_ratios[reference_borehole_index])
                     and np.allclose(
                         segment_ratios[i],
                         segment_ratios[reference_borehole_index],
                         rtol=rtol
                         )
                     for i in remaining_borehole_indices]
                    )
            # Create a unique borehole for all found similar boreholes
            unique_borehole_indices.append(remaining_borehole_indices[similar_boreholes])
            # Remove them from the remaining boreholes
            remaining_borehole_indices = remaining_borehole_indices[~similar_boreholes]
            remaining_boreholes = remaining_boreholes[~similar_boreholes]
            n = n + 1
        # Create a Borefield object from the unqiue borehole geometries
        m = np.array(
            [indices[0] for indices in unique_borehole_indices],
            dtype=int)
        unique_boreholes = Borefield(
            borefield.H[m], borefield.D[m], borefield.r_b[m], 0., 0., tilt=borefield.tilt[m])
        # Only return an array of nSegments if a list or array was provided
        if isinstance(nSegments, np.ndarray):
            unique_nSegments = nSegments[m]
        else:
            unique_nSegments = nSegments
        # Only return a list of segment_ratios if a list was provided
        if isinstance(segment_ratios, list):
            unique_segment_ratios = [segment_ratios[n] for n in m]
        else:
            unique_segment_ratios = segment_ratios
        return unique_boreholes, unique_borehole_indices, unique_nSegments, unique_segment_ratios

    @classmethod
    def _segment_pairs_same_borehole(
            cls,
            borefield: Borefield,
            nSegments: npt.ArrayLike,
            segment_ratios: Union[None, Callable[[int], npt.ArrayLike], List[np.ndarray], np.ndarray] = None
            ) -> Tuple[Borefield, Borefield]:
        """
        Returns condensed borefields for all non-repeated pairs of segments
        along boreholes of a borefield and themselves.

        Parameters
        ----------
        borefield : Borefield object
            The borefield.
        nSegments : int or (nBoreholes,) array of int
            Unmber of segments per borehole.
        segment_ratios : array, list of arrays, or callable, optional
            Ratio of the borehole length represented by each segment. The
            sum of ratios must be equal to 1. The shape of the array is of
            (nSegments,) or list of (nSegments[i],). If segment_ratios==None,
            segments of equal lengths are considered. If a callable is
            provided, it must return an array of size (nSegments,) when
            provided with nSegments (of type int) as an argument, or an array
            of size (nSegments[i],) when provided with an element of nSegments
            (of type list).
            Default is None.

        Returns
        -------
        segments_j : Borefield object
            Borefield object of segments extracting heat.
        segments_i : Borefield object
            Borefield object of segments where the temperature is evaluated.

        """
        # Segment the borefield
        segments = borefield.segments(nSegments, segment_ratios=segment_ratios)
        # Indices of list ranges of first and last segment along boreholes
        n1 = np.cumsum(
            np.broadcast_to(nSegments, len(borefield)),
            dtype=int)
        n0 = np.concatenate(
            ([0], n1[:-1]),
            dtype=int)
        # Condensed arrays of indices of segment pairs
        i = [np.arange(m0, m1)[np.triu_indices(m1 - m0)[0]] for m0, m1 in zip(n0, n1)]
        j = [np.arange(m0, m1)[np.triu_indices(m1 - m0)[1]] for m0, m1 in zip(n0, n1)]
        indices_j = np.concatenate(j)
        indices_i = np.concatenate(i)
        # Expand segments into Borefield objects of segment pairs
        segments_j = segments[indices_j]
        segments_i = segments[indices_i]
        return segments_j, segments_i

    @classmethod
    def _segment_pairs(
            cls,
            borehole_j: Borehole,
            borehole_i: Borehole,
            nSegments_j: int,
            nSegments_i: int,
            segment_ratios_j: Union[None, Callable[[int], npt.ArrayLike], np.ndarray] = None,
            segment_ratios_i: Union[None, Callable[[int], npt.ArrayLike], np.ndarray] = None,
            to_self: bool = False) -> Tuple[Borefield, Borefield]:
        """
        Returns borefields of segments for the evaluation of segment-to-segment
        thermal response factors.

        Parameters
        ----------
        borehole_j : Borehole object
            The borehole extracting heat.
        borehole_i : Borehole object
            The borehole where temperatures are evaluated.
        nSegments_j : int
            Number of segments along borehole_j.
        nSegments_i : int
            Number of segments along borehole_i.
        segment_ratios_j : array, list of arrays, or callable, optional
            Ratio of the borehole length represented by each segment of
            borehole_j. The sum of ratios must be equal to 1. The shape of the
            array is of (nSegments_j,) or list of (nSegments[j],). If
            segment_ratios is None, segments of equal lengths are considered.
            If a callable is provided, it must return an array of size
            (nSegments,) when provided with nSegments (of type int) as an
            argument, or an array of size (nSegments[j],) when provided with an
            element of nSegments (of type list).
            Default is None.
        segment_ratios_i : array, list of arrays, or callable, optional
            Ratio of the borehole length represented by each segment of
            borehole_i.
        to_self : bool, optional
            True if segment pairs are created for the interaction between a
            borehole and itself, in which case the method returns condensed
            borefields for all non-repeated pairs of segments along the
            borehole and itself. If False, the returned Borefield objects
            are of lengths (nSegment_j,) and (nSegments_i,).
            Default is False.

        Returns
        -------
        segments_j: Borefield object
            Borefield object of segments extracting heat.
        segments_i: Borefield object
            Borefield object of segments where the temperature is evaluated.

        """
        # Segment boreholes
        segments_j = Borefield.from_boreholes(
            borehole_j.segments(
                nSegments_j,
                segment_ratios=segment_ratios_j))
        segments_i = Borefield.from_boreholes(
            borehole_i.segments(
                nSegments_i,
                segment_ratios=segment_ratios_i))

        # Create condensed Borefield objects for non-repeated segment pairs
        if to_self:
            i, j = np.triu_indices(nSegments_j, k=0)
            segments_j = segments_j[j]
            segments_i = segments_j[i]
        return segments_j, segments_i

    @classmethod
    def _split_vertical_and_inclined_boreholes(
            cls,
            borefield: Borefield,
            nSegments: Union[None, npt.ArrayLike] = None,
            segment_ratios: Union[None, Callable[[int], npt.ArrayLike], List[np.ndarray], np.ndarray] = None,
            indices: Union[None, List[np.ndarray]] = None) -> Tuple[
                Borefield,
                Union[None, int, np.ndarray],
                Union[None, Callable, List[np.ndarray], np.ndarray],
                List[np.ndarray],
                Union[None, int, np.ndarray],
                Union[None, Callable, List[np.ndarray], np.ndarray],
                List[np.ndarray]]:
        """
        Splits a borefield into a borefield of vertical boreholes and a
        borefield of inclined boreholes.

        Parameters
        ----------
        borefield : Borefield object
            The borefield.
        nSegments : int or (nBoreholes,) array of int, optional
            Number of segments along boreholes. If nSegments is None,
            None is returned for vertical_nSegments and inclined_nSegments.
            The default is None.
        segment_ratios : array, list of arrays, or callable, optional
            Ratio of the borehole length represented by each segment of
            the boreholes. The sum of ratios must be equal to 1. The shape of
            the array is of (nSegments,) or list of (nSegments[i],). If
            segment_ratios is None, segments of equal lengths are considered.
            If a callable is provided, it must return an array of size
            (nSegments,) when provided with nSegments (of type int) as an
            argument, or an array of size (nSegments[i],) when provided with an
            element of nSegments (of type list).
            Default is None.
        indices : (nBoreholes,) list of arrays of int, optional
            Arrays of indices corresponding to each borehole in Borefield. If
            indices is None, None is returned for vertical_indices and
            inclined_indices.
            Default is None.

        Returns
        -------
        vertical_boreholes : Borefield object
            Vertical boreholes in the borefield.
        vertical_nSegments : int or array of int
            Number of segments per vertical borehole.
        vertical_segment_ratios : (vertical_nSegments,) array, (nVerticalBoreholes,) list of (vertical_nSegments_i,) arrays or callable
            Segment ratios for the discretization along vertical boreholes
            geometries. None is returned if segment_ratios is None.
        vertical_indices : (nVerticalBoreholes,) list of  arrays of int
            Indices of the vertical boreholes.
        inclined_boreholes : Borefield object
            Inclined boreholes in the borefield.
        inclined_nSegments : int or array of int
            Number of segments per inclined borehole.
        inclined_segment_ratios : (inclined_nSegments,) array, (nInclinedBoreholes,) list of (inclined_nSegments_i,) arrays or callable
            Segment ratios for the discretization along inclined boreholes
            geometries. None is returned if segment_ratios is None.
        inclined_indices : (nInclinedBoreholes,) list of  arrays of int
            Indices of the inclined boreholes.

        """
        # Find vertical and inclined boreholes
        vertical_indices = np.arange(
            len(borefield), dtype=int)[borefield.is_vertical]
        vertical_boreholes = borefield[vertical_indices]
        inclined_indices = np.arange(
            len(borefield), dtype=int)[borefield.is_tilted]
        inclined_boreholes = borefield[inclined_indices]
        # Return nSegments as None or int if provided as such
        if nSegments is None or isinstance(nSegments, (int, np.integer)):
            vertical_nSegments = nSegments
            inclined_nSegments = nSegments
        else:
            vertical_nSegments = np.asarray(nSegments, dtype=int)[vertical_indices]
            inclined_nSegments = np.asarray(nSegments, dtype=int)[inclined_indices]
        # Only return segment_ratios as list if a list was provided
        if not isinstance(segment_ratios, list):
            vertical_segment_ratios = segment_ratios
            inclined_segment_ratios = segment_ratios
        else:
            vertical_segment_ratios = [segment_ratios[i] for i in vertical_indices]
            inclined_segment_ratios = [segment_ratios[i] for i in inclined_indices]
        # If a list of arrays of indices was provided, create lists for
        # vertical and inclined boreholes
        if isinstance(indices, list):
            vertical_indices = [indices[m] for m in vertical_indices]
            inclined_indices = [indices[m] for m in inclined_indices]
        return vertical_boreholes, vertical_nSegments, vertical_segment_ratios, \
            vertical_indices, inclined_boreholes, inclined_nSegments, \
            inclined_segment_ratios, inclined_indices

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
